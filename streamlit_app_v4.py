# =============================================================================
# DRIVER SAFETY ANALYTICS PLATFORM  —  v4.0.0
# =============================================================================
# NEW in v4:
#   • Chunk-based real-time streaming inference (no more full-video buffering)
#   • Lighting normalization (CLAHE / gamma) for sunlight / night robustness
#   • Live annotated frame display updated after every chunk
#   • AlertManager: real-time alerts with cooldown + 5-level escalation logic
#   • RollingBuffer: rolling temporal state carried across chunks
#   • Confidence score shown on every alert
#   • Colour-coded Driver Timeline bar (HTML)
#   • REALTIME_ALERTS Snowflake table written incrementally
#   • Full post-inference analytics with 9 enhanced tabs
# =============================================================================
# SNOWFLAKE SCHEMA CHANGES REQUIRED (run before deploying):
#   -- New table:
#   CREATE TABLE IF NOT EXISTS DEMO_DB.PUBLIC.REALTIME_ALERTS (
#       ALERT_ID VARCHAR(64), SESSION_ID VARCHAR(64), DRIVER_ID VARCHAR(64),
#       TRIP_ID VARCHAR(64), ALERT_TIMESTAMP_SECONDS FLOAT,
#       ALERT_WALL_TIME VARCHAR(64), ALERT_TYPE VARCHAR(64),
#       SEVERITY VARCHAR(16), SEVERITY_SCORE FLOAT, ESCALATION_LEVEL INT,
#       FATIGUE_SCORE FLOAT, OFFROAD_PROB FLOAT, ZONE_AT_ALERT VARCHAR(32),
#       CONFIDENCE FLOAT, MESSAGE VARCHAR(512),
#       CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
#   );
#   -- New columns on existing tables:
#   ALTER TABLE DEMO_DB.PUBLIC.MODULE_A_FRAME_PREDICTIONS
#       ADD COLUMN IF NOT EXISTS LIGHTING_METHOD VARCHAR(32);
#   ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
#       ADD COLUMN IF NOT EXISTS MEAN_MODEL_CONFIDENCE FLOAT;
#   ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
#       ADD COLUMN IF NOT EXISTS TOTAL_REALTIME_ALERTS INT;
# =============================================================================

import io
import os
import tempfile
import time
import uuid
import hashlib
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ─── optional dependencies ────────────────────────────────────────────────────
try:
    import torch, torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    from snowflake.snowpark.context import get_active_session
    _session = get_active_session()
except Exception:
    try:
        from snowflake.snowpark import Session
        _session = Session.builder.config("connection_name", "default").create()
    except Exception:
        _session = None

# ─── constants ─────────────────────────────────────────────────────────────────
DATABASE        = "DEMO_DB"
SCHEMA          = "PUBLIC"
APP_VERSION     = "4.0.0"
MODEL_VER_A1    = "a1_fold1.pt"
MODEL_VER_A2    = "a2_fold1.pt"
MODEL_VER_B     = "best_lisa_fold5_calibrated.pt"
_MODEL_DIR      = tempfile.mkdtemp()

ZONE_CLASSES = ["Forward","Lap","Left Mirror","Radio","Rearview","Right Mirror","Shoulder","Speedometer"]
RISK_GROUPS  = {"Safe": ["Forward"], "LowRisk": ["Left Mirror","Right Mirror","Rearview"],
                "HighRisk": ["Lap","Radio","Speedometer","Shoulder"]}

OFFROAD_THRESHOLD      = 0.35
ZONE_TEMPERATURE       = 2.600605
NORM_MEAN              = [0.485, 0.456, 0.406]
NORM_STD               = [0.229, 0.224, 0.225]
IMG_SIZE               = 224
MIN_ALERT_CONF         = 0.45
MAX_UNCERTAIN_ENTROPY  = 1.55
MIN_FRAME_QUALITY      = 0.35
EAR_CLOSED_THRESHOLD   = 0.21
POOR_QUALITY_RATIO     = 0.35
OFFROAD_EXIT_HYSTERESIS= 0.08
CHUNK_SECONDS          = 5         # seconds of video per processing chunk
ALERT_COOLDOWN_S       = 25        # seconds between same-type alerts
DISPLAY_WIDTH          = 620       # px for annotated frame display

MP_LEFT_EYE  = [33,  160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

DEFAULT_A = dict(w_drowsy=0.30, w_eye_closed=0.30, w_nod=0.15, w_yawn=0.10,
                 w_perclos=0.10, w_blink=0.05, t_eye=0.30, t_caution=0.40,
                 t_alert=0.70, closure_frames=5, fps=5, win_s=10, win_m=30,
                 win_l=60, ear_closed=EAR_CLOSED_THRESHOLD)

STATE_COLORS = {"normal_driving":"#22c55e","mild_fatigue":"#f59e0b",
                "high_fatigue":"#ef4444","offroad_glance":"#f97316",
                "repeated_distraction":"#dc2626","prolonged_eye_closure":"#b91c1c",
                "needs_review":"#6b7280"}

ESC_LABELS = ["Normal ✅","Low ⚠️","Medium 🟠","High 🔴","Critical 🆘"]
ESC_HEX    = ["#22c55e","#f59e0b","#f97316","#ef4444","#7f1d1d"]

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
if TORCH_AVAILABLE:
    class _A1Model(nn.Module):
        def __init__(self):
            super().__init__()
            import torchvision.models as m
            bb = m.resnet18(weights=None); nf = bb.fc.in_features; bb.fc = nn.Identity()
            self.bb = bb
            self.h_d = nn.Linear(nf, 1); self.h_y = nn.Linear(nf, 1); self.h_n = nn.Linear(nf, 1)
        def forward(self, x):
            f = self.bb(x)
            return {"drowsy": torch.sigmoid(self.h_d(f)).squeeze(-1),
                    "yawn":   torch.sigmoid(self.h_y(f)).squeeze(-1),
                    "nod":    torch.sigmoid(self.h_n(f)).squeeze(-1)}

    class _A2Model(nn.Module):
        def __init__(self):
            super().__init__()
            import torchvision.models as m
            bb = m.resnet18(weights=None); nf = bb.fc.in_features; bb.fc = nn.Identity()
            self.bb = bb; self.h_e = nn.Linear(nf, 1)
        def forward(self, x):
            f = self.bb(x)
            return {"eye_closed": torch.sigmoid(self.h_e(f)).squeeze(-1)}

    class _BModel(nn.Module):
        def __init__(self, bb, nf):
            super().__init__()
            self.bb = bb
            self.zone_head    = nn.Linear(nf, 8)
            self.offroad_head = nn.Linear(nf, 1)
        def forward(self, x):
            f = self.bb(x)
            return {"zone_logits": self.zone_head(f), "offroad_logit": self.offroad_head(f)}


@st.cache_resource
def load_a1():
    if not TORCH_AVAILABLE: return None
    p = _download_model(MODEL_VER_A1)
    if not p: return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, nn.Module): ckpt.eval(); return ckpt
        mdl = _A1Model()
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        mdl.load_state_dict(sd, strict=False); mdl.eval(); return mdl
    except Exception: return None

@st.cache_resource
def load_a2():
    if not TORCH_AVAILABLE: return None
    p = _download_model(MODEL_VER_A2)
    if not p: return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, nn.Module): ckpt.eval(); return ckpt
        mdl = _A2Model()
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        mdl.load_state_dict(sd, strict=False); mdl.eval(); return mdl
    except Exception: return None

@st.cache_resource
def load_b():
    if not TORCH_AVAILABLE: return None
    p = _download_model(MODEL_VER_B)
    if not p: return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, nn.Module): ckpt.eval(); return ckpt
        try:
            import timm
            bb = timm.create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=0)
            nf = bb.num_features
        except Exception:
            import torchvision.models as tm
            bb = tm.efficientnet_b0(weights=None); nf = bb.classifier[1].in_features; bb.classifier = nn.Identity()
        mdl = _BModel(bb, nf)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        mdl.load_state_dict(sd, strict=False); mdl.eval(); return mdl
    except Exception: return None

# =============================================================================
# SNOWFLAKE UTILITIES
# =============================================================================
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def get_session():
    return _session

def _download_model(filename: str):
    path = os.path.join(_MODEL_DIR, filename)
    if os.path.exists(path): return path
    s = get_session()
    if s is None: return None
    try:
        s.file.get(f"@{DATABASE}.{SCHEMA}.DRIVER_SAFETY_MODELS/{filename}", _MODEL_DIR)
        if os.path.exists(path): return path
        gz = path + ".gz"
        if os.path.exists(gz):
            import gzip, shutil
            with gzip.open(gz,"rb") as fi, open(path,"wb") as fo: shutil.copyfileobj(fi,fo)
            os.remove(gz); return path
    except Exception: pass
    return None

def write_to_snowflake(df: pd.DataFrame, table_name: str, retries: int = 2):
    s = get_session()
    if s is None or df.empty: return False
    up = df.copy()
    if "created_at_client" not in up.columns: up["created_at_client"] = utc_now_iso()
    up.columns = [c.upper() for c in up.columns]
    for attempt in range(retries + 1):
        try:
            s.write_pandas(up, table_name, database=DATABASE, schema=SCHEMA, overwrite=False)
            return True
        except Exception as e:
            if attempt == retries:
                st.toast(f"DB write warning ({table_name}): {e}", icon=":material/warning:")
                return False
            time.sleep(0.3 * (attempt + 1))
    return False

def query_sf(sql: str) -> pd.DataFrame:
    s = get_session()
    if s is None: return pd.DataFrame()
    try: return s.sql(sql).to_pandas()
    except Exception: return pd.DataFrame()

def fetch_recent(table: str, limit: int = 200) -> pd.DataFrame:
    for order in ["CREATED_AT DESC", "CREATED_AT_CLIENT DESC", ""]:
        clause = f"ORDER BY {order}" if order else ""
        df = query_sf(f"SELECT * FROM {DATABASE}.{SCHEMA}.{table} {clause} LIMIT {limit}")
        if not df.empty: return df
    return pd.DataFrame()

# =============================================================================
# GENERAL UTILITIES
# =============================================================================
def _clamp01(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)): return float(default)
        return float(np.clip(v, 0.0, 1.0))
    except Exception: return float(default)

def _make_rng(source_name: str, frame_idx: int):
    key = f"{source_name}:{frame_idx}".encode()
    return np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:8], 16))

def get_risk_group(zone: str) -> str:
    for g, zones in RISK_GROUPS.items():
        if zone in zones: return g
    return "Unknown"

def _certainty_from_prob(p):
    if p is None: return 0.0
    return float(np.clip(abs(p - 0.5) * 2.0, 0.0, 1.0))

def _severity_bucket(score, duration):
    c = float(score + min(duration / 4.0, 1.0) * 0.25)
    if c >= 0.9: return "critical", min(1.0, c)
    if c >= 0.7: return "high",     c
    if c >= 0.45: return "medium",  c
    return "low", c

def _alert_type_from_severity(sev):
    return {"critical":"dashboard_critical","high":"dashboard_alert",
            "medium":"dashboard_warning","low":"none"}.get(sev,"none")

# =============================================================================
# LIGHTING NORMALIZATION  (NEW in v4)
# =============================================================================
def normalize_lighting(frame_bgr):
    """
    Adaptive pre-processing for robust inference under all lighting conditions.
    Returns (corrected_frame, method_label).
    Handles: sunlight glare (overexposed), night/tunnel (underexposed),
    flat overcast (low contrast), and normal conditions.
    """
    if not CV2_AVAILABLE or frame_bgr is None:
        return frame_bgr, "none"
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        std_b  = float(np.std(gray))

        if mean_b > 175:          # ── Overexposed / harsh sunlight
            # Gamma compress highlights (gamma > 1)
            gamma  = 1.8
            lut    = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in range(256)], dtype=np.uint8)
            out    = cv2.LUT(frame_bgr, lut)
            # CLAHE on V-channel of HSV to recover facial texture
            hsv    = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:,:,2] = clahe.apply(hsv[:,:,2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "overexposed_correction"

        elif mean_b < 55:          # ── Underexposed / night
            lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), "underexposed_correction"

        elif std_b < 22:           # ── Low contrast / overcast / fog
            lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), "low_contrast_correction"

        else:
            return frame_bgr, "normal"
    except Exception:
        return frame_bgr, "error"

# =============================================================================
# FACE + EYE DETECTION
# =============================================================================
@st.cache_resource
def get_face_mesh():
    if not MP_AVAILABLE: return None
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def _brightness_score(gray):
    if gray is None or gray.size == 0: return 0.0
    return float(np.clip(1.0 - abs(float(np.mean(gray)) - 128.0) / 128.0, 0.0, 1.0))

def _blur_score(gray):
    if gray is None or gray.size == 0: return 0.0
    return float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0, 0.0, 1.0))

def _safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1,y1 = max(0,int(x1)), max(0,int(y1))
    x2,y2 = min(w,int(x2)), min(h,int(y2))
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2]

def _lm_xy(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x*w, p.y*h], dtype=np.float32)

def _ear(pts):
    p1,p2,p3,p4,p5,p6 = pts
    return float((np.linalg.norm(p2-p6)+np.linalg.norm(p3-p5))/(2.0*np.linalg.norm(p1-p4)+1e-6))

def _crop_pts(frame, pts, pad=0.28):
    xs,ys = pts[:,0], pts[:,1]
    x1,x2,y1,y2 = xs.min(),xs.max(),ys.min(),ys.max()
    w,h = x2-x1, y2-y1
    return _safe_crop(frame, x1-w*pad, y1-h*pad, x2+w*pad, y2+h*pad)

def _head_pose(lm, shape):
    h,w = shape[:2]
    ip = np.array([_lm_xy(lm,1,w,h),_lm_xy(lm,152,w,h),_lm_xy(lm,33,w,h),
                   _lm_xy(lm,263,w,h),_lm_xy(lm,61,w,h),_lm_xy(lm,291,w,h)],dtype=np.float64)
    mp3 = np.array([(0,0,0),(0,-63.6,-12.5),(-43.3,32.7,-26.0),
                    (43.3,32.7,-26.0),(-28.9,-28.9,-24.1),(28.9,-28.9,-24.1)],dtype=np.float64)
    cm = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(mp3,ip,cm,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return None,None,None
        rm,_ = cv2.Rodrigues(rvec)
        _,_,_,_,_,_,euler = cv2.decomposeProjectionMatrix(np.hstack((rm,tvec)))
        return float(euler[0]),float(euler[1]),float(euler[2])
    except Exception: return None,None,None

def detect_face_eyes(frame_bgr):
    """Extract face ROI, eye strip, EAR, and head pose from a BGR frame."""
    default = dict(face=None,eye_strip=None,left_eye=None,right_eye=None,
                   face_detected=False,preprocess_method="none",quality_score=0.0,
                   face_confidence=0.0,brightness_score=0.0,blur_score=0.0,
                   ear=None,left_ear=None,right_ear=None,
                   head_pitch=None,head_yaw=None,head_roll=None)
    if not CV2_AVAILABLE: return default

    h,w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = _brightness_score(gray)
    blur = _blur_score(gray)
    mesh = get_face_mesh()

    if MP_AVAILABLE and mesh is not None:
        try:
            res = mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pts = np.array([[p.x*w, p.y*h] for p in lm], dtype=np.float32)
                x1,y1 = pts.min(axis=0);  x2,y2 = pts.max(axis=0)
                face = _safe_crop(frame_bgr, x1-12, y1-12, x2+12, y2+12)

                lp = np.array([_lm_xy(lm,i,w,h) for i in MP_LEFT_EYE],  dtype=np.float32)
                rp = np.array([_lm_xy(lm,i,w,h) for i in MP_RIGHT_EYE], dtype=np.float32)
                le = _crop_pts(frame_bgr, lp, 0.45)
                re = _crop_pts(frame_bgr, rp, 0.45)

                eye_strip = None
                if le is not None and re is not None:
                    mh = max(le.shape[0], re.shape[0])
                    if le.shape[0]!=mh: le = cv2.resize(le,(le.shape[1],mh))
                    if re.shape[0]!=mh: re = cv2.resize(re,(re.shape[1],mh))
                    eye_strip = cv2.hconcat([le, re])
                elif le is not None: eye_strip = le
                elif re is not None: eye_strip = re

                l_ear = _ear(lp); r_ear = _ear(rp); ear = (l_ear+r_ear)/2.0
                pitch,yaw,roll = _head_pose(lm, frame_bgr.shape)
                area = float(np.clip((x2-x1)*(y2-y1)/(w*h), 0.0, 1.0))
                quality = float(np.clip(0.40*blur+0.30*brightness+0.30*min(area/0.2,1.0), 0.0, 1.0))
                return dict(face=face,eye_strip=eye_strip,left_eye=le,right_eye=re,
                            face_detected=face is not None,preprocess_method="mediapipe",
                            quality_score=quality,face_confidence=min(1.0,0.5+quality/2.0),
                            brightness_score=brightness,blur_score=blur,
                            ear=float(ear),left_ear=float(l_ear),right_ear=float(r_ear),
                            head_pitch=pitch,head_yaw=yaw,head_roll=roll)
        except Exception: pass

    # Haar cascade fallback
    try:
        cp = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
        if os.path.exists(cp):
            fc = cv2.CascadeClassifier(cp)
            faces = fc.detectMultiScale(gray, 1.3, 5)
            if len(faces)>0:
                x,y,fw,fh = faces[0]; face = frame_bgr[y:y+fh, x:x+fw]
                strip = face[int(fh*0.15):int(fh*0.45), :]
                ar = float(np.clip((fw*fh)/(w*h), 0.0, 1.0))
                q = float(np.clip(0.45*blur+0.30*brightness+0.25*min(ar/0.2,1.0), 0.0, 1.0))
                return dict(face=face,eye_strip=strip,left_eye=None,right_eye=None,
                            face_detected=True,preprocess_method="haar",
                            quality_score=q,face_confidence=min(0.8,0.35+q/2.0),
                            brightness_score=brightness,blur_score=blur,
                            ear=None,left_ear=None,right_ear=None,
                            head_pitch=None,head_yaw=None,head_roll=None)
    except Exception: pass

    # Center-crop last resort
    cx,cy = w//2, h//2; fw,fh = int(w*0.5),int(h*0.6)
    x1 = max(0,cx-fw//2); y1 = max(0,cy-fh//2)
    face  = frame_bgr[y1:y1+fh, x1:x1+fw]
    strip = face[int(fh*0.15):int(fh*0.45), :]
    q = float(np.clip(0.35*blur+0.35*brightness+0.15, 0.0, 0.55))
    return dict(face=face,eye_strip=strip,left_eye=None,right_eye=None,face_detected=False,
                preprocess_method="center_crop",quality_score=q,face_confidence=0.20,
                brightness_score=brightness,blur_score=blur,
                ear=None,left_ear=None,right_ear=None,head_pitch=None,head_yaw=None,head_roll=None)

# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================
def _prep_tensor(img_pil, size=224):
    t = transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
                             transforms.Normalize(NORM_MEAN,NORM_STD)])
    return t(img_pil).unsqueeze(0)

def infer_a1(mdl, face):
    if mdl is None or face is None or not PIL_AVAILABLE or not CV2_AVAILABLE: return None
    try:
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        with torch.no_grad(): o = mdl(_prep_tensor(img))
        d = _clamp01(float(o["drowsy"].item()))
        y = _clamp01(float(o["yawn"].item()))
        n = _clamp01(float(o["nod"].item()))
        return {"a1_prob_drowsy":d,"a1_prob_yawn":y,"a1_prob_nod":n,
                "a1_confidence":float(np.mean([_certainty_from_prob(d),_certainty_from_prob(y),_certainty_from_prob(n)]))}
    except Exception: return None

def infer_a2(mdl, strip):
    if mdl is None or strip is None or not PIL_AVAILABLE or not CV2_AVAILABLE: return None
    try:
        img = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
        t = transforms.Compose([transforms.Resize((64,224)),transforms.ToTensor(),
                                 transforms.Normalize(NORM_MEAN,NORM_STD)])
        with torch.no_grad(): o = mdl(t(img).unsqueeze(0))
        e = _clamp01(float(o["eye_closed"].item()))
        return {"a2_prob_eye_closed":e,"a2_eye_openness_score":_clamp01(1.0-e),
                "a2_confidence":_certainty_from_prob(e)}
    except Exception: return None

def infer_b(mdl, frame, thr=OFFROAD_THRESHOLD):
    if mdl is None or not PIL_AVAILABLE or not CV2_AVAILABLE: return None
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad(): o = mdl(_prep_tensor(img, IMG_SIZE))
        zl  = o["zone_logits"][0] / ZONE_TEMPERATURE
        zp  = torch.softmax(zl,dim=0).detach().cpu().numpy()
        si  = np.argsort(zp)[::-1]; sp = zp[si]
        zi  = int(si[0])
        op  = _clamp01(float(torch.sigmoid(o["offroad_logit"][0]).item()))
        ent = float(-np.sum(zp*np.log(zp+1e-8)))
        conf= _clamp01(float(zp.max()))
        mar = float(sp[0]-sp[1]) if len(sp)>1 else conf
        unc = float(np.clip((ent/np.log(len(ZONE_CLASSES)))*0.6+(1.0-conf)*0.4, 0.0, 1.0))
        return {"zone_pred":ZONE_CLASSES[zi],"risk_group_pred":get_risk_group(ZONE_CLASSES[zi]),
                "offroad_prob":op,"offroad_pred":1 if op>=thr else 0,
                "confidence":conf,"entropy":ent,"margin":mar,"uncertainty_score":unc,
                "zone_top2":" | ".join([ZONE_CLASSES[int(i)] for i in si[:2]]),
                **{f"zone_prob_{z.lower().replace(' ','_')}":float(zp[i]) for i,z in enumerate(ZONE_CLASSES)}}
    except Exception: return None

# =============================================================================
# SIMULATION FALLBACKS  (deterministic per-frame)
# =============================================================================
def sim_a(rng):
    ec = float(rng.beta(2,5)); d = float(rng.beta(2,5))
    y = float(rng.beta(1.5,8)); n = float(rng.beta(1.5,6))
    return {"a1_prob_drowsy":d,"a1_prob_yawn":y,"a1_prob_nod":n,
            "a2_prob_eye_closed":ec,"a2_eye_openness_score":float(1.0-ec),
            "a1_confidence":float(rng.uniform(0.45,0.85)),
            "a2_confidence":float(rng.uniform(0.45,0.85))}

def sim_b(rng, thr=OFFROAD_THRESHOLD):
    pr = rng.dirichlet([5,1,1.5,0.8,1.5,1.5,0.5,0.8])
    zi = int(np.argmax(pr)); zone = ZONE_CLASSES[zi]
    op = float(np.clip(1-pr[0]+rng.normal(0,0.05), 0, 1))
    conf=float(pr.max()); ent=float(-np.sum(pr*np.log(pr+1e-8)))
    si=np.argsort(pr)[::-1]; sp=pr[si]
    mar=float(sp[0]-sp[1]); unc=float(np.clip((ent/np.log(len(ZONE_CLASSES)))*0.6+(1-conf)*0.4,0,1))
    return {"zone_pred":zone,"risk_group_pred":get_risk_group(zone),
            "offroad_prob":op,"offroad_pred":1 if op>=thr else 0,
            "confidence":conf,"entropy":ent,"margin":mar,"uncertainty_score":unc,
            "zone_top2":" | ".join([ZONE_CLASSES[int(i)] for i in si[:2]]),
            **{f"zone_prob_{z.lower().replace(' ','_')}":float(pr[i]) for i,z in enumerate(ZONE_CLASSES)}}

# =============================================================================
# TEMPORAL ANALYSIS HELPERS
# =============================================================================
def _streak_events(binary, fps, min_dur, label):
    events=[]; start=None
    for i,v in enumerate(binary):
        if int(v)==1:
            if start is None: start=i
        else:
            if start is not None:
                dur=(i-start)/fps
                if dur>=min_dur: events.append({"type":label,"start":start/fps,"end":(i-1)/fps,"dur":dur})
                start=None
    if start is not None:
        dur=(len(binary)-start)/fps
        if dur>=min_dur: events.append({"type":label,"start":start/fps,"end":(len(binary)-1)/fps,"dur":dur})
    return events

def _derive_blinks(df, fps):
    blink_events=[]; blink_binary=[]
    min_f=max(1,int(0.08*fps)); max_f=max(min_f+1,int(0.8*fps))
    src=df["eye_closed_binary"].fillna(0).astype(int).tolist(); start=None
    for i,v in enumerate(src):
        if v==1 and start is None: start=i
        elif v==0 and start is not None:
            length=i-start
            if min_f<=length<=max_f:
                blink_events.append({"type":"blink","start":start/fps,"end":(i-1)/fps,"dur":length/fps})
                blink_binary.extend([1]*length)
            else: blink_binary.extend([0]*length)
            start=None
        elif v==0: blink_binary.append(0)
    if start is not None:
        length=len(src)-start
        if min_f<=length<=max_f:
            blink_events.append({"type":"blink","start":start/fps,"end":(len(src)-1)/fps,"dur":length/fps})
            blink_binary.extend([1]*length)
        else: blink_binary.extend([0]*length)
    if len(blink_binary)<len(df): blink_binary.extend([0]*(len(df)-len(blink_binary)))
    return blink_events, np.array(blink_binary[:len(df)],dtype=int)

# =============================================================================
# FULL TEMPORAL ANALYSIS  (post-processing on complete dataset)
# =============================================================================
def temporal_a(df, cfg):
    if df.empty: return df,{},[]
    df=df.copy(); fps=max(1,int(cfg["fps"]))
    for c in ["a1_prob_drowsy","a1_prob_yawn","a1_prob_nod","a2_prob_eye_closed","a1_confidence","a2_confidence"]:
        if c not in df.columns: df[c]=0.0
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0).clip(0,1)
    df["ear"]          = pd.to_numeric(df.get("ear",np.nan),errors="coerce")
    df["quality_score"]= pd.to_numeric(df.get("quality_score",0.0),errors="coerce").fillna(0.0).clip(0,1)
    df["face_confidence"]=pd.to_numeric(df.get("face_confidence",0.0),errors="coerce").fillna(0.0).clip(0,1)
    sw=max(1,int(0.6*fps))
    for c in ["a1_prob_drowsy","a1_prob_yawn","a1_prob_nod","a2_prob_eye_closed"]:
        df[f"{c}_sm"]=df[c].rolling(sw,min_periods=1).mean()
    df["preprocess_confidence"]=(df["quality_score"]*0.6+df["face_confidence"]*0.4).clip(0,1)
    df["eye_closed_binary"]=((df["a2_prob_eye_closed_sm"]>=cfg["t_eye"])|
                              (df["ear"].notna()&(df["ear"]<=cfg["ear_closed"]))).astype(int)
    blink_events,blink_binary=_derive_blinks(df,fps)
    df["blink_binary"]=blink_binary
    win_s=max(1,int(cfg["win_s"]*fps)); win_m=max(1,int(cfg["win_m"]*fps)); win_l=max(1,int(cfg["win_l"]*fps))
    df["perclos_30s"]       =df["eye_closed_binary"].rolling(win_m,min_periods=1).mean()
    df["perclos_60s"]       =df["eye_closed_binary"].rolling(win_l,min_periods=1).mean()
    df["blink_rate_per_min"]=df["blink_binary"].rolling(win_m,min_periods=1).sum()/max(cfg["win_m"]/60.0,1e-6)
    df["avg_blink_duration_s"]=0.0
    for evt in blink_events:
        s=int(evt["start"]*fps); e=int(evt["end"]*fps)+1
        df.loc[s:e-1,"avg_blink_duration_s"]=evt["dur"]
    df["avg_blink_duration_s"]=df["avg_blink_duration_s"].rolling(win_m,min_periods=1).mean()
    for wn,ws in [("s",cfg["win_s"]),("m",cfg["win_m"]),("l",cfg["win_l"])]:
        w=max(1,int(ws*fps))
        df[f"rd_{wn}"]=df["a1_prob_drowsy_sm"].rolling(w,min_periods=1).mean()
        df[f"re_{wn}"]=df["a2_prob_eye_closed_sm"].rolling(w,min_periods=1).mean()
        df[f"ry_{wn}"]=df["a1_prob_yawn_sm"].rolling(w,min_periods=1).mean()
        df[f"rn_{wn}"]=df["a1_prob_nod_sm"].rolling(w,min_periods=1).mean()
    df["fatigue_signal_confidence"]=(0.35*df["preprocess_confidence"]+0.35*df["a1_confidence"]+0.30*df["a2_confidence"]).clip(0,1)
    df["uncertain_frame"]=((df["fatigue_signal_confidence"]<MIN_ALERT_CONF)|
                            (df["quality_score"]<MIN_FRAME_QUALITY)).astype(int)
    df["review_flag"]=np.where(df["uncertain_frame"]==1,"needs_review","clear")
    nb=np.clip(df["blink_rate_per_min"]/20.0, 0.0, 1.0)
    df["fatigue_score_raw"]=(cfg["w_drowsy"]*df["a1_prob_drowsy_sm"]+cfg["w_eye_closed"]*df["a2_prob_eye_closed_sm"]+
                              cfg["w_nod"]*df["a1_prob_nod_sm"]+cfg["w_yawn"]*df["a1_prob_yawn_sm"]+
                              cfg["w_perclos"]*df["perclos_30s"]+cfg["w_blink"]*nb).clip(0,1)
    df["fatigue_score_calibrated"]=(df["fatigue_score_raw"]*(0.65+0.35*df["preprocess_confidence"])).clip(0,1)
    df["frame_risk"]=np.where(df["uncertain_frame"]==1, df["fatigue_score_calibrated"]*0.70, df["fatigue_score_calibrated"]).clip(0,1)
    ta_adj=np.clip(cfg["t_alert"] +(0.5-df["preprocess_confidence"])*0.10, 0.5, 0.9)
    tc_adj=np.clip(cfg["t_caution"]+(0.5-df["preprocess_confidence"])*0.08, 0.3, 0.8)
    df["risk_level"]=np.where(df["frame_risk"]>=ta_adj,"high",np.where(df["frame_risk"]>=tc_adj,"medium","low"))
    df.loc[df["uncertain_frame"]==1,"risk_level"]=df.loc[df["uncertain_frame"]==1,"risk_level"]+"_uncertain"

    # Events
    closure_events=[]
    for evt in _streak_events(df["eye_closed_binary"].values,fps,cfg["closure_frames"]/fps,"prolonged_eye_closure"):
        mask=(df["timestamp_seconds"]>=evt["start"])&(df["timestamp_seconds"]<=evt["end"])
        conf=float(df.loc[mask,"fatigue_signal_confidence"].mean()) if mask.any() else 0.0
        sev,ss=_severity_bucket(max(df.loc[mask,"frame_risk"].mean(),0.55) if mask.any() else 0.55, evt["dur"])
        closure_events.append({**evt,"severity":sev,"severity_score":ss,"confidence":conf,
                                "alert_sent":conf>=MIN_ALERT_CONF and sev in ["high","critical"],
                                "alert_type":_alert_type_from_severity(sev),
                                "event_confirmation_status":"pending_review" if conf<MIN_ALERT_CONF else "auto_confirmed",
                                "risk_group":"fatigue"})
    high_binary=(df["frame_risk"]>=cfg["t_alert"]).astype(int)
    high_events=[]
    for evt in _streak_events(high_binary.values,fps,2.0,"high_fatigue"):
        mask=(df["timestamp_seconds"]>=evt["start"])&(df["timestamp_seconds"]<=evt["end"])
        conf=float(df.loc[mask,"fatigue_signal_confidence"].mean()) if mask.any() else 0.0
        sev,ss=_severity_bucket(max(df.loc[mask,"frame_risk"].mean(),0.60) if mask.any() else 0.60, evt["dur"])
        high_events.append({**evt,"severity":sev,"severity_score":ss,"confidence":conf,
                             "alert_sent":conf>=MIN_ALERT_CONF and sev in ["high","critical"],
                             "alert_type":_alert_type_from_severity(sev),
                             "event_confirmation_status":"pending_review" if conf<MIN_ALERT_CONF else "auto_confirmed",
                             "risk_group":"fatigue"})
    uncertain_rows=[]
    for evt in _streak_events(df["uncertain_frame"].values,fps,2.0,"uncertain_segment_review"):
        uncertain_rows.append({**evt,"severity":"medium","severity_score":0.50,"confidence":0.20,
                                "alert_sent":False,"alert_type":"review_queue",
                                "event_confirmation_status":"pending_review","risk_group":"review"})
    all_events=closure_events+high_events+uncertain_rows
    total_dur=len(df)/fps if fps>0 else 0
    blink_count=len(blink_events)
    uncertain_ratio=float(df["uncertain_frame"].mean())
    avg_conf=float(df["fatigue_signal_confidence"].mean())
    hr_dur=float((df["frame_risk"]>=cfg["t_alert"]).sum()/fps)
    ca_dur=float(((df["frame_risk"]>=cfg["t_caution"])&(df["frame_risk"]<cfg["t_alert"])).sum()/fps)
    final=("high" if hr_dur>5 or len(closure_events)>=2 else
           ("medium" if hr_dur>2 or df["perclos_30s"].mean()>0.15 or len(closure_events)>=1 else "low"))
    if uncertain_ratio>=POOR_QUALITY_RATIO: final=final+"_review"
    summary={"total_frames":int(len(df)),"total_duration":float(total_dur),
             "avg_drowsy":float(df["a1_prob_drowsy_sm"].mean()),"max_drowsy":float(df["a1_prob_drowsy_sm"].max()),
             "avg_eye":float(df["a2_prob_eye_closed_sm"].mean()),"perclos":float(df["perclos_30s"].mean()),
             "eye_closure_burden":float(df["eye_closed_binary"].mean()),"closure_count":int(len(closure_events)),
             "blink_count":int(blink_count),"blink_freq_per_min":float(blink_count/total_dur*60) if total_dur>0 else 0.0,
             "avg_blink_duration":float(np.mean([e["dur"] for e in blink_events])) if blink_events else 0.0,
             "avg_ear":float(df["ear"].dropna().mean()) if df["ear"].notna().any() else None,
             "yawn_sup":float(df["a1_prob_yawn_sm"].mean()),"nod_sup":float(df["a1_prob_nod_sm"].mean()),
             "hr_dur":hr_dur,"caut_dur":ca_dur,"uncertain_ratio":uncertain_ratio,
             "poor_quality_session":uncertain_ratio>=POOR_QUALITY_RATIO,"mean_confidence":avg_conf,"final":final}
    return df,summary,all_events

def temporal_b(df, fps, sw=3, thr=OFFROAD_THRESHOLD):
    if df.empty: return df,{},[]
    df=df.copy(); fps=max(1,int(fps)); td=len(df)/fps if fps>0 else 0
    for c in ["offroad_prob","confidence","entropy","uncertainty_score"]:
        if c not in df.columns: df[c]=0.0
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0).clip(0,1)
    if sw>1:
        for c in ["offroad_prob","confidence","entropy","uncertainty_score"]:
            df[f"{c}_sm"]=df[c].rolling(sw,min_periods=1).mean()
    else:
        for c in ["offroad_prob","confidence","entropy","uncertainty_score"]:
            df[f"{c}_sm"]=df[c]
    ne=df["entropy_sm"]/np.log(len(ZONE_CLASSES))
    df["attn_uncertain"]=((df["confidence_sm"]<MIN_ALERT_CONF)|(ne>0.75)|
                           (df["uncertainty_score_sm"]>0.65)|(df["entropy_sm"]>MAX_UNCERTAIN_ENTROPY)).astype(int)
    df["review_flag"]=np.where(df["attn_uncertain"]==1,"needs_review","clear")
    enter_t=thr; exit_t=max(0.05,thr-OFFROAD_EXIT_HYSTERESIS)
    op=[]; active=False
    for prob,unc in zip(df["offroad_prob_sm"].values,df["attn_uncertain"].values):
        if int(unc)==1: active=False; op.append(0); continue
        if not active and prob>=enter_t: active=True
        elif active and prob<=exit_t: active=False
        op.append(1 if active else 0)
    df["offroad_pred"]=op
    ob=df["offroad_pred"].astype(int).values
    hb=(df["risk_group_pred"]=="HighRisk").astype(int).values
    mb=df["zone_pred"].isin(["Left Mirror","Right Mirror","Rearview"]).astype(int).values

    def max_streak(a):
        mx,c=0,0
        for v in a: c=c+1 if v else 0; mx=max(mx,c)
        return mx
    def trans(a): return sum(1 for i in range(1,len(a)) if a[i]==1 and a[i-1]==0)

    mos=max_streak(ob)/fps; mhs=max_streak(hb)/fps
    ot=trans(ob); ht=trans(hb); mt=trans(mb)
    md_list=[]; cc=0
    for v in mb:
        if v: cc+=1
        else:
            if cc>0: md_list.append(cc/fps)
            cc=0
    if cc>0: md_list.append(cc/fps)
    offroad_events=[]
    for evt in _streak_events(ob,fps,2.0,"offroad_glance"):
        mask=(df["timestamp_seconds"]>=evt["start"])&(df["timestamp_seconds"]<=evt["end"])
        conf=float(df.loc[mask,"confidence_sm"].mean()) if mask.any() else 0.0
        sev,ss=_severity_bucket(max(df.loc[mask,"offroad_prob_sm"].mean(),0.55) if mask.any() else 0.55, evt["dur"])
        offroad_events.append({**evt,"dominant_zone":df.loc[mask,"zone_pred"].mode().iloc[0] if mask.any() else "",
                                "risk_group":"distraction","severity":sev,"severity_score":ss,"confidence":conf,
                                "alert_sent":conf>=MIN_ALERT_CONF and sev in ["high","critical"],
                                "alert_type":_alert_type_from_severity(sev),
                                "event_confirmation_status":"pending_review" if conf<MIN_ALERT_CONF else "auto_confirmed"})
    repeat_dist=[]
    if td>0 and ot>=3:
        repeat_dist.append({"type":"repeated_distraction","start":0.0,"end":float(df["timestamp_seconds"].iloc[-1]),
                             "dur":float(df["timestamp_seconds"].iloc[-1]),
                             "dominant_zone":df[df["offroad_pred"]==1]["zone_pred"].mode().iloc[0] if (df["offroad_pred"]==1).any() else "",
                             "risk_group":"distraction","severity":"high" if ot>=5 else "medium",
                             "severity_score":0.80 if ot>=5 else 0.60,"confidence":float(df["confidence_sm"].mean()),
                             "alert_sent":ot>=4,"alert_type":"dashboard_escalation" if ot>=4 else "dashboard_warning",
                             "event_confirmation_status":"auto_confirmed"})
    mirror_events=[]
    for evt in _streak_events(mb,fps,0.0,"mirror_glance"):
        mask=(df["timestamp_seconds"]>=evt["start"])&(df["timestamp_seconds"]<=evt["end"])
        mirror_events.append({**evt,"dominant_zone":df.loc[mask,"zone_pred"].mode().iloc[0] if mask.any() else "",
                               "risk_group":"monitoring","severity":"low","severity_score":0.20,
                               "confidence":float(df.loc[mask,"confidence_sm"].mean()) if mask.any() else 0.0,
                               "alert_sent":False,"alert_type":"none","event_confirmation_status":"auto_confirmed"})
    unc_events=[]
    for evt in _streak_events(df["attn_uncertain"].values,fps,2.0,"uncertain_segment_review"):
        unc_events.append({**evt,"dominant_zone":"","risk_group":"review","severity":"medium",
                           "severity_score":0.50,"confidence":0.20,"alert_sent":False,
                           "alert_type":"review_queue","event_confirmation_status":"pending_review"})
    events=offroad_events+repeat_dist+mirror_events+unc_events
    kpis={"nf":int(len(df)),"td":float(td),"or":float(np.mean(ob)) if len(ob) else 0.0,
          "mos":float(mos),"oepm":float(ot/td*60) if td>0 else 0.0,
          "hr":float(np.mean(hb)) if len(hb) else 0.0,"mhs":float(mhs),
          "hepm":float(ht/td*60) if td>0 else 0.0,"mfpm":float(mt/td*60) if td>0 else 0.0,
          "amd":float(np.mean(md_list)) if md_list else 0.0,"sfr":float((df["zone_pred"]=="Forward").mean()),
          "mc":float(df["confidence_sm"].mean()),"me":float(df["entropy_sm"].mean()),
          "uncertain_ratio":float(df["attn_uncertain"].mean()),
          "poor_quality_session":float(df["attn_uncertain"].mean())>=POOR_QUALITY_RATIO,
          "repeated_distraction_events":int(len(repeat_dist))}
    return df,kpis,events

# =============================================================================
# TIMELINE  +  SCORECARD
# =============================================================================
def build_timeline(df_a, df_b):
    if df_a.empty and df_b.empty: return pd.DataFrame()
    if not df_a.empty and not df_b.empty:
        L=df_a[["frame_id","timestamp_seconds","frame_risk","eye_closed_binary","uncertain_frame"]].copy()
        R=df_b[["frame_index","timestamp_seconds","offroad_pred","risk_group_pred","attn_uncertain"]].copy()
        L["ts_k"]=L["timestamp_seconds"].round(3); R["ts_k"]=R["timestamp_seconds"].round(3)
        tl=pd.merge(L,R,on="ts_k",how="outer",suffixes=("_a","_b")).sort_values("ts_k")
        tl["timestamp_seconds"]=tl["timestamp_seconds_a"].fillna(tl["timestamp_seconds_b"])
        tl["frame_id"]=tl["frame_id"].fillna(tl["frame_index"])
    elif not df_a.empty:
        tl=df_a[["frame_id","timestamp_seconds","frame_risk","eye_closed_binary","uncertain_frame"]].copy()
        tl["offroad_pred"]=0; tl["risk_group_pred"]="Unknown"; tl["attn_uncertain"]=0
    else:
        tl=df_b[["frame_index","timestamp_seconds","offroad_pred","risk_group_pred","attn_uncertain"]].copy()
        tl=tl.rename(columns={"frame_index":"frame_id"}); tl["frame_risk"]=0.0
        tl["eye_closed_binary"]=0; tl["uncertain_frame"]=0
    keep=["frame_id","timestamp_seconds","frame_risk","eye_closed_binary",
          "uncertain_frame","offroad_pred","risk_group_pred","attn_uncertain"]
    tl=tl[[c for c in keep if c in tl.columns]].copy()
    def label_row(r):
        if int(r.get("uncertain_frame",0))==1 or int(r.get("attn_uncertain",0))==1: return "needs_review"
        if int(r.get("eye_closed_binary",0))==1:
            return "prolonged_eye_closure" if float(r.get("frame_risk",0))>=0.6 else "mild_fatigue"
        if int(r.get("offroad_pred",0))==1: return "offroad_glance"
        if str(r.get("risk_group_pred",""))=="HighRisk":
            return "repeated_distraction" if float(r.get("frame_risk",0))>=0.4 else "offroad_glance"
        if float(r.get("frame_risk",0))>=0.7: return "high_fatigue"
        if float(r.get("frame_risk",0))>=0.4: return "mild_fatigue"
        return "normal_driving"
    tl["timeline_label"]=tl.apply(label_row,axis=1)
    tl["overall_risk_score"]=(0.55*pd.to_numeric(tl.get("frame_risk",0),errors="coerce").fillna(0)+
                               0.45*pd.to_numeric(tl.get("offroad_pred",0),errors="coerce").fillna(0)).clip(0,1)
    return tl.sort_values("timestamp_seconds").reset_index(drop=True)

def summarize_timeline(tl, fps):
    if tl.empty: return []
    labels=tl["timeline_label"].tolist(); times=tl["timestamp_seconds"].tolist()
    rows=[]; start=0; cur=labels[0]
    for i in range(1,len(labels)):
        if labels[i]!=cur:
            rows.append({"timeline_state":cur,"start_ts":times[start],"end_ts":times[i-1],
                         "duration_seconds":max(0.0,times[i-1]-times[start]+(1/fps if fps else 0))})
            start=i; cur=labels[i]
    rows.append({"timeline_state":cur,"start_ts":times[start],"end_ts":times[-1],
                 "duration_seconds":max(0.0,times[-1]-times[start]+(1/fps if fps else 0))})
    return rows

def build_driver_scorecard(driver_id, trip_id, sid, a_sum, b_sum):
    fs=float(np.clip(0.45*a_sum.get("avg_drowsy",0)+0.25*a_sum.get("perclos",0)+
                     0.20*min(a_sum.get("closure_count",0)/3.0,1.0)+0.10*min(a_sum.get("hr_dur",0)/10.0,1.0),0,1))
    ds=float(np.clip(0.45*b_sum.get("or",0)+0.25*min(b_sum.get("oepm",0)/6.0,1.0)+
                     0.20*(1-b_sum.get("sfr",0))+0.10*min(b_sum.get("repeated_distraction_events",0)/2.0,1.0),0,1))
    mc=float(np.clip((a_sum.get("mean_confidence",0.0)+b_sum.get("mc",0.0))/2.0,0.0,1.0))
    unc_pen=0.08*max(0.0,0.6-mc)
    combined=float(np.clip(0.55*fs+0.45*ds+unc_pen,0,1))
    rating="A" if combined<=0.20 else ("B" if combined<=0.40 else ("C" if combined<=0.60 else ("D" if combined<=0.80 else "E")))
    return {"driver_id":driver_id,"trip_id":trip_id,"session_id":sid,
            "fatigue_risk_score":fs,"distraction_risk_score":ds,"combined_driver_safety_score":combined,
            "average_offroad_ratio":float(b_sum.get("or",0)),
            "prolonged_eye_closure_count":int(a_sum.get("closure_count",0)),
            "mirror_usage_pattern":"healthy" if b_sum.get("mfpm",0)>=1 else "low_mirror_usage",
            "safe_forward_ratio":float(b_sum.get("sfr",0)),
            "highrisk_events_per_driving_hour":float(b_sum.get("hepm",0)),
            "mean_model_confidence":mc,"driver_rating":rating,"created_at_client":utc_now_iso()}

def _explain_unified(a_sum, b_sum, scorecard):
    parts=[]
    if a_sum.get("perclos",0)>0.15: parts.append(f"PERCLOS elevated at {a_sum['perclos']:.1%}")
    if a_sum.get("closure_count",0)>0: parts.append(f"{a_sum['closure_count']} prolonged eye closure event(s)")
    if b_sum.get("or",0)>0.20: parts.append(f"off-road ratio {b_sum['or']:.1%}")
    if b_sum.get("repeated_distraction_events",0)>0: parts.append("repeated distraction pattern detected")
    if a_sum.get("poor_quality_session") or b_sum.get("poor_quality_session"): parts.append("some frames flagged for review")
    if not parts: return "Session looks stable — no sustained fatigue or distraction pattern detected."
    return "Summary: "+("; ".join(parts))+f". Overall rating: {scorecard['driver_rating']}."

# =============================================================================
# ALERT MANAGER  (NEW in v4)
# =============================================================================
class AlertManager:
    """
    Manages real-time alert firing, cooldowns, and 5-level escalation logic.
    Escalation windows: levels based on actionable-alert count in last 5 minutes.
    Also supports de-escalation after sustained safe-driving periods.
    """
    def __init__(self, cooldown_s: float = ALERT_COOLDOWN_S):
        self.cooldown_s   = cooldown_s
        self.last_fired   = {}          # alert_type → last timestamp
        self.alert_history= []          # all fired alerts
        self.esc_level    = 0           # 0–4
        self._esc_ts      = deque()     # timestamps of escalation-worthy alerts
        self._safe_since  = None        # timestamp when last safe period started

    # ── public API ────────────────────────────────────────────────────────────
    def try_fire(self, alert_type: str, severity: str, severity_score: float,
                 ts: float, message: str, confidence: float,
                 fatigue_score: float = 0.0, offroad_prob: float = 0.0,
                 zone: str = "Unknown") -> dict | None:
        """Fire an alert if cooldown allows. Returns alert dict or None."""
        last = self.last_fired.get(alert_type, -999)
        if ts - last < self.cooldown_s:
            return None
        self.last_fired[alert_type] = ts
        alert = {"alert_id":       str(uuid.uuid4())[:16],
                 "alert_type":     alert_type,
                 "severity":       severity,
                 "severity_score": severity_score,
                 "alert_timestamp_seconds": ts,
                 "alert_wall_time": utc_now_iso(),
                 "message":        message,
                 "confidence":     round(confidence, 3),
                 "fatigue_score":  round(fatigue_score, 3),
                 "offroad_prob":   round(offroad_prob, 3),
                 "zone_at_alert":  zone,
                 "escalation_level": self.esc_level}
        self.alert_history.append(alert)
        if severity in ["medium","high","critical"]:
            self._esc_ts.append(ts)
            self._safe_since = None
            self._update_escalation(ts)
        return alert

    def mark_safe_frame(self, ts: float):
        """Called when current state is safe — tracks de-escalation."""
        if self._safe_since is None:
            self._safe_since = ts
        elif ts - self._safe_since >= 120:   # 2 min of safety → step down
            if self.esc_level > 0:
                self.esc_level -= 1
            self._safe_since = ts

    def _update_escalation(self, now: float):
        window = 300.0   # 5-minute window
        while self._esc_ts and now - self._esc_ts[0] > window:
            self._esc_ts.popleft()
        n = len(self._esc_ts)
        if   n >= 8: self.esc_level = 4
        elif n >= 5: self.esc_level = 3
        elif n >= 3: self.esc_level = 2
        elif n >= 1: self.esc_level = 1

    @property
    def total_alerts(self): return len(self.alert_history)

    def alerts_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.alert_history) if self.alert_history else pd.DataFrame()

# =============================================================================
# ROLLING BUFFER  (NEW in v4)
# =============================================================================
class RollingBuffer:
    """Maintains the last N frames of A and B data for rolling temporal analysis."""
    def __init__(self, max_size: int = 300):
        self._a: deque = deque(maxlen=max_size)
        self._b: deque = deque(maxlen=max_size)

    def push_a(self, row: dict): self._a.append(row)
    def push_b(self, row: dict): self._b.append(row)

    def df_a(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._a)) if self._a else pd.DataFrame()
    def df_b(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._b)) if self._b else pd.DataFrame()

# =============================================================================
# REAL-TIME STATE COMPUTATION  (NEW in v4 — lightweight, runs every chunk)
# =============================================================================
def compute_current_state(roll_a: pd.DataFrame, roll_b: pd.DataFrame, cfg: dict) -> dict:
    """
    Quickly compute the current driver state from the rolling buffer.
    Lightweight version of temporal_a/b suitable for per-chunk updates.
    """
    out = {"state":"normal_driving","frame_risk":0.0,"perclos":0.0,"eye_closed":0,
           "offroad_prob":0.0,"zone":"Unknown","confidence":0.5,"ear":None,
           "fatigue_conf":0.5,"attn_uncertain":False}
    fps = max(1, cfg["fps"])

    if not roll_a.empty:
        df = roll_a.copy()
        for c in ["a1_prob_drowsy","a2_prob_eye_closed","a1_confidence","a2_confidence",
                  "quality_score","face_confidence"]:
            if c not in df.columns: df[c]=0.0
            df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
        sw = max(1, int(0.6*fps))
        df["d_sm"] = df["a1_prob_drowsy"].rolling(sw,min_periods=1).mean()
        df["e_sm"] = df["a2_prob_eye_closed"].rolling(sw,min_periods=1).mean()
        df["ear_"]  = pd.to_numeric(df.get("ear",np.nan),errors="coerce")
        df["ec_bin"]= ((df["e_sm"]>=cfg["t_eye"])|(df["ear_"].notna()&(df["ear_"]<=cfg["ear_closed"]))).astype(int)
        win_m = max(1, int(cfg["win_m"]*fps))
        perclos = float(df["ec_bin"].rolling(win_m,min_periods=1).mean().iloc[-1])
        last = df.iloc[-1]
        pc = float((float(last.get("quality_score",0))*0.6+float(last.get("face_confidence",0))*0.4))
        fr = float(np.clip(cfg["w_drowsy"]*float(last.get("d_sm",0))+
                           cfg["w_eye_closed"]*float(last.get("e_sm",0))+
                           cfg["w_perclos"]*perclos, 0, 1))
        fr_cal = float(np.clip(fr*(0.65+0.35*pc), 0, 1))
        fc = float((float(last.get("a1_confidence",0.5))*0.5+float(last.get("a2_confidence",0.5))*0.5))
        out.update({"frame_risk":fr_cal,"perclos":perclos,
                    "eye_closed":int(df["ec_bin"].iloc[-1]),
                    "ear":float(last.get("ear_")) if pd.notna(last.get("ear_")) else None,
                    "fatigue_conf":float(np.clip(pc*0.4+fc*0.6,0,1))})

    if not roll_b.empty:
        df2 = roll_b.copy()
        for c in ["offroad_prob","confidence","uncertainty_score"]:
            if c not in df2.columns: df2[c]=0.0
            df2[c]=pd.to_numeric(df2[c],errors="coerce").fillna(0.0)
        last2 = df2.iloc[-1]
        out.update({"offroad_prob":float(last2.get("offroad_prob",0)),
                    "zone":str(last2.get("zone_pred","Unknown")),
                    "confidence":float(last2.get("confidence",0.5)),
                    "attn_uncertain":bool(float(last2.get("uncertainty_score",0))>0.65 or
                                         float(last2.get("confidence",1))<MIN_ALERT_CONF)})

    fr = out["frame_risk"]
    if out.get("attn_uncertain") and out.get("eye_closed"):
        out["state"] = "needs_review"
    elif out["eye_closed"] and fr >= 0.55:
        out["state"] = "prolonged_eye_closure"
    elif fr >= cfg["t_alert"]:
        out["state"] = "high_fatigue"
    elif fr >= cfg["t_caution"]:
        out["state"] = "mild_fatigue"
    elif out["offroad_prob"] >= OFFROAD_THRESHOLD and not out.get("attn_uncertain"):
        risk_g = get_risk_group(out["zone"])
        out["state"] = "repeated_distraction" if risk_g=="HighRisk" else "offroad_glance"
    return out

# =============================================================================
# FRAME ANNOTATION  (NEW in v4)
# =============================================================================
def annotate_frame(frame_bgr, state: str, cs: dict, esc_level: int, alert_active: bool) -> np.ndarray:
    """
    Draw HUD overlays on a frame:
      - Colour-coded border (state-based)
      - Top-left semi-transparent panel with metrics
      - Bottom alert banner when alert is active
      - Escalation level badge
    """
    if not CV2_AVAILABLE or frame_bgr is None: return frame_bgr
    f = frame_bgr.copy(); h,w = f.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    border_bgr = {"normal_driving":(0,200,0),"mild_fatigue":(0,165,255),
                  "high_fatigue":(0,0,255),"offroad_glance":(0,130,255),
                  "repeated_distraction":(0,0,220),"prolonged_eye_closure":(0,0,255),
                  "needs_review":(128,128,128)}.get(state,(0,200,0))
    thickness = 7 if alert_active else 4
    cv2.rectangle(f,(0,0),(w-1,h-1),border_bgr,thickness)

    # Semi-transparent HUD box
    ov = f.copy()
    cv2.rectangle(ov,(0,0),(min(w,300),175),(0,0,0),-1)
    cv2.addWeighted(ov,0.50,f,0.50,0,f)

    state_col = {"normal_driving":(50,220,50),"mild_fatigue":(0,180,255),
                 "high_fatigue":(0,60,255),"offroad_glance":(0,130,255),
                 "prolonged_eye_closure":(0,0,255),"repeated_distraction":(0,0,200)
                 }.get(state,(220,220,220))
    label = state.upper().replace("_"," ")
    cv2.putText(f, label,           (8,24),  font, 0.48, state_col, 1, cv2.LINE_AA)
    cv2.putText(f, f"FATIGUE: {cs['frame_risk']:.2f}  CONF: {cs['fatigue_conf']:.2f}",
                (8,48),  font,0.42,(220,220,220),1,cv2.LINE_AA)
    cv2.putText(f, f"OFFROAD: {cs['offroad_prob']:.2f}  ZONE: {cs['zone']}",
                (8,70),  font,0.42,(220,220,220),1,cv2.LINE_AA)
    cv2.putText(f, f"PERCLOS: {cs['perclos']:.1%}",
                (8,92),  font,0.42,(220,220,220),1,cv2.LINE_AA)
    ear_txt = f"EAR: {cs['ear']:.3f}" if cs.get("ear") is not None else "EAR: N/A"
    cv2.putText(f, ear_txt,         (8,114), font,0.42,(220,220,220),1,cv2.LINE_AA)

    esc_bgr = [(20,180,20),(20,165,0),(0,130,255),(0,0,220),(0,0,160)][min(esc_level,4)]
    esc_txt = ["NORMAL","LOW","MEDIUM","HIGH","CRITICAL"][min(esc_level,4)]
    cv2.rectangle(f,(8,122),(110,142),esc_bgr,-1)
    cv2.putText(f, f"ESC:{esc_txt}",  (12,138),font,0.38,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(f, f"T:{cs.get('ts',0):.1f}s", (8,160),font,0.38,(180,180,180),1,cv2.LINE_AA)

    if alert_active:
        ov2=f.copy(); cv2.rectangle(ov2,(0,h-52),(w,h),(0,0,200),-1)
        cv2.addWeighted(ov2,0.65,f,0.35,0,f)
        cv2.putText(f,"⚠  ALERT  —  DRIVER NEEDS ATTENTION",(max(4,w//2-200),h-18),
                    font,0.65,(255,255,255),2,cv2.LINE_AA)
    return f

# =============================================================================
# ANALYTICS BUILDERS
# =============================================================================
def build_unified_frame_predictions(df_a, df_b, sid, driver_id, trip_id, src):
    if df_a.empty and df_b.empty: return pd.DataFrame()
    if not df_a.empty and not df_b.empty:
        mg=pd.merge(df_a,df_b,left_on="frame_id",right_on="frame_index",
                    how="outer",suffixes=("_a","_b"))
        mg["timestamp_seconds"]=pd.to_numeric(mg.get("timestamp_seconds_a"),errors="coerce").fillna(
                                 pd.to_numeric(mg.get("timestamp_seconds_b"),errors="coerce"))
    elif not df_a.empty:
        mg=df_a.copy(); mg["frame_index"]=mg["frame_id"]
        for c in ["offroad_prob","offroad_pred","confidence","uncertainty_score","attn_uncertain"]:
            mg[c]=0.0 if c!="uncertainty_score" else 1.0
        mg["zone_pred"]="Unknown"; mg["risk_group_pred"]="Unknown"; mg["inference_mode_b"]="missing"
    else:
        mg=df_b.copy(); mg["frame_id"]=mg["frame_index"]
        mg["frame_risk"]=0.0; mg["risk_level"]="unknown"; mg["uncertain_frame"]=1; mg["inference_mode_a"]="missing"
    for col in ["frame_risk","offroad_prob","confidence","uncertainty_score"]:
        if col not in mg.columns: mg[col]=0.0
        mg[col]=pd.to_numeric(mg[col],errors="coerce").fillna(0.0).clip(0,1)
    if "offroad_pred" not in mg.columns: mg["offroad_pred"]=(mg["offroad_prob"]>=OFFROAD_THRESHOLD).astype(int)
    for col,default in [("risk_level","unknown"),("zone_pred","Unknown"),("risk_group_pred","Unknown")]:
        if col not in mg.columns: mg[col]=default
    for col in ["uncertain_frame","attn_uncertain"]:
        if col not in mg.columns: mg[col]=1
    mg["overall_risk_score"]=(0.50*mg["frame_risk"]+0.35*mg["offroad_prob"]+0.15*(1.0-mg["confidence"])).clip(0,1)
    mg["needs_review"]=((mg["uncertain_frame"].astype(int)==1)|(mg["attn_uncertain"].astype(int)==1)).astype(int)
    mg["overall_risk_label"]=np.select([mg["needs_review"]==1,mg["overall_risk_score"]>=0.70,
                                         mg["overall_risk_score"]>=0.40],["needs_review","high","medium"],default="low")
    for k,v in [("session_id",sid),("driver_id",driver_id),("trip_id",trip_id),
                ("source_file_name",src),("created_at_client",utc_now_iso())]:
        mg[k]=v
    keep=["session_id","driver_id","trip_id","source_file_name","frame_id","frame_index",
          "timestamp_seconds","overall_risk_score","overall_risk_label","needs_review",
          "frame_risk","risk_level","offroad_prob","offroad_pred","zone_pred","risk_group_pred",
          "confidence","uncertainty_score","uncertain_frame","attn_uncertain",
          "a1_prob_drowsy","a2_prob_eye_closed","perclos_30s","blink_rate_per_min",
          "quality_score","face_confidence","preprocess_method","inference_mode_a",
          "inference_mode_b","created_at_client"]+[f"zone_prob_{z.lower().replace(' ','_')}" for z in ZONE_CLASSES]
    for c in ["inference_mode_a","inference_mode_b"]:
        if c not in mg.columns: mg[c]="unknown"
    out=mg[[c for c in keep if c in mg.columns]].copy()
    out["frame_id"]=pd.to_numeric(out.get("frame_id",np.nan),errors="coerce").fillna(
                    pd.to_numeric(out.get("frame_index",np.nan),errors="coerce"))
    return out.sort_values("frame_id").reset_index(drop=True)

def build_zone_transitions(df_b, sid, driver_id, trip_id):
    if df_b.empty: return pd.DataFrame()
    x=df_b[["frame_index","timestamp_seconds","zone_pred","risk_group_pred","confidence"]].copy()
    x=x.sort_values("frame_index").reset_index(drop=True)
    rows=[]; prev=None; st_ts=None; st_idx=None
    for _,row in x.iterrows():
        z,ts,idx=row["zone_pred"],float(row["timestamp_seconds"]),int(row["frame_index"])
        if prev is None: prev=z;st_ts=ts;st_idx=idx; continue
        if z!=prev:
            rows.append({"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
                         "from_zone":prev,"to_zone":z,"start_ts":st_ts,"end_ts":ts,
                         "duration_seconds":max(0.0,ts-st_ts),"start_frame_index":st_idx,
                         "end_frame_index":idx,"created_at_client":utc_now_iso()})
            prev=z;st_ts=ts;st_idx=idx
    return pd.DataFrame(rows)

def build_quality_summary(df_a, df_b, sid, driver_id, trip_id, src):
    qa=float(pd.to_numeric(df_a.get("quality_score",0.0),errors="coerce").fillna(0.0).mean()) if not df_a.empty else 0.0
    fa=float(pd.to_numeric(df_a.get("face_confidence",0.0),errors="coerce").fillna(0.0).mean()) if not df_a.empty else 0.0
    ua=float(pd.to_numeric(df_a.get("uncertain_frame",1),errors="coerce").fillna(1.0).mean()) if not df_a.empty else 1.0
    ub=float(pd.to_numeric(df_b.get("attn_uncertain",1),errors="coerce").fillna(1.0).mean()) if not df_b.empty else 1.0
    pm="unknown"
    if not df_a.empty and "preprocess_method" in df_a.columns and df_a["preprocess_method"].notna().any():
        pm=str(df_a["preprocess_method"].mode().iloc[0])
    lm_counts={}
    if not df_a.empty and "lighting_method" in df_a.columns:
        lm_counts=df_a["lighting_method"].value_counts().to_dict()
    return pd.DataFrame([{"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,"source_file_name":src,
                           "avg_frame_quality":qa,"avg_face_confidence":fa,
                           "fatigue_uncertain_ratio":ua,"attn_uncertain_ratio":ub,
                           "dominant_preprocess_method":pm,
                           "lighting_corrections_applied":str(lm_counts),
                           "overall_quality_score":float(np.clip(0.4*qa+0.3*fa+0.3*(1-max(ua,ub)),0,1)),
                           "created_at_client":utc_now_iso()}])

# =============================================================================
# HTML TIMELINE RENDERER  (NEW in v4)
# =============================================================================
def render_timeline_html(timeline_rows: list, total_dur: float) -> str:
    """Render a colour-coded horizontal driver timeline bar as HTML."""
    if not timeline_rows or total_dur <= 0: return ""
    bars=""
    for r in timeline_rows:
        dur=r.get("duration_seconds",0); pct=max(0.1, dur/total_dur*100)
        col=STATE_COLORS.get(r.get("timeline_state","normal_driving"),"#22c55e")
        s=r.get("start_ts",0); e=r.get("end_ts",0); lbl=r.get("timeline_state","").replace("_"," ").title()
        tip=f"{lbl} | {_fmt_ts(s)} → {_fmt_ts(e)} ({dur:.1f}s)"
        bars+=f'<div title="{tip}" style="display:inline-block;width:{pct:.3f}%;height:44px;background:{col};cursor:pointer;vertical-align:top;border-right:1px solid rgba(0,0,0,0.15);"></div>'
    legend="".join([f'<span style="display:inline-flex;align-items:center;margin-right:12px;font-size:11px;">'
                    f'<span style="background:{col};display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;"></span>'
                    f'{lbl.replace("_"," ").title()}</span>'
                    for lbl,col in STATE_COLORS.items()])
    return (f'<div style="margin:8px 0 4px 0;">'
            f'<div style="width:100%;border-radius:4px;overflow:hidden;border:1px solid #333;">{bars}</div>'
            f'<div style="margin-top:8px;display:flex;flex-wrap:wrap;">{legend}</div>'
            f'</div>')

def _fmt_ts(s: float) -> str:
    h=int(s//3600); m=int((s%3600)//60); sec=int(s%60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h>0 else f"{m:02d}:{sec:02d}"

# =============================================================================
# FRAME EXPLORER
# =============================================================================
def render_frame_explorer(frame_map: dict, unified_df: pd.DataFrame):
    st.subheader("🔍 Frame-Level Explorer")
    if unified_df.empty or not frame_map: st.caption("No data available."); return
    fdf=unified_df.copy()
    fdf["frame_id"]=pd.to_numeric(fdf["frame_id"],errors="coerce").fillna(-1).astype(int)
    fdf=fdf[fdf["frame_id"]>=0].sort_values("frame_id").reset_index(drop=True)
    risk_opts=["needs_review","high","medium","low"]
    sel=st.multiselect("Filter by risk label",risk_opts,default=risk_opts,key="fe_risk")
    fdf=fdf[fdf["overall_risk_label"].isin(sel)]
    if fdf.empty: st.caption("No frames match."); return
    idx=st.slider("Pick frame",0,len(fdf)-1,0,key="fe_slider")
    r=fdf.iloc[idx]; fid=int(r["frame_id"])
    c1,c2=st.columns([2,3])
    with c1:
        if fid in frame_map:
            fr=frame_map[fid]
            st.image(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB),caption=f"Frame {fid} | {float(r['timestamp_seconds']):.2f}s",use_container_width=True)
    with c2:
        a,b,c,d=st.columns(4)
        a.metric("Overall Risk", f"{float(r['overall_risk_score']):.2f}")
        b.metric("Fatigue Risk", f"{float(r['frame_risk']):.2f}")
        c.metric("Off-road",     f"{float(r['offroad_prob']):.2f}")
        d.metric("Confidence",   f"{float(r['confidence']):.2f}")
        st.caption(f"Label: **{r['overall_risk_label']}** | Zone: **{r.get('zone_pred','?')}** | "
                   f"Risk Group: **{r.get('risk_group_pred','?')}** | Review: {int(r.get('needs_review',0))}")
        zpc=[f"zone_prob_{z.lower().replace(' ','_')}" for z in ZONE_CLASSES if f"zone_prob_{z.lower().replace(' ','_')}" in fdf.columns]
        if zpc:
            zr=pd.DataFrame({"Zone":[c.replace("zone_prob_","").replace("_"," ").title() for c in zpc],
                             "Prob":[float(r[c]) for c in zpc]}).sort_values("Prob",ascending=False).head(4)
            st.bar_chart(zr,x="Zone",y="Prob")
    view_cols=["frame_id","timestamp_seconds","overall_risk_label","overall_risk_score",
               "frame_risk","risk_level","offroad_prob","zone_pred","risk_group_pred","confidence","uncertainty_score"]
    st.dataframe(fdf[[c for c in view_cols if c in fdf.columns]],hide_index=True,use_container_width=True,height=280)

    st.subheader("📸 Frame Gallery")
    pg_size=st.select_slider("Per page",[4,6,9,12],6,key="fe_ps")
    total_pg=max(1,int(np.ceil(len(fdf)/pg_size)))
    pg=st.number_input("Page",1,total_pg,1,key="fe_pg")
    sub=fdf.iloc[(pg-1)*pg_size:pg*pg_size]
    cols=st.columns(3)
    for i,(_,row) in enumerate(sub.iterrows()):
        with cols[i%3]:
            fid2=int(row["frame_id"])
            if fid2 in frame_map:
                st.image(cv2.cvtColor(frame_map[fid2],cv2.COLOR_BGR2RGB),use_container_width=True)
            st.caption(f"F{fid2} | {float(row['timestamp_seconds']):.1f}s | {row['overall_risk_label']} | {row.get('zone_pred','?')}")

# =============================================================================
# VIDEO UTILITIES
# =============================================================================
def open_video(video_bytes: bytes, file_ext: str):
    """Write bytes to temp file, return (path, metadata_dict)."""
    tmp=tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False)
    tmp.write(video_bytes); tmp.close()
    cap=cv2.VideoCapture(tmp.name)
    vfps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return tmp.name, {"video_fps":float(vfps),"total_frames":int(total),
                      "duration_seconds":float(total/vfps) if vfps>0 else 0.0,
                      "width":W,"height":H,"resolution":f"{W}x{H}"}

# =============================================================================
# SNOWFLAKE BATCH WRITERS
# =============================================================================
def _write_all_to_sf(df_a, df_b, a_events, b_events, a_sum, b_sum, scorecard,
                     unified_frame_df, zone_trans_df, quality_df, tl_rows,
                     unified_sum, sid, driver_id, trip_id, session_start_ts, session_end_ts, uploaded_name):
    A_COLS=["session_id","driver_id","trip_id","source_type","source_name","frame_id","timestamp_seconds",
            "a1_prob_drowsy","a1_prob_yawn","a1_prob_nod","a2_prob_eye_closed","a2_eye_openness_score",
            "a1_prob_drowsy_sm","a2_prob_eye_closed_sm","frame_risk","risk_level","head_pitch","head_yaw",
            "head_roll","ear","left_ear","right_ear","quality_score","brightness_score","blur_score",
            "face_confidence","preprocess_method","lighting_method","fatigue_signal_confidence",
            "uncertain_frame","review_flag","eye_closed_binary","blink_binary","perclos_30s",
            "blink_rate_per_min","inference_mode"]
    write_to_snowflake(df_a[[c for c in A_COLS if c in df_a.columns]].copy(),"MODULE_A_FRAME_PREDICTIONS")

    B_COLS=["session_id","driver_id","trip_id","input_type","source_file_name","frame_index",
            "timestamp_seconds","zone_pred","risk_group_pred","zone_top2","offroad_prob","offroad_prob_sm",
            "offroad_pred","confidence","confidence_sm","entropy","entropy_sm","margin","uncertainty_score",
            "uncertainty_score_sm","attn_uncertain","review_flag","inference_mode"
            ]+[f"zone_prob_{z.lower().replace(' ','_')}" for z in ZONE_CLASSES]
    write_to_snowflake(df_b[[c for c in B_COLS if c in df_b.columns]].copy(),"MODULE_B_FRAME_PREDICTIONS")

    def _ev_row(ev, sid, driver_id, trip_id, module):
        base={"event_id":str(uuid.uuid4())[:16],"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
              "event_type":ev["type"],"event_start_ts":ev["start"],"event_end_ts":ev["end"],
              "duration_seconds":ev["dur"],"severity":ev["severity"],"severity_score":ev["severity_score"],
              "confidence":ev["confidence"],"alert_sent":ev["alert_sent"],"alert_type":ev["alert_type"],
              "event_confirmation_status":ev["event_confirmation_status"],"risk_group":ev.get("risk_group",""),
              "created_at_client":utc_now_iso()}
        if module=="a": base.update({"dominant_zone":"","explanation":f"{ev['type']} for {ev['dur']:.2f}s"})
        else:           base.update({"dominant_zone":ev.get("dominant_zone","")})
        return base
    if a_events: write_to_snowflake(pd.DataFrame([_ev_row(e,sid,driver_id,trip_id,"a") for e in a_events]),"MODULE_A_EVENTS")
    if b_events: write_to_snowflake(pd.DataFrame([_ev_row(e,sid,driver_id,trip_id,"b") for e in b_events]),"MODULE_B_EVENTS")

    write_to_snowflake(pd.DataFrame([{
        "session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
        "source_name":uploaded_name,"source_type":"video",
        "session_start_ts":session_start_ts,"session_end_ts":session_end_ts,
        "duration_seconds":a_sum["total_duration"],"total_frames_processed":a_sum["total_frames"],
        "total_duration_seconds":a_sum["total_duration"],"avg_a1_prob_drowsy":a_sum["avg_drowsy"],
        "max_a1_prob_drowsy":a_sum["max_drowsy"],"avg_a2_prob_eye_closed":a_sum["avg_eye"],
        "eye_closure_burden":a_sum["eye_closure_burden"],"perclos":a_sum["perclos"],
        "blink_count":a_sum["blink_count"],"blink_freq_per_min":a_sum["blink_freq_per_min"],
        "avg_blink_duration_seconds":a_sum["avg_blink_duration"],"avg_ear":a_sum["avg_ear"],
        "prolonged_closure_count":a_sum["closure_count"],"yawn_support_score":a_sum["yawn_sup"],
        "nod_support_score":a_sum["nod_sup"],"total_high_risk_duration":a_sum["hr_dur"],
        "total_caution_duration":a_sum["caut_dur"],"mean_confidence":a_sum["mean_confidence"],
        "uncertain_ratio":a_sum["uncertain_ratio"],"poor_quality_session":a_sum["poor_quality_session"],
        "final_session_risk":a_sum["final"],"model_version_a1":MODEL_VER_A1,
        "model_version_a2":MODEL_VER_A2,"app_version":APP_VERSION,"created_at_client":utc_now_iso()}]),"MODULE_A_SESSION_SUMMARY")

    write_to_snowflake(pd.DataFrame([{
        "session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
        "source_file_name":uploaded_name,"session_start_ts":session_start_ts,"session_end_ts":session_end_ts,
        "duration_seconds":b_sum["td"],"total_frames":b_sum["nf"],"total_duration_seconds":b_sum["td"],
        "offroad_ratio":b_sum["or"],"max_offroad_streak_seconds":b_sum["mos"],
        "offroad_events_per_min":b_sum["oepm"],"highrisk_ratio":b_sum["hr"],
        "max_highrisk_streak_seconds":b_sum["mhs"],"highrisk_events_per_min":b_sum["hepm"],
        "mirror_glance_frequency_per_min":b_sum["mfpm"],"avg_mirror_glance_duration_seconds":b_sum["amd"],
        "safe_forward_ratio":b_sum["sfr"],"mean_confidence":b_sum["mc"],"mean_entropy":b_sum["me"],
        "uncertain_ratio":b_sum["uncertain_ratio"],"poor_quality_session":b_sum["poor_quality_session"],
        "repeated_distraction_events":b_sum["repeated_distraction_events"],
        "model_version_b":MODEL_VER_B,"app_version":APP_VERSION,"created_at_client":utc_now_iso()}]),"MODULE_B_SESSION_SUMMARY")

    write_to_snowflake(pd.DataFrame([unified_sum]),"UNIFIED_DRIVER_SESSION_SUMMARY")
    write_to_snowflake(pd.DataFrame([scorecard]),"DRIVER_SCORECARDS")
    write_to_snowflake(unified_frame_df,"UNIFIED_FRAME_PREDICTIONS")
    if not zone_trans_df.empty: write_to_snowflake(zone_trans_df,"MODULE_B_ZONE_TRANSITIONS")
    write_to_snowflake(quality_df,"SESSION_DATA_QUALITY_SUMMARY")
    if tl_rows:
        tl_df=pd.DataFrame(tl_rows); tl_df["session_id"]=sid; tl_df["driver_id"]=driver_id
        tl_df["trip_id"]=trip_id; tl_df["created_at_client"]=utc_now_iso()
        write_to_snowflake(tl_df,"DRIVER_TIMELINE")
    rev_q=unified_frame_df[unified_frame_df["needs_review"]==1].copy() if not unified_frame_df.empty else pd.DataFrame()
    if not rev_q.empty: write_to_snowflake(rev_q,"FRAME_REVIEW_QUEUE")

# =============================================================================
# STREAMING INFERENCE  (THE CORE NEW FUNCTION — v4)
# =============================================================================
def run_streaming_inference(video_path: str, video_meta: dict, cfg_a: dict,
                             fps_b: int, sw_b: int, offroad_thr: float,
                             driver_id: str, trip_id: str, source_name: str,
                             sim_a_mode: bool, sim_b_mode: bool, sid: str):
    """
    Chunk-based streaming inference.
    Processes CHUNK_SECONDS of video at a time, updates UI live, fires alerts
    immediately. Returns (all_rows_a, all_rows_b, alert_manager, frame_map).
    """
    m1,m2,bm = load_a1(), load_a2(), load_b()
    vfps      = video_meta["video_fps"]
    total_vf  = video_meta["total_frames"]
    afps      = cfg_a["fps"]
    interval  = max(1, int(vfps / max(1, afps)))   # raw frames to skip
    chunk_raw = max(interval, int(CHUNK_SECONDS * vfps))  # raw frames per chunk
    roll_size = max(60, int(60 * afps))

    rolling   = RollingBuffer(max_size=roll_size)
    alert_mgr = AlertManager(cooldown_s=ALERT_COOLDOWN_S)

    all_rows_a: list = []
    all_rows_b: list = []
    frame_map:  dict = {}   # global_sampled_idx → BGR frame (thumbnails only)

    # ── UI setup ──────────────────────────────────────────────────────────────
    st.markdown("### 🎬 Live Inference")
    prog_bar   = st.progress(0, "Initialising...")
    alert_out  = st.empty()
    col_v, col_m = st.columns([3, 2])
    with col_v:
        frame_ph  = st.empty()
    with col_m:
        metrics_ph= st.empty()

    st.markdown("---")
    st.markdown("#### 🚨 Real-time Alert Log")
    log_ph   = st.empty()
    st.markdown("#### 📈 Rolling Fatigue Signal (last 60 s)")
    chart_ph = st.empty()

    # ── main loop ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    global_raw_idx = 0
    global_samp_idx= 0
    chunk_idx      = 0

    while cap.isOpened():
        chunk_frames: list = []   # [(raw_idx, ts, bgr)]

        for _ in range(chunk_raw):
            ret, frame = cap.read()
            if not ret: break
            if global_raw_idx % interval == 0:
                ts = global_raw_idx / vfps
                chunk_frames.append((global_raw_idx, ts, frame, global_samp_idx))
                global_samp_idx += 1
            global_raw_idx += 1

        if not chunk_frames:
            break

        chunk_new_alerts: list = []

        for raw_idx, ts, frame_bgr, samp_idx in chunk_frames:
            # ── lighting normalisation (v4 NEW) ───────────────────────────────
            frame_norm, light_method = normalize_lighting(frame_bgr)
            meta = detect_face_eyes(frame_norm)
            rng  = _make_rng(source_name, samp_idx)

            # ── Module A inference ────────────────────────────────────────────
            if sim_a_mode:
                ra = sim_a(rng); a_src = "simulation"
            else:
                o1  = infer_a1(m1, meta["face"])
                o2  = infer_a2(m2, meta["eye_strip"])
                if o1 or o2:
                    fb  = sim_a(rng); ra = {**fb,**(o1 or {}),**(o2 or {})}
                    a_src = "hybrid_model" if not (o1 and o2) else "model"
                else:
                    ra = sim_a(rng); a_src = "simulation_fallback"

            # ── Module B inference ────────────────────────────────────────────
            if sim_b_mode:
                rb = sim_b(rng, offroad_thr); b_src = "simulation"
            else:
                rb = infer_b(bm, frame_bgr, offroad_thr)
                if rb is None: rb = sim_b(rng, offroad_thr); b_src = "simulation_fallback"
                else: b_src = "model"

            row_a = {"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
                     "source_type":"video","source_name":source_name,
                     "frame_id":samp_idx,"timestamp_seconds":ts,
                     "a1_prob_drowsy":ra.get("a1_prob_drowsy"),"a1_prob_yawn":ra.get("a1_prob_yawn"),
                     "a1_prob_nod":ra.get("a1_prob_nod"),"a2_prob_eye_closed":ra.get("a2_prob_eye_closed"),
                     "a2_eye_openness_score":ra.get("a2_eye_openness_score"),
                     "a1_confidence":ra.get("a1_confidence"),"a2_confidence":ra.get("a2_confidence"),
                     "ear":meta.get("ear"),"left_ear":meta.get("left_ear"),"right_ear":meta.get("right_ear"),
                     "head_pitch":meta.get("head_pitch"),"head_yaw":meta.get("head_yaw"),"head_roll":meta.get("head_roll"),
                     "quality_score":meta.get("quality_score"),"face_confidence":meta.get("face_confidence"),
                     "brightness_score":meta.get("brightness_score"),"blur_score":meta.get("blur_score"),
                     "preprocess_method":meta.get("preprocess_method"),"lighting_method":light_method,
                     "inference_mode":a_src}
            row_b = {"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
                     "input_type":"video","source_file_name":source_name,
                     "frame_index":samp_idx,"timestamp_seconds":ts,"inference_mode":b_src,**rb}

            rolling.push_a(row_a)
            rolling.push_b(row_b)
            all_rows_a.append(row_a)
            all_rows_b.append(row_b)
            # Keep thumbnail every ~12th sampled frame (for post-inference explorer)
            if samp_idx % max(1, max(1, int(afps*5))) == 0:
                frame_map[samp_idx] = frame_bgr.copy()

        # ── end of chunk: compute current state ───────────────────────────────
        cs = compute_current_state(rolling.df_a(), rolling.df_b(), cfg_a)
        cs["ts"] = chunk_frames[-1][1]
        state    = cs["state"]
        ts_last  = chunk_frames[-1][1]

        # ── alert firing ─────────────────────────────────────────────────────
        fr   = cs["frame_risk"]
        op   = cs["offroad_prob"]
        conf = cs["fatigue_conf"]
        zone = cs["zone"]

        if state in ["high_fatigue","prolonged_eye_closure"]:
            a = alert_mgr.try_fire(
                "drowsiness_alert","high",fr,ts_last,
                f"🚨 DROWSINESS DETECTED @ {_fmt_ts(ts_last)} | Fatigue:{fr:.2f} | Conf:{conf:.2f}",
                conf, fr, op, zone)
            if a: chunk_new_alerts.append(a)

        if cs["perclos"] >= 0.25:
            a = alert_mgr.try_fire(
                "perclos_alert","high",cs["perclos"],ts_last,
                f"😴 HIGH PERCLOS @ {_fmt_ts(ts_last)} | PERCLOS:{cs['perclos']:.1%} | Conf:{conf:.2f}",
                conf, fr, op, zone)
            if a: chunk_new_alerts.append(a)

        if state in ["offroad_glance","repeated_distraction"] and conf >= MIN_ALERT_CONF:
            sev = "high" if get_risk_group(zone)=="HighRisk" else "medium"
            a = alert_mgr.try_fire(
                "distraction_alert",sev,op,ts_last,
                f"👁 DISTRACTION @ {_fmt_ts(ts_last)} | Zone:{zone} | Off-road:{op:.2f} | Conf:{conf:.2f}",
                conf, fr, op, zone)
            if a: chunk_new_alerts.append(a)

        if state == "normal_driving":
            alert_mgr.mark_safe_frame(ts_last)

        # ── escalation-level critical override ────────────────────────────────
        if alert_mgr.esc_level == 4:
            a = alert_mgr.try_fire(
                "escalation_critical","critical",1.0,ts_last,
                f"🆘 CRITICAL ESCALATION @ {_fmt_ts(ts_last)} — IMMEDIATE ACTION REQUIRED",
                conf, fr, op, zone)
            if a: chunk_new_alerts.append(a)

        # ── annotate display frame ────────────────────────────────────────────
        disp_raw = chunk_frames[-1][2]
        disp_ann = annotate_frame(disp_raw, state, cs, alert_mgr.esc_level, bool(chunk_new_alerts))
        dw = DISPLAY_WIDTH; dh = int(dw * disp_ann.shape[0] / max(1, disp_ann.shape[1]))
        disp_small = cv2.resize(disp_ann, (dw, dh))

        # ── UI updates ────────────────────────────────────────────────────────
        pct = min(global_raw_idx / max(total_vf, 1), 1.0)
        prog_bar.progress(pct, f"Processing {_fmt_ts(ts_last)} / {_fmt_ts(video_meta['duration_seconds'])} "
                               f"| Alerts: {alert_mgr.total_alerts} | Escalation: {ESC_LABELS[alert_mgr.esc_level]}")

        if chunk_new_alerts:
            esc = alert_mgr.esc_level
            esc_col = ["green","orange","orange","red","red"][min(esc,4)]
            msgs = "\n\n".join([f"**{a['message']}**" for a in chunk_new_alerts])
            with alert_out.container():
                st.error(f"{'🆘' if esc>=4 else '🚨'} **{ESC_LABELS[min(esc,4)]}** — {msgs}")
        elif state == "normal_driving" and chunk_idx % 5 == 0:
            alert_out.success(f"✅ Normal driving | {_fmt_ts(ts_last)}")

        with frame_ph.container():
            st.image(cv2.cvtColor(disp_small, cv2.COLOR_BGR2RGB),
                     caption=f"{state.replace('_',' ').title()} | {_fmt_ts(ts_last)} | "
                             f"Fatigue:{fr:.2f} | Zone:{zone} | Escalation:{ESC_LABELS[alert_mgr.esc_level]}",
                     use_container_width=True)

        with metrics_ph.container():
            st.markdown(f"<div style='background:{ESC_HEX[alert_mgr.esc_level]};padding:8px 12px;border-radius:8px;"
                        f"color:white;font-weight:bold;font-size:15px;margin-bottom:8px;'>"
                        f"Escalation: {ESC_LABELS[alert_mgr.esc_level]}</div>",unsafe_allow_html=True)
            r1c1,r1c2 = st.columns(2)
            r1c1.metric("Fatigue Risk",  f"{fr:.2f}",     delta="↑HIGH" if fr>=cfg_a["t_alert"] else None, delta_color="inverse")
            r1c2.metric("Off-road Prob", f"{op:.2f}",     delta="↑" if op>=offroad_thr else None, delta_color="inverse")
            r2c1,r2c2 = st.columns(2)
            r2c1.metric("PERCLOS",       f"{cs['perclos']:.1%}")
            r2c2.metric("Gaze Zone",     zone)
            r3c1,r3c2 = st.columns(2)
            ear_val = f"{cs['ear']:.3f}" if cs.get("ear") is not None else "N/A"
            ear_del = ("CLOSED" if cs.get("ear") is not None and cs["ear"]<EAR_CLOSED_THRESHOLD else None)
            r3c1.metric("EAR", ear_val, delta=ear_del, delta_color="inverse")
            r3c2.metric("Alerts Fired", alert_mgr.total_alerts)
            st.metric("Confidence", f"{conf:.2f}")

        # Alert log (last 8)
        if alert_mgr.alert_history:
            with log_ph.container():
                log_df = alert_mgr.alerts_df().tail(8)[
                    [c for c in ["alert_timestamp_seconds","severity","alert_type","message","confidence","escalation_level"]
                     if c in alert_mgr.alerts_df().columns]]
                log_df = log_df.copy()
                if "alert_timestamp_seconds" in log_df.columns:
                    log_df["time"] = log_df["alert_timestamp_seconds"].apply(_fmt_ts)
                    log_df = log_df.drop(columns=["alert_timestamp_seconds"])
                st.dataframe(log_df.iloc[::-1], hide_index=True, use_container_width=True)

        # Rolling chart (last 60s of fatigue)
        recent_n = min(len(all_rows_a), int(60*afps))
        if recent_n >= 3:
            recent = all_rows_a[-recent_n:]
            chart_df = pd.DataFrame({
                "Time": [r["timestamp_seconds"] for r in recent],
                "Fatigue": [float(r.get("a1_prob_drowsy") or 0) for r in recent],
                "Eye Closed": [float(r.get("a2_prob_eye_closed") or 0) for r in recent],
            })
            with chart_ph.container():
                st.line_chart(chart_df, x="Time", y=["Fatigue","Eye Closed"])

        # Incremental Snowflake writes
        if all_rows_a:
            write_to_snowflake(pd.DataFrame(all_rows_a[-len(chunk_frames):]),"MODULE_A_FRAME_PREDICTIONS")
        if all_rows_b:
            write_to_snowflake(pd.DataFrame(all_rows_b[-len(chunk_frames):]),"MODULE_B_FRAME_PREDICTIONS")
        if chunk_new_alerts:
            write_to_snowflake(pd.DataFrame([{**a,"session_id":sid,"driver_id":driver_id,
                                              "trip_id":trip_id,"created_at_client":utc_now_iso()}
                                             for a in chunk_new_alerts]),"REALTIME_ALERTS")
        chunk_idx += 1

    cap.release()
    prog_bar.progress(1.0, f"✅ Processing complete — {alert_mgr.total_alerts} alerts fired.")
    time.sleep(0.8); prog_bar.empty()
    return all_rows_a, all_rows_b, alert_mgr, frame_map

# =============================================================================
# POST-INFERENCE ANALYTICS  (runs after streaming completes)
# =============================================================================
def render_post_analytics(df_a, df_b, cfg_a, afps, offroad_thr, sw_b,
                           sid, driver_id, trip_id, uploaded_name,
                           session_start_ts, session_end_ts,
                           alert_mgr: AlertManager, frame_map: dict):
    st.markdown("---")
    st.markdown("## 📊 Full Session Analytics")

    # Run full temporal analysis
    with st.spinner("Running final temporal analysis…"):
        df_a, a_sum, a_events = temporal_a(df_a, cfg_a)
        df_b, b_sum, b_events = temporal_b(df_b, afps, sw_b, offroad_thr)

    timeline_df  = build_timeline(df_a, df_b)
    timeline_df["session_id"]=sid; timeline_df["driver_id"]=driver_id; timeline_df["trip_id"]=trip_id
    tl_rows      = summarize_timeline(timeline_df, afps)
    scorecard    = build_driver_scorecard(driver_id, trip_id, sid, a_sum, b_sum)
    summary_text = _explain_unified(a_sum, b_sum, scorecard)
    uf_df        = build_unified_frame_predictions(df_a, df_b, sid, driver_id, trip_id, uploaded_name)
    zt_df        = build_zone_transitions(df_b, sid, driver_id, trip_id)
    qsum_df      = build_quality_summary(df_a, df_b, sid, driver_id, trip_id, uploaded_name)
    total_alrts  = alert_mgr.total_alerts

    cds = scorecard["combined_driver_safety_score"]
    esc_level = min(4, alert_mgr.esc_level)
    escalation_level_final = min(3, max(esc_level,
        len([e for e in a_events+b_events if e.get("alert_sent")])))

    unified_sum = {
        "session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
        "session_start_ts":session_start_ts,"session_end_ts":session_end_ts,
        "duration_seconds":float(a_sum.get("total_duration",0)),
        "source_file_name":uploaded_name,"model_version_a1":MODEL_VER_A1,
        "model_version_a2":MODEL_VER_A2,"model_version_b":MODEL_VER_B,
        "app_version":APP_VERSION,"fatigue_score":scorecard["fatigue_risk_score"],
        "distraction_score":scorecard["distraction_risk_score"],
        "combined_driver_safety_score":cds,"driver_rating":scorecard["driver_rating"],
        "summary_text":summary_text,"escalation_level":escalation_level_final,
        "mean_model_confidence":scorecard["mean_model_confidence"],
        "total_realtime_alerts":total_alrts,
        "trip_start_ts":session_start_ts,"trip_end_ts":session_end_ts,
        "created_at_client":utc_now_iso()}

    # ── Overall rating banner ─────────────────────────────────────────────────
    r_color={"A":"green","B":"green","C":"orange","D":"red","E":"red"}.get(scorecard["driver_rating"],"gray")
    st.markdown(f"<div style='display:flex;gap:20px;flex-wrap:wrap;margin-bottom:8px;'>"
                f"<div style='background:{ESC_HEX[esc_level]};padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>"
                f"Escalation: {ESC_LABELS[esc_level]}</div>"
                f"<div style='background:#1e293b;padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>"
                f"Driver Rating: <span style='color:{'#22c55e' if r_color=='green' else '#f97316' if r_color=='orange' else '#ef4444'};font-size:20px'>"
                f"{scorecard['driver_rating']}</span></div>"
                f"<div style='background:#1e293b;padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>"
                f"Safety Score: {cds:.2f}</div>"
                f"<div style='background:#1e293b;padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>"
                f"Alerts Fired: {total_alrts}</div></div>",unsafe_allow_html=True)
    st.caption(summary_text)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Overview","Fatigue","Attentiveness","Driver Timeline",
                    "Real-time Alerts","Frame Explorer","Events","Scorecard","Data Export"])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tabs[0]:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Fatigue Score",     f"{scorecard['fatigue_risk_score']:.2f}")
        c2.metric("Distraction Score", f"{scorecard['distraction_risk_score']:.2f}")
        c3.metric("Safety Score",      f"{cds:.2f}")
        c4.metric("Driver Rating",     scorecard["driver_rating"])
        c5.metric("Realtime Alerts",   total_alrts)
        c6,c7,c8,c9 = st.columns(4)
        c6.metric("PERCLOS",         f"{a_sum['perclos']:.1%}")
        c7.metric("Blinks/min",      f"{a_sum['blink_freq_per_min']:.1f}")
        c8.metric("Off-road Ratio",  f"{b_sum['or']:.1%}")
        c9.metric("Safe Forward %",  f"{b_sum['sfr']:.1%}")

        # Sample annotated thumbnails
        if frame_map:
            sample_keys = sorted(frame_map.keys())[:6]
            cols_thumb = st.columns(min(3, len(sample_keys)))
            for i, k in enumerate(sample_keys[:6]):
                with cols_thumb[i % 3]:
                    st.image(cv2.cvtColor(frame_map[k], cv2.COLOR_BGR2RGB),
                             caption=f"Frame {k}", use_container_width=True)
        st.caption("ℹ️ Lower safety score = safer driver. Escalation reflects peak alert intensity during session.")

    # ── Tab 2: Fatigue ────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Fatigue Signal Over Time")
        fat_cols = [c for c in ["timestamp_seconds","frame_risk","a1_prob_drowsy_sm",
                                 "a2_prob_eye_closed_sm","perclos_30s","blink_rate_per_min"]
                    if c in df_a.columns]
        if len(fat_cols) > 1:
            fat_df = df_a[fat_cols].rename(columns={"timestamp_seconds":"Time (s)",
                "frame_risk":"Fatigue Risk","a1_prob_drowsy_sm":"Drowsy (sm)",
                "a2_prob_eye_closed_sm":"Eye Closed (sm)","perclos_30s":"PERCLOS 30s",
                "blink_rate_per_min":"Blink/min"})
            # Mark alert timestamps on chart
            if not df_a.empty:
                st.line_chart(fat_df, x="Time (s)")

        # Alert markers
        if alert_mgr.alert_history:
            fat_alerts = [a for a in alert_mgr.alert_history if "drowsiness" in a["alert_type"] or "perclos" in a["alert_type"]]
            if fat_alerts:
                fat_alert_df = pd.DataFrame(fat_alerts)[["alert_timestamp_seconds","severity","message","confidence"]]
                fat_alert_df["time"] = fat_alert_df["alert_timestamp_seconds"].apply(_fmt_ts)
                st.markdown("**⚠️ Fatigue Alerts Fired During Session:**")
                st.dataframe(fat_alert_df.drop(columns=["alert_timestamp_seconds"]),
                             hide_index=True, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg EAR",          f"{a_sum['avg_ear']:.3f}" if a_sum.get("avg_ear") else "N/A")
        c2.metric("Blink Count",      a_sum["blink_count"])
        c3.metric("Avg Blink Dur",    f"{a_sum['avg_blink_duration']:.2f}s")
        c4.metric("Uncertain Ratio",  f"{a_sum['uncertain_ratio']:.1%}")
        st.caption(f"Thresholds → Caution:{cfg_a['t_caution']:.2f}  Alert:{cfg_a['t_alert']:.2f}  "
                   f"Eye closed:{cfg_a['t_eye']:.2f}  EAR:{cfg_a['ear_closed']:.2f}")

    # ── Tab 3: Attentiveness ──────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Attentiveness Signal Over Time")
        att_cols=[c for c in ["timestamp_seconds","offroad_prob_sm","confidence_sm","uncertainty_score_sm"] if c in df_b.columns]
        if len(att_cols)>1:
            att_df=df_b[att_cols].rename(columns={"timestamp_seconds":"Time (s)","offroad_prob_sm":"Off-road (sm)",
                                                    "confidence_sm":"Confidence (sm)","uncertainty_score_sm":"Uncertainty (sm)"})
            st.line_chart(att_df, x="Time (s)")
        st.subheader("Gaze Zone Distribution")
        zd=df_b["zone_pred"].value_counts().reset_index(); zd.columns=["Zone","Count"]
        st.bar_chart(zd, x="Zone", y="Count")

        # Distraction alerts
        if alert_mgr.alert_history:
            dist_alerts=[a for a in alert_mgr.alert_history if "distraction" in a["alert_type"]]
            if dist_alerts:
                dat=pd.DataFrame(dist_alerts)[["alert_timestamp_seconds","severity","zone_at_alert","message","confidence"]]
                dat["time"]=dat["alert_timestamp_seconds"].apply(_fmt_ts)
                st.markdown("**👁 Distraction Alerts:**")
                st.dataframe(dat.drop(columns=["alert_timestamp_seconds"]),hide_index=True,use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Off-road Ratio",  f"{b_sum['or']:.1%}")
        c2.metric("Max Streak",      f"{b_sum['mos']:.1f}s")
        c3.metric("Mirror Freq/min", f"{b_sum['mfpm']:.1f}")
        c4.metric("HighRisk Ratio",  f"{b_sum['hr']:.1%}")

    # ── Tab 4: Driver Timeline ────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Driver State Timeline")
        if tl_rows:
            total_dur = sum(r["duration_seconds"] for r in tl_rows)
            html = render_timeline_html(tl_rows, total_dur)
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("**Segment Breakdown:**")
            tl_show = pd.DataFrame(tl_rows).copy()
            tl_show["start"] = tl_show["start_ts"].apply(_fmt_ts)
            tl_show["end"]   = tl_show["end_ts"].apply(_fmt_ts)
            tl_show["duration_min"] = (tl_show["duration_seconds"] / 60).round(2)
            st.dataframe(tl_show[["timeline_state","start","end","duration_seconds","duration_min"]],
                         hide_index=True, use_container_width=True)
            state_agg = tl_show.groupby("timeline_state")["duration_seconds"].sum().reset_index()
            state_agg.columns=["State","Duration (s)"]
            state_agg["Duration (min)"]=(state_agg["Duration (s)"]/60).round(2)
            st.bar_chart(state_agg, x="State", y="Duration (s)")
        else:
            st.caption("No timeline segments available.")

    # ── Tab 5: Real-time Alerts ───────────────────────────────────────────────
    with tabs[4]:
        st.subheader(f"Real-time Alert History ({total_alrts} alerts)")
        if alert_mgr.alert_history:
            aldf = alert_mgr.alerts_df()
            aldf["time"] = aldf["alert_timestamp_seconds"].apply(_fmt_ts)

            # Escalation progression chart
            esc_over_time = aldf[["time","escalation_level","severity","confidence"]].copy()
            st.markdown("**Escalation Level Progression:**")
            esc_chart = aldf[["alert_timestamp_seconds","escalation_level"]].copy()
            esc_chart.columns=["Time (s)","Escalation Level"]
            st.line_chart(esc_chart, x="Time (s)", y="Escalation Level")

            st.markdown("**Alert Type Distribution:**")
            type_cnt = aldf["alert_type"].value_counts().reset_index()
            type_cnt.columns=["Alert Type","Count"]
            st.bar_chart(type_cnt, x="Alert Type", y="Count")

            st.markdown("**Full Alert Log:**")
            disp_cols = [c for c in ["time","severity","alert_type","message","confidence",
                                      "fatigue_score","offroad_prob","zone_at_alert","escalation_level"]
                         if c in aldf.columns]
            st.dataframe(aldf[disp_cols].iloc[::-1], hide_index=True, use_container_width=True, height=400)

            sev_counts = aldf["severity"].value_counts().reset_index()
            sev_counts.columns=["Severity","Count"]
            st.bar_chart(sev_counts, x="Severity", y="Count")
        else:
            st.success("No alerts were fired during this session — excellent driving!")

    # ── Tab 6: Frame Explorer ─────────────────────────────────────────────────
    with tabs[5]:
        render_frame_explorer(frame_map, uf_df)

    # ── Tab 7: Events ─────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Detected Events")
        all_ev_rows=[]
        for ev in a_events:
            all_ev_rows.append({"module":"A","event_type":ev["type"],"start":_fmt_ts(ev["start"]),
                                 "end":_fmt_ts(ev["end"]),"duration_s":round(ev["dur"],2),
                                 "severity":ev["severity"],"severity_score":round(ev["severity_score"],3),
                                 "confidence":round(ev["confidence"],3),"alert_sent":ev["alert_sent"],
                                 "risk_group":ev.get("risk_group","")})
        for ev in b_events:
            all_ev_rows.append({"module":"B","event_type":ev["type"],"start":_fmt_ts(ev["start"]),
                                 "end":_fmt_ts(ev["end"]),"duration_s":round(ev["dur"],2),
                                 "severity":ev["severity"],"severity_score":round(ev["severity_score"],3),
                                 "confidence":round(ev["confidence"],3),"alert_sent":ev["alert_sent"],
                                 "risk_group":ev.get("risk_group",""),
                                 "dominant_zone":ev.get("dominant_zone","")})
        if all_ev_rows:
            ev_df=pd.DataFrame(all_ev_rows)
            st.dataframe(ev_df, hide_index=True, use_container_width=True, height=380)
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Events",           len(ev_df))
            c2.metric("Actionable Alerts",      int(ev_df["alert_sent"].astype(int).sum()))
            c3.metric("Prolonged Eye Closures", int((ev_df["event_type"]=="prolonged_eye_closure").sum()))
            sev_br=ev_df["severity"].value_counts().reset_index(); sev_br.columns=["Severity","Count"]
            st.bar_chart(sev_br, x="Severity", y="Count")
        else:
            st.caption("No events detected.")

    # ── Tab 8: Scorecard ──────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Driver Scorecard")
        sc_display = {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in scorecard.items()}
        st.dataframe(pd.DataFrame([sc_display]), hide_index=True, use_container_width=True)

        st.markdown("**Score Breakdown:**")
        bd={"Component":["Fatigue (55%)","Distraction (45%)","Uncertainty Penalty"],
            "Score":[scorecard["fatigue_risk_score"],scorecard["distraction_risk_score"],
                     round(0.08*max(0.0,0.6-scorecard["mean_model_confidence"]),4)]}
        st.bar_chart(pd.DataFrame(bd), x="Component", y="Score")

        st.markdown(f"**Rating Interpretation:**")
        st.info("A (≤0.20): Excellent  |  B (≤0.40): Good  |  C (≤0.60): Caution  |  D (≤0.80): Poor  |  E (>0.80): Critical")
        st.download_button("⬇ Download Scorecard CSV", pd.DataFrame([scorecard]).to_csv(index=False),
                           f"scorecard_{sid}.csv","text/csv")
        st.dataframe(qsum_df, hide_index=True, use_container_width=True)

    # ── Tab 9: Data Export ────────────────────────────────────────────────────
    with tabs[8]:
        st.subheader("Snowflake Export Summary")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Module A Frames", len(df_a)); c2.metric("Module B Frames", len(df_b))
        c3.metric("Unified Frames",  len(uf_df));c4.metric("Realtime Alerts",  total_alrts)
        combined_export=pd.merge(df_a,df_b,on="timestamp_seconds",how="outer",suffixes=("_A","_B"))
        st.download_button("⬇ Combined Frame CSV", combined_export.to_csv(index=False),
                           f"combined_frames_{sid}.csv","text/csv")
        st.download_button("⬇ Unified Predictions CSV", uf_df.to_csv(index=False),
                           f"unified_preds_{sid}.csv","text/csv")
        if not alert_mgr.alerts_df().empty:
            st.download_button("⬇ Real-time Alerts CSV", alert_mgr.alerts_df().to_csv(index=False),
                               f"realtime_alerts_{sid}.csv","text/csv")
        if tl_rows:
            st.download_button("⬇ Driver Timeline CSV", pd.DataFrame(tl_rows).to_csv(index=False),
                               f"driver_timeline_{sid}.csv","text/csv")

    # ── Final Snowflake batch write ───────────────────────────────────────────
    with st.spinner("Writing final summaries to Snowflake…"):
        _write_all_to_sf(df_a, df_b, a_events, b_events, a_sum, b_sum, scorecard,
                         uf_df, zt_df, qsum_df, tl_rows, unified_sum,
                         sid, driver_id, trip_id, session_start_ts, session_end_ts, uploaded_name)
    st.success("✅ All data written to Snowflake successfully.")

# =============================================================================
# PAGES
# =============================================================================
def home_page():
    st.title(":material/directions_car: Driver Safety Analytics Platform  v4")
    st.markdown("**Real-time chunk-based inference · Live alerts with escalation · Lighting-robust preprocessing · Full Snowflake analytics**")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("A1 Model", "Loaded ✅" if load_a1() else "Simulation mode")
        st.metric("A2 Model", "Loaded ✅" if load_a2() else "Simulation mode")
    with c2:
        st.metric("Module B",  "Loaded ✅" if load_b()  else "Simulation mode")
        st.metric("MediaPipe", "Available ✅" if MP_AVAILABLE else "Fallback (Haar/crop)")
    with c3:
        for pkg,ok in [("PyTorch",TORCH_AVAILABLE),("OpenCV",CV2_AVAILABLE),("Pillow",PIL_AVAILABLE)]:
            st.write(f"{'✅' if ok else '❌'} {pkg}")
        st.caption(f"Snowflake: {'Connected ✅' if get_session() else 'Not connected ⚠️'}")
    st.divider()
    with st.container(border=True):
        st.subheader("What's New in v4.0.0")
        st.markdown("""
**Real-time inference**
- Chunk-based processing (5-second windows) — alerts fire immediately, not after the full video
- Live annotated frame display updates after every chunk with HUD overlays
- Rolling temporal buffer carries PERCLOS / blink / smoothing state across chunk boundaries

**Smart alerting**
- 5-level escalation logic (Normal → Low → Medium → High → Critical) based on alert density in the last 5 minutes
- Automatic de-escalation after 2 minutes of clean safe-driving
- Alert cooldown (25 s per type) prevents spam
- Confidence score shown on every alert
- Incremental writes to new `REALTIME_ALERTS` Snowflake table

**Lighting robustness**
- Adaptive per-frame normalisation: gamma correction + CLAHE for sunlight glare, CLAHE for night / low-light, histogram equalisation for flat overcast
- Applied before face detection AND before model inference
- Lighting method logged in `MODULE_A_FRAME_PREDICTIONS`

**Analytics**
- Colour-coded interactive Driver Timeline bar (HTML)
- Dedicated "Real-time Alerts" tab with escalation progression chart
- Enhanced Fatigue / Attentiveness / Events tabs with alert markers at exact timestamps
- Frame Gallery retains annotated thumbnails
        """)

    st.divider()
    st.subheader("📋 Snowflake Schema Changes Required for v4")
    st.code("""
-- 1. New table (required for real-time alerts)
CREATE TABLE IF NOT EXISTS DEMO_DB.PUBLIC.REALTIME_ALERTS (
    ALERT_ID VARCHAR(64), SESSION_ID VARCHAR(64), DRIVER_ID VARCHAR(64),
    TRIP_ID VARCHAR(64), ALERT_TIMESTAMP_SECONDS FLOAT,
    ALERT_WALL_TIME VARCHAR(64), ALERT_TYPE VARCHAR(64),
    SEVERITY VARCHAR(16), SEVERITY_SCORE FLOAT, ESCALATION_LEVEL INT,
    FATIGUE_SCORE FLOAT, OFFROAD_PROB FLOAT, ZONE_AT_ALERT VARCHAR(32),
    CONFIDENCE FLOAT, MESSAGE VARCHAR(512),
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 2. New columns on existing tables
ALTER TABLE DEMO_DB.PUBLIC.MODULE_A_FRAME_PREDICTIONS
    ADD COLUMN IF NOT EXISTS LIGHTING_METHOD VARCHAR(32);
ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
    ADD COLUMN IF NOT EXISTS MEAN_MODEL_CONFIDENCE FLOAT;
ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
    ADD COLUMN IF NOT EXISTS TOTAL_REALTIME_ALERTS INT;
""", language="sql")


def unified_page():
    st.title(":material/security: Unified Driver Safety — Real-time Analysis")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("🪪 Driver & Trip")
        driver_id = st.text_input("Driver ID",  "DRV_001")
        trip_id   = st.text_input("Trip ID",    "TRIP_001")
        trip_start= st.text_input("Trip Start", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        trip_end  = st.text_input("Trip End",   "")

        st.subheader("⚙️ Sampling")
        afps      = st.number_input("Target FPS", 1, 30, DEFAULT_A["fps"])

        st.subheader("🔥 Fatigue Weights")
        wd = st.slider("Drowsy",     0.0, 1.0, DEFAULT_A["w_drowsy"],    0.05)
        we = st.slider("Eye Closed", 0.0, 1.0, DEFAULT_A["w_eye_closed"],0.05)
        wn = st.slider("Nod",        0.0, 1.0, DEFAULT_A["w_nod"],       0.05)
        wy = st.slider("Yawn",       0.0, 1.0, DEFAULT_A["w_yawn"],      0.05)
        wp = st.slider("PERCLOS",    0.0, 1.0, DEFAULT_A["w_perclos"],   0.05)
        wb = st.slider("Blink",      0.0, 1.0, DEFAULT_A["w_blink"],     0.05)

        st.subheader("🎯 Thresholds")
        te  = st.slider("Eye closed thr",  0.0, 1.0, DEFAULT_A["t_eye"],     0.05)
        tc  = st.slider("Caution thr",     0.0, 1.0, DEFAULT_A["t_caution"], 0.05)
        ta  = st.slider("Alert thr",       0.0, 1.0, DEFAULT_A["t_alert"],   0.05)
        pcf = st.number_input("Min closure frames", 2, 30, DEFAULT_A["closure_frames"])
        ear_c=st.slider("EAR closed",      0.10,0.40, DEFAULT_A["ear_closed"],0.01)

        st.subheader("👁 Attentiveness")
        sw_b  = st.number_input("Smoothing window", 1, 10, 3)
        bort  = st.slider("Off-road threshold", 0.0, 1.0, OFFROAD_THRESHOLD, 0.05)

        st.subheader("🚨 Alert Settings")
        st.caption(f"Cooldown: {ALERT_COOLDOWN_S}s between same-type alerts")
        st.caption(f"Chunk size: {CHUNK_SECONDS}s of video per update")

    cfg_a = dict(w_drowsy=wd,w_eye_closed=we,w_nod=wn,w_yawn=wy,w_perclos=wp,w_blink=wb,
                 t_eye=te,t_caution=tc,t_alert=ta,closure_frames=pcf,fps=afps,
                 win_s=DEFAULT_A["win_s"],win_m=DEFAULT_A["win_m"],win_l=DEFAULT_A["win_l"],ear_closed=ear_c)

    sim_a_mode = load_a1() is None or load_a2() is None
    sim_b_mode = load_b() is None
    if sim_a_mode or sim_b_mode:
        st.info(":material/science: One or more models not loaded — deterministic simulation will be used for those modules.")

    uploaded = st.file_uploader("📂 Upload driver video", type=["mp4","avi","mov","mkv"])
    if not uploaded:
        st.caption("Upload a video file to begin real-time analysis.")
        return

    ext    = uploaded.name.rsplit(".",1)[-1].lower()
    sid    = str(uuid.uuid4())[:12]
    ss_ts  = trip_start if trip_start else utc_now_iso()
    se_ts  = trip_end   if trip_end   else ""

    if not CV2_AVAILABLE:
        st.error("OpenCV is not available — cannot process video."); return

    st.subheader("📹 Uploaded Video (original)")
    st.video(uploaded.getvalue())

    if st.button("🚀 Start Real-time Analysis", type="primary", use_container_width=True):
        video_bytes = uploaded.getvalue()
        with st.spinner("Preparing video…"):
            video_path, video_meta = open_video(video_bytes, ext)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Original FPS", f"{video_meta['video_fps']:.2f}")
        c2.metric("Total Frames", int(video_meta["total_frames"]))
        c3.metric("Duration",     f"{video_meta['duration_seconds']:.1f}s")
        c4.metric("Resolution",   video_meta["resolution"])
        sample_int = max(1, int(video_meta["video_fps"] / max(1, afps)))
        c5.metric("Sample Interval", f"every {sample_int} frame(s)")

        try:
            all_rows_a, all_rows_b, alert_mgr, frame_map = run_streaming_inference(
                video_path, video_meta, cfg_a, afps, sw_b, bort,
                driver_id, trip_id, uploaded.name, sim_a_mode, sim_b_mode, sid)
        finally:
            try: os.unlink(video_path)
            except Exception: pass

        if not all_rows_a:
            st.error("No frames were processed. Check the video file."); return

        df_a = pd.DataFrame(all_rows_a)
        df_b = pd.DataFrame(all_rows_b)
        df_a["session_id"] = sid; df_b["session_id"] = sid

        render_post_analytics(df_a, df_b, cfg_a, afps, bort, sw_b,
                               sid, driver_id, trip_id, uploaded.name,
                               ss_ts, se_ts, alert_mgr, frame_map)


# =============================================================================
# HISTORICAL ANALYTICS PAGE
# =============================================================================
def analytics_page():
    st.title(":material/analytics: Historical Analytics")
    tabs = st.tabs(["Sessions","Scorecards","Driver Timeline","Real-time Alerts",
                    "Frame Risk","Zone Transitions","Review Queue"])

    with tabs[0]:
        df=fetch_recent("UNIFIED_DRIVER_SESSION_SUMMARY",200)
        if df.empty: st.caption("No sessions yet."); return
        st.dataframe(df, hide_index=True, use_container_width=True)
        for col,label in [("COMBINED_DRIVER_SAFETY_SCORE","Safety Score"),
                           ("FATIGUE_SCORE","Fatigue Score"),("DISTRACTION_SCORE","Distraction Score")]:
            if {"DRIVER_ID",col}.issubset(df.columns):
                agg=df.groupby("DRIVER_ID",as_index=False)[col].mean()
                agg.columns=["Driver",label]
                st.bar_chart(agg,x="Driver",y=label)
        if "DRIVER_RATING" in df.columns:
            rc=df["DRIVER_RATING"].value_counts().reset_index(); rc.columns=["Rating","Count"]
            st.bar_chart(rc,x="Rating",y="Count")
        if "TOTAL_REALTIME_ALERTS" in df.columns:
            st.bar_chart(df[["SESSION_ID","TOTAL_REALTIME_ALERTS"]].dropna(),x="SESSION_ID",y="TOTAL_REALTIME_ALERTS")

    with tabs[1]:
        sc=fetch_recent("DRIVER_SCORECARDS",200)
        if sc.empty: st.caption("No scorecards yet.")
        else:
            st.dataframe(sc,hide_index=True,use_container_width=True)
            for col in ["FATIGUE_RISK_SCORE","DISTRACTION_RISK_SCORE","COMBINED_DRIVER_SAFETY_SCORE"]:
                if {"DRIVER_ID",col}.issubset(sc.columns):
                    agg=sc.groupby("DRIVER_ID",as_index=False)[col].mean(); agg.columns=["Driver",col]
                    st.bar_chart(agg,x="Driver",y=col)

    with tabs[2]:
        tl=fetch_recent("DRIVER_TIMELINE",2000)
        if tl.empty: st.caption("No timeline data yet.")
        else:
            st.dataframe(tl.head(500),hide_index=True,use_container_width=True)
            if {"TIMELINE_STATE","DURATION_SECONDS"}.issubset(tl.columns):
                sd=tl.groupby("TIMELINE_STATE",as_index=False)["DURATION_SECONDS"].sum()
                sd.columns=["State","Total Duration (s)"]
                st.bar_chart(sd,x="State",y="Total Duration (s)")

    with tabs[3]:
        ra=fetch_recent("REALTIME_ALERTS",500)
        if ra.empty: st.caption("No real-time alerts yet.")
        else:
            st.dataframe(ra.head(300),hide_index=True,use_container_width=True)
            if "SEVERITY" in ra.columns:
                sc2=ra["SEVERITY"].value_counts().reset_index(); sc2.columns=["Severity","Count"]
                st.bar_chart(sc2,x="Severity",y="Count")
            if "ALERT_TYPE" in ra.columns:
                at=ra["ALERT_TYPE"].value_counts().reset_index(); at.columns=["Alert Type","Count"]
                st.bar_chart(at,x="Alert Type",y="Count")
            if {"SESSION_ID","ESCALATION_LEVEL"}.issubset(ra.columns):
                peak=ra.groupby("SESSION_ID",as_index=False)["ESCALATION_LEVEL"].max()
                peak.columns=["Session","Peak Escalation"]
                st.bar_chart(peak,x="Session",y="Peak Escalation")

    with tabs[4]:
        uf=fetch_recent("UNIFIED_FRAME_PREDICTIONS",3000)
        if uf.empty: st.caption("No frame predictions yet.")
        else:
            st.dataframe(uf.head(300),hide_index=True,use_container_width=True)
            for col,label in [("OVERALL_RISK_LABEL","Risk Label"),("ZONE_PRED","Gaze Zone"),("RISK_GROUP_PRED","Risk Group")]:
                if col in uf.columns:
                    rc=uf[col].value_counts().reset_index(); rc.columns=[label,"Count"]
                    st.bar_chart(rc,x=label,y="Count")

    with tabs[5]:
        zt=fetch_recent("MODULE_B_ZONE_TRANSITIONS",1000)
        if zt.empty: st.caption("No zone transitions yet.")
        else:
            st.dataframe(zt.head(300),hide_index=True,use_container_width=True)
            if {"FROM_ZONE","TO_ZONE"}.issubset(zt.columns):
                pair=(zt["FROM_ZONE"]+" → "+zt["TO_ZONE"]).value_counts().head(10).reset_index()
                pair.columns=["Transition","Count"]
                st.bar_chart(pair,x="Transition",y="Count")

    with tabs[6]:
        rq=fetch_recent("FRAME_REVIEW_QUEUE",1000)
        if rq.empty: st.caption("No review queue items.")
        else:
            st.dataframe(rq.head(300),hide_index=True,use_container_width=True)
            if "DRIVER_ID" in rq.columns:
                bd=rq["DRIVER_ID"].value_counts().reset_index(); bd.columns=["Driver","Review Frames"]
                st.bar_chart(bd,x="Driver",y="Review Frames")


# =============================================================================
# NAVIGATION
# =============================================================================
pages = [
    st.Page(home_page,      title="Home",               icon=":material/home:"),
    st.Page(unified_page,   title="Real-time Analysis", icon=":material/security:"),
    st.Page(analytics_page, title="Historical Analytics",icon=":material/analytics:"),
]
pg = st.navigation(pages)
pg.run()
