# =============================================================================
# DRIVER SAFETY ANALYTICS PLATFORM  —  v5.0.0
# =============================================================================
# NEW in v5 (built on v4):
#
# REAL-TIME UI
#   • Structured live dashboard: large annotated frame + always-visible 2×3
#     metrics grid with colour-coded backgrounds that change as risk rises
#   • Progress bar now shows ETA, running alerts/hr rate, and trend direction
#   • Lighting badge overlay: "☀ Glare Correction Active", "🌙 Night Mode", etc.
#   • Side-by-side before/after lighting correction panel when correction fires
#   • Pulsing CSS alert banner (animated) coloured by alert type
#   • Blink event indicator on HUD
#
# ALERT SYSTEM
#   • Minimum 3-second sustained signal before alert fires (no single-frame spikes)
#   • Consensus check: 2 of 3 signals (A1 drowsy, A2 eye-closed, EAR) must agree
#   • Dynamic confidence floor: raised to 0.65 under active lighting correction
#   • Contextual suppression: mirror-check cadence and isolated yawns suppressed
#   • Alert fatigue guard: >15 alerts in 10 min → conservative mode (Critical only)
#   • Graded output: operational alert + plain-language recommendation + evidence
#   • Recovery events written to DRIVER_RECOVERY_EVENTS Snowflake table
#
# ANALYTICS & SNOWFLAKE
#   • Fleet Overview tab: time-of-day heatmap, session duration vs escalation scatter
#   • Driver Trend tab: week-over-week safety trend, personal PERCLOS baseline
#   • Lighting Quality Report: correction breakdown + impact on alert confidence
#   • Alert Precision Retrospective: real-time alerts vs confirmed post-hoc events
#   • Snowflake Dynamic Table DDL (DRIVER_RISK_LEADERBOARD) shown + queried
#   • Snowflake Cortex FORECAST query demonstrated in-app
#   • Time Travel query panel for audit history
#   • DRIVER_WEEKLY_TREND view DDL shown + queried
#   • Recovery events tab in historical analytics
#
# SNOWFLAKE SCHEMA CHANGES REQUIRED (run all before deploying):
#   -- From v4 (if not already run):
#   CREATE TABLE IF NOT EXISTS DEMO_DB.PUBLIC.REALTIME_ALERTS (
#       ALERT_ID VARCHAR(64), SESSION_ID VARCHAR(64), DRIVER_ID VARCHAR(64),
#       TRIP_ID VARCHAR(64), ALERT_TIMESTAMP_SECONDS FLOAT,
#       ALERT_WALL_TIME VARCHAR(64), ALERT_TYPE VARCHAR(64),
#       SEVERITY VARCHAR(16), SEVERITY_SCORE FLOAT, ESCALATION_LEVEL INT,
#       FATIGUE_SCORE FLOAT, OFFROAD_PROB FLOAT, ZONE_AT_ALERT VARCHAR(32),
#       CONFIDENCE FLOAT, MESSAGE VARCHAR(512), RECOMMENDATION VARCHAR(512),
#       CONSENSUS_SIGNALS VARCHAR(128), LIGHTING_AT_ALERT VARCHAR(32),
#       CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
#   );
#   ALTER TABLE DEMO_DB.PUBLIC.MODULE_A_FRAME_PREDICTIONS
#       ADD COLUMN IF NOT EXISTS LIGHTING_METHOD VARCHAR(32);
#   ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
#       ADD COLUMN IF NOT EXISTS MEAN_MODEL_CONFIDENCE FLOAT;
#   ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
#       ADD COLUMN IF NOT EXISTS TOTAL_REALTIME_ALERTS INT;
#   -- New in v5:
#   CREATE TABLE IF NOT EXISTS DEMO_DB.PUBLIC.DRIVER_RECOVERY_EVENTS (
#       RECOVERY_ID VARCHAR(64), SESSION_ID VARCHAR(64), DRIVER_ID VARCHAR(64),
#       TRIP_ID VARCHAR(64), RECOVERY_TIMESTAMP_SECONDS FLOAT,
#       PEAK_ESCALATION_BEFORE INT, RECOVERY_DURATION_SECONDS FLOAT,
#       PREVIOUS_STATE VARCHAR(64), ALERT_COUNT_BEFORE INT,
#       CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
#   );
#   CREATE OR REPLACE VIEW DEMO_DB.PUBLIC.DRIVER_WEEKLY_TREND AS
#   SELECT DRIVER_ID,
#          DATE_TRUNC('week', TO_TIMESTAMP(SESSION_START_TS)) AS WEEK,
#          AVG(COMBINED_DRIVER_SAFETY_SCORE) AS AVG_SCORE,
#          SUM(TOTAL_REALTIME_ALERTS) AS WEEKLY_ALERTS,
#          COUNT(DISTINCT SESSION_ID) AS TRIPS
#   FROM DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
#   GROUP BY 1, 2;
# =============================================================================

import io, os, tempfile, time, uuid, hashlib
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ── optional deps ──────────────────────────────────────────────────────────────
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

# ── constants ──────────────────────────────────────────────────────────────────
DATABASE        = "DEMO_DB"
SCHEMA          = "PUBLIC"
APP_VERSION     = "5.0.0"
MODEL_VER_A1    = "a1_fold1.pt"
MODEL_VER_A2    = "a2_fold1.pt"
MODEL_VER_B     = "best_lisa_fold5_calibrated.pt"
_MODEL_DIR      = tempfile.mkdtemp()

ZONE_CLASSES = ["Forward","Lap","Left Mirror","Radio","Rearview","Right Mirror","Shoulder","Speedometer"]
RISK_GROUPS  = {"Safe":["Forward"],"LowRisk":["Left Mirror","Right Mirror","Rearview"],
                "HighRisk":["Lap","Radio","Speedometer","Shoulder"]}
MIRROR_ZONES = {"Left Mirror","Right Mirror","Rearview"}

OFFROAD_THRESHOLD       = 0.35
ZONE_TEMPERATURE        = 2.600605
NORM_MEAN               = [0.485, 0.456, 0.406]
NORM_STD                = [0.229, 0.224, 0.225]
IMG_SIZE                = 224
MIN_ALERT_CONF          = 0.45
MIN_ALERT_CONF_LIT      = 0.65   # raised when lighting correction is active
MAX_UNCERTAIN_ENTROPY   = 1.55
MIN_FRAME_QUALITY       = 0.35
EAR_CLOSED_THRESHOLD    = 0.21
POOR_QUALITY_RATIO      = 0.35
OFFROAD_EXIT_HYSTERESIS = 0.08
CHUNK_SECONDS           = 5
ALERT_COOLDOWN_S        = 25
SUSTAINED_ALERT_S       = 3.0    # signal must last ≥ 3 s before alerting
ALERT_FATIGUE_LIMIT     = 15     # alerts in 10 min → conservative mode
ALERT_FATIGUE_WINDOW    = 600    # 10 minutes
DE_ESCALATE_SAFE_S      = 120    # seconds of safe driving before de-escalation
DISPLAY_WIDTH           = 620

MP_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
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
LIGHT_BADGE = {"overexposed_correction":"☀️ Glare Correction","underexposed_correction":"🌙 Night Mode",
               "low_contrast_correction":"🌫️ Contrast Boost","normal":"✅ Normal Lighting",
               "none":"","error":"⚠️ Lighting Error"}

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
            self.h_d = nn.Linear(nf,1); self.h_y = nn.Linear(nf,1); self.h_n = nn.Linear(nf,1)
        def forward(self, x):
            f = self.bb(x)
            return {"drowsy":torch.sigmoid(self.h_d(f)).squeeze(-1),
                    "yawn":  torch.sigmoid(self.h_y(f)).squeeze(-1),
                    "nod":   torch.sigmoid(self.h_n(f)).squeeze(-1)}

    class _A2Model(nn.Module):
        def __init__(self):
            super().__init__()
            import torchvision.models as m
            bb = m.resnet18(weights=None); nf = bb.fc.in_features; bb.fc = nn.Identity()
            self.bb = bb; self.h_e = nn.Linear(nf,1)
        def forward(self, x):
            f = self.bb(x)
            return {"eye_closed":torch.sigmoid(self.h_e(f)).squeeze(-1)}

    class _BModel(nn.Module):
        def __init__(self, bb, nf):
            super().__init__()
            self.bb = bb
            self.zone_head    = nn.Linear(nf,8)
            self.offroad_head = nn.Linear(nf,1)
        def forward(self, x):
            f = self.bb(x)
            return {"zone_logits":self.zone_head(f),"offroad_logit":self.offroad_head(f)}

@st.cache_resource
def load_a1():
    if not TORCH_AVAILABLE: return None
    p = _download_model(MODEL_VER_A1)
    if not p: return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, nn.Module): ckpt.eval(); return ckpt
        mdl = _A1Model()
        sd = ckpt.get("state_dict",ckpt) if isinstance(ckpt,dict) else ckpt
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
        sd = ckpt.get("state_dict",ckpt) if isinstance(ckpt,dict) else ckpt
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
        sd = ckpt.get("state_dict",ckpt) if isinstance(ckpt,dict) else ckpt
        mdl.load_state_dict(sd, strict=False); mdl.eval(); return mdl
    except Exception: return None

# =============================================================================
# SNOWFLAKE UTILITIES
# =============================================================================
def utc_now_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def get_session(): return _session

def _download_model(filename):
    path = os.path.join(_MODEL_DIR, filename)
    if os.path.exists(path): return path
    s = get_session()
    if s is None: return None
    try:
        s.file.get(f"@{DATABASE}.{SCHEMA}.DRIVER_SAFETY_MODELS/{filename}", _MODEL_DIR)
        if os.path.exists(path): return path
        gz = path+".gz"
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
    for attempt in range(retries+1):
        try:
            s.write_pandas(up, table_name, database=DATABASE, schema=SCHEMA, overwrite=False)
            return True
        except Exception as e:
            if attempt == retries:
                st.toast(f"DB write warning ({table_name}): {e}", icon=":material/warning:")
                return False
            time.sleep(0.3*(attempt+1))
    return False

def query_sf(sql: str) -> pd.DataFrame:
    s = get_session()
    if s is None: return pd.DataFrame()
    try: return s.sql(sql).to_pandas()
    except Exception: return pd.DataFrame()

def fetch_recent(table: str, limit: int = 200) -> pd.DataFrame:
    for order in ["CREATED_AT DESC","CREATED_AT_CLIENT DESC",""]:
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

def _make_rng(source_name, frame_idx):
    key = f"{source_name}:{frame_idx}".encode()
    return np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:8], 16))

def get_risk_group(zone):
    for g, zones in RISK_GROUPS.items():
        if zone in zones: return g
    return "Unknown"

def _certainty_from_prob(p):
    if p is None: return 0.0
    return float(np.clip(abs(p-0.5)*2.0, 0.0, 1.0))

def _severity_bucket(score, duration):
    c = float(score + min(duration/4.0, 1.0)*0.25)
    if c >= 0.9: return "critical", min(1.0, c)
    if c >= 0.7: return "high", c
    if c >= 0.45: return "medium", c
    return "low", c

def _alert_type_from_severity(sev):
    return {"critical":"dashboard_critical","high":"dashboard_alert",
            "medium":"dashboard_warning","low":"none"}.get(sev,"none")

def _fmt_ts(s: float) -> str:
    h=int(s//3600); m=int((s%3600)//60); sec=int(s%60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h>0 else f"{m:02d}:{sec:02d}"

def _recommendation_from_state(state: str, esc_level: int, perclos: float) -> str:
    if esc_level >= 4: return "🆘 STOP VEHICLE IMMEDIATELY — Extreme fatigue risk detected."
    if esc_level == 3: return "🔴 Urgent rest stop required within next 5 minutes."
    if state == "prolonged_eye_closure": return "🟠 Eyes closing detected — take a break soon."
    if state == "high_fatigue" and perclos > 0.20: return "🟠 High fatigue sustained — recommend rest within 15 min."
    if state == "repeated_distraction": return "🟡 Repeated distraction pattern — refocus on road."
    if state == "offroad_glance": return "🟡 Eyes off road — keep attention forward."
    if state == "mild_fatigue": return "🟢 Mild fatigue signal — monitor closely."
    return "✅ Normal driving detected."

# =============================================================================
# LIGHTING NORMALISATION
# =============================================================================
def normalize_lighting(frame_bgr):
    if not CV2_AVAILABLE or frame_bgr is None: return frame_bgr, "none"
    try:
        gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        std_b  = float(np.std(gray))
        if mean_b > 175:
            gamma = 1.8
            lut   = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)], dtype=np.uint8)
            out   = cv2.LUT(frame_bgr, lut)
            hsv   = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            hsv[:,:,2] = clahe.apply(hsv[:,:,2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "overexposed_correction"
        elif mean_b < 55:
            lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), "underexposed_correction"
        elif std_b < 22:
            lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), "low_contrast_correction"
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
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

def _brightness_score(gray):
    if gray is None or gray.size == 0: return 0.0
    return float(np.clip(1.0-abs(float(np.mean(gray))-128.0)/128.0, 0.0, 1.0))

def _blur_score(gray):
    if gray is None or gray.size == 0: return 0.0
    return float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var()/300.0, 0.0, 1.0))

def _safe_crop(img, x1, y1, x2, y2):
    h,w = img.shape[:2]
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
    ip  = np.array([_lm_xy(lm,1,w,h),_lm_xy(lm,152,w,h),_lm_xy(lm,33,w,h),
                    _lm_xy(lm,263,w,h),_lm_xy(lm,61,w,h),_lm_xy(lm,291,w,h)],dtype=np.float64)
    mp3 = np.array([(0,0,0),(0,-63.6,-12.5),(-43.3,32.7,-26.0),
                    (43.3,32.7,-26.0),(-28.9,-28.9,-24.1),(28.9,-28.9,-24.1)],dtype=np.float64)
    cm  = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float64)
    try:
        ok,rvec,tvec = cv2.solvePnP(mp3,ip,cm,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return None,None,None
        rm,_ = cv2.Rodrigues(rvec)
        _,_,_,_,_,_,euler = cv2.decomposeProjectionMatrix(np.hstack((rm,tvec)))
        return float(euler[0]),float(euler[1]),float(euler[2])
    except Exception: return None,None,None

def detect_face_eyes(frame_bgr):
    default = dict(face=None,eye_strip=None,left_eye=None,right_eye=None,
                   face_detected=False,preprocess_method="none",quality_score=0.0,
                   face_confidence=0.0,brightness_score=0.0,blur_score=0.0,
                   ear=None,left_ear=None,right_ear=None,head_pitch=None,head_yaw=None,head_roll=None)
    if not CV2_AVAILABLE: return default
    h,w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = _brightness_score(gray); blur = _blur_score(gray)
    mesh = get_face_mesh()
    if MP_AVAILABLE and mesh is not None:
        try:
            res = mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm  = res.multi_face_landmarks[0].landmark
                pts = np.array([[p.x*w, p.y*h] for p in lm], dtype=np.float32)
                x1,y1 = pts.min(axis=0); x2,y2 = pts.max(axis=0)
                face = _safe_crop(frame_bgr, x1-12, y1-12, x2+12, y2+12)
                lp   = np.array([_lm_xy(lm,i,w,h) for i in MP_LEFT_EYE],  dtype=np.float32)
                rp   = np.array([_lm_xy(lm,i,w,h) for i in MP_RIGHT_EYE], dtype=np.float32)
                le   = _crop_pts(frame_bgr, lp, 0.45)
                re   = _crop_pts(frame_bgr, rp, 0.45)
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
                area  = float(np.clip((x2-x1)*(y2-y1)/(w*h), 0.0, 1.0))
                quality = float(np.clip(0.40*blur+0.30*brightness+0.30*min(area/0.2,1.0), 0.0, 1.0))
                return dict(face=face,eye_strip=eye_strip,left_eye=le,right_eye=re,
                            face_detected=face is not None,preprocess_method="mediapipe",
                            quality_score=quality,face_confidence=min(1.0,0.5+quality/2.0),
                            brightness_score=brightness,blur_score=blur,
                            ear=float(ear),left_ear=float(l_ear),right_ear=float(r_ear),
                            head_pitch=pitch,head_yaw=yaw,head_roll=roll)
        except Exception: pass
    try:
        cp = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
        if os.path.exists(cp):
            fc = cv2.CascadeClassifier(cp); faces = fc.detectMultiScale(gray, 1.3, 5)
            if len(faces)>0:
                x,y,fw,fh = faces[0]; face = frame_bgr[y:y+fh, x:x+fw]
                strip = face[int(fh*0.15):int(fh*0.45), :]
                ar  = float(np.clip((fw*fh)/(w*h), 0.0, 1.0))
                q   = float(np.clip(0.45*blur+0.30*brightness+0.25*min(ar/0.2,1.0), 0.0, 1.0))
                return dict(face=face,eye_strip=strip,left_eye=None,right_eye=None,
                            face_detected=True,preprocess_method="haar",
                            quality_score=q,face_confidence=min(0.8,0.35+q/2.0),
                            brightness_score=brightness,blur_score=blur,
                            ear=None,left_ear=None,right_ear=None,head_pitch=None,head_yaw=None,head_roll=None)
    except Exception: pass
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
# INFERENCE
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
        d=_clamp01(float(o["drowsy"].item())); y=_clamp01(float(o["yawn"].item())); n=_clamp01(float(o["nod"].item()))
        return {"a1_prob_drowsy":d,"a1_prob_yawn":y,"a1_prob_nod":n,
                "a1_confidence":float(np.mean([_certainty_from_prob(d),_certainty_from_prob(y),_certainty_from_prob(n)]))}
    except Exception: return None

def infer_a2(mdl, strip):
    if mdl is None or strip is None or not PIL_AVAILABLE or not CV2_AVAILABLE: return None
    try:
        img = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
        t   = transforms.Compose([transforms.Resize((64,224)),transforms.ToTensor(),
                                   transforms.Normalize(NORM_MEAN,NORM_STD)])
        with torch.no_grad(): o = mdl(t(img).unsqueeze(0))
        e = _clamp01(float(o["eye_closed"].item()))
        return {"a2_prob_eye_closed":e,"a2_eye_openness_score":_clamp01(1.0-e),"a2_confidence":_certainty_from_prob(e)}
    except Exception: return None

def infer_b(mdl, frame, thr=OFFROAD_THRESHOLD):
    if mdl is None or not PIL_AVAILABLE or not CV2_AVAILABLE: return None
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad(): o = mdl(_prep_tensor(img, IMG_SIZE))
        zl  = o["zone_logits"][0]/ZONE_TEMPERATURE
        zp  = torch.softmax(zl,dim=0).detach().cpu().numpy()
        si  = np.argsort(zp)[::-1]; sp = zp[si]; zi = int(si[0])
        op  = _clamp01(float(torch.sigmoid(o["offroad_logit"][0]).item()))
        ent = float(-np.sum(zp*np.log(zp+1e-8))); conf = _clamp01(float(zp.max()))
        mar = float(sp[0]-sp[1]) if len(sp)>1 else conf
        unc = float(np.clip((ent/np.log(len(ZONE_CLASSES)))*0.6+(1.0-conf)*0.4, 0.0, 1.0))
        return {"zone_pred":ZONE_CLASSES[zi],"risk_group_pred":get_risk_group(ZONE_CLASSES[zi]),
                "offroad_prob":op,"offroad_pred":1 if op>=thr else 0,
                "confidence":conf,"entropy":ent,"margin":mar,"uncertainty_score":unc,
                "zone_top2":" | ".join([ZONE_CLASSES[int(i)] for i in si[:2]]),
                **{f"zone_prob_{z.lower().replace(' ','_')}":float(zp[i]) for i,z in enumerate(ZONE_CLASSES)}}
    except Exception: return None

def sim_a(rng):
    ec=float(rng.beta(2,5)); d=float(rng.beta(2,5)); y=float(rng.beta(1.5,8)); n=float(rng.beta(1.5,6))
    return {"a1_prob_drowsy":d,"a1_prob_yawn":y,"a1_prob_nod":n,
            "a2_prob_eye_closed":ec,"a2_eye_openness_score":float(1.0-ec),
            "a1_confidence":float(rng.uniform(0.45,0.85)),"a2_confidence":float(rng.uniform(0.45,0.85))}

def sim_b(rng, thr=OFFROAD_THRESHOLD):
    pr=rng.dirichlet([5,1,1.5,0.8,1.5,1.5,0.5,0.8]); zi=int(np.argmax(pr)); zone=ZONE_CLASSES[zi]
    op=float(np.clip(1-pr[0]+rng.normal(0,0.05),0,1)); conf=float(pr.max())
    ent=float(-np.sum(pr*np.log(pr+1e-8))); si=np.argsort(pr)[::-1]; sp=pr[si]
    mar=float(sp[0]-sp[1]); unc=float(np.clip((ent/np.log(len(ZONE_CLASSES)))*0.6+(1-conf)*0.4,0,1))
    return {"zone_pred":zone,"risk_group_pred":get_risk_group(zone),"offroad_prob":op,
            "offroad_pred":1 if op>=thr else 0,"confidence":conf,"entropy":ent,"margin":mar,
            "uncertainty_score":unc,"zone_top2":" | ".join([ZONE_CLASSES[int(i)] for i in si[:2]]),
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
# FULL TEMPORAL ANALYSIS (post-processing)
# =============================================================================
def temporal_a(df, cfg):
    if df.empty: return df,{},[]
    df=df.copy(); fps=max(1,int(cfg["fps"]))
    for c in ["a1_prob_drowsy","a1_prob_yawn","a1_prob_nod","a2_prob_eye_closed","a1_confidence","a2_confidence"]:
        if c not in df.columns: df[c]=0.0
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0).clip(0,1)
    df["ear"]=pd.to_numeric(df.get("ear",np.nan),errors="coerce")
    df["quality_score"]=pd.to_numeric(df.get("quality_score",0.0),errors="coerce").fillna(0.0).clip(0,1)
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
        for sig in ["rd","re","ry","rn"]:
            col={"rd":"a1_prob_drowsy_sm","re":"a2_prob_eye_closed_sm","ry":"a1_prob_yawn_sm","rn":"a1_prob_nod_sm"}[sig]
            df[f"{sig}_{wn}"]=df[col].rolling(w,min_periods=1).mean()
    df["fatigue_signal_confidence"]=(0.35*df["preprocess_confidence"]+0.35*df["a1_confidence"]+0.30*df["a2_confidence"]).clip(0,1)
    df["uncertain_frame"]=((df["fatigue_signal_confidence"]<MIN_ALERT_CONF)|(df["quality_score"]<MIN_FRAME_QUALITY)).astype(int)
    df["review_flag"]=np.where(df["uncertain_frame"]==1,"needs_review","clear")
    nb=np.clip(df["blink_rate_per_min"]/20.0, 0.0, 1.0)
    df["fatigue_score_raw"]=(cfg["w_drowsy"]*df["a1_prob_drowsy_sm"]+cfg["w_eye_closed"]*df["a2_prob_eye_closed_sm"]+
                              cfg["w_nod"]*df["a1_prob_nod_sm"]+cfg["w_yawn"]*df["a1_prob_yawn_sm"]+
                              cfg["w_perclos"]*df["perclos_30s"]+cfg["w_blink"]*nb).clip(0,1)
    df["fatigue_score_calibrated"]=(df["fatigue_score_raw"]*(0.65+0.35*df["preprocess_confidence"])).clip(0,1)
    df["frame_risk"]=np.where(df["uncertain_frame"]==1,df["fatigue_score_calibrated"]*0.70,df["fatigue_score_calibrated"]).clip(0,1)
    ta_adj=np.clip(cfg["t_alert"] +(0.5-df["preprocess_confidence"])*0.10, 0.5, 0.9)
    tc_adj=np.clip(cfg["t_caution"]+(0.5-df["preprocess_confidence"])*0.08, 0.3, 0.8)
    df["risk_level"]=np.where(df["frame_risk"]>=ta_adj,"high",np.where(df["frame_risk"]>=tc_adj,"medium","low"))
    df.loc[df["uncertain_frame"]==1,"risk_level"]=df.loc[df["uncertain_frame"]==1,"risk_level"]+"_uncertain"
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
# TIMELINE + SCORECARD
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
    keep=["frame_id","timestamp_seconds","frame_risk","eye_closed_binary","uncertain_frame","offroad_pred","risk_group_pred","attn_uncertain"]
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
# ALERT MANAGER v5  (sustained signals + consensus + recovery tracking)
# =============================================================================
class AlertManager:
    """
    v5 improvements:
      • Minimum sustained duration (SUSTAINED_ALERT_S) before any alert fires
      • Signal consensus check: requires ≥2 of 3 fatigue signals to agree
      • Dynamic confidence floor: raised when lighting correction is active
      • Contextual suppression: mirror cadence, isolated yawns
      • Alert fatigue guard: >15 alerts in 10 min → conservative (Critical only)
      • Recovery events tracked and written to Snowflake
    """
    def __init__(self, cooldown_s=ALERT_COOLDOWN_S):
        self.cooldown_s      = cooldown_s
        self.last_fired      = {}
        self.alert_history   = []
        self.recovery_events = []
        self.esc_level       = 0
        self._esc_ts         = deque()
        self._safe_since     = None
        self._peak_before    = 0
        self._prev_state     = "normal_driving"
        self._alerts_in_window = deque()  # for fatigue-guard window
        self.conservative_mode = False

    # ── sustained signal buffer ────────────────────────────────────────────────
        self._sustained: dict = {}   # alert_type → first_seen_ts

    def _check_sustained(self, alert_type, ts):
        if alert_type not in self._sustained:
            self._sustained[alert_type] = ts; return False
        return (ts - self._sustained[alert_type]) >= SUSTAINED_ALERT_S

    def _reset_sustained(self, alert_type):
        self._sustained.pop(alert_type, None)

    # ── mirror cadence suppressor ──────────────────────────────────────────────
    def _is_mirror_cadence(self, zone, ts) -> bool:
        """Returns True when zone is a mirror and the pattern is brief + regular."""
        if zone not in MIRROR_ZONES: return False
        mirror_alerts = [a for a in self.alert_history if a.get("zone_at_alert") in MIRROR_ZONES]
        if len(mirror_alerts) < 2: return True  # too few data → suppress as routine
        gaps = [mirror_alerts[i]["alert_timestamp_seconds"] - mirror_alerts[i-1]["alert_timestamp_seconds"]
                for i in range(1, len(mirror_alerts))]
        avg_gap = float(np.mean(gaps[-5:])) if gaps else 999
        return avg_gap > 15.0   # regular check every 15+ s → treat as safe mirror use

    # ── consensus check: ≥2 of {a1_drowsy, a2_eye_closed, ear_closed} agree ──
    @staticmethod
    def _check_consensus(cs: dict) -> tuple[bool, list]:
        signals = []
        if float(cs.get("frame_risk",0)) >= 0.55: signals.append("a1_drowsy_high")
        if int(cs.get("eye_closed",0)) == 1:       signals.append("a2_eye_closed")
        ear = cs.get("ear")
        if ear is not None and ear <= EAR_CLOSED_THRESHOLD: signals.append("ear_below_threshold")
        return len(signals) >= 2, signals

    # ── alert fatigue guard ────────────────────────────────────────────────────
    def _check_fatigue_guard(self, ts):
        cutoff = ts - ALERT_FATIGUE_WINDOW
        while self._alerts_in_window and self._alerts_in_window[0] < cutoff:
            self._alerts_in_window.popleft()
        if len(self._alerts_in_window) >= ALERT_FATIGUE_LIMIT:
            if not self.conservative_mode:
                self.conservative_mode = True
                st.toast("⚠️ Alert fatigue guard active — switching to Critical-only mode.", icon="🛡️")
        else:
            self.conservative_mode = False

    # ── main fire method ───────────────────────────────────────────────────────
    def try_fire(self, alert_type, severity, severity_score, ts, message, confidence,
                 fatigue_score=0.0, offroad_prob=0.0, zone="Unknown",
                 cs: dict = None, lighting: str = "normal") -> dict | None:

        self._check_fatigue_guard(ts)
        if self.conservative_mode and severity not in ["critical"]:
            return None

        # Cooldown gate
        if ts - self.last_fired.get(alert_type, -999) < self.cooldown_s:
            return None

        # Sustained signal gate
        if not self._check_sustained(alert_type, ts):
            return None

        # Dynamic confidence floor
        min_conf = MIN_ALERT_CONF_LIT if lighting not in ["normal","none","error"] else MIN_ALERT_CONF
        if confidence < min_conf:
            return None

        # Consensus check (fatigue alerts only)
        consensus_signals = []
        if cs is not None and "drowsiness" in alert_type:
            ok, consensus_signals = self._check_consensus(cs)
            if not ok:
                return None

        # Mirror cadence suppression
        if alert_type == "distraction_alert" and self._is_mirror_cadence(zone, ts):
            return None

        # Isolated yawn suppression: yawn without eye closure or high drowsy
        if alert_type == "drowsiness_alert" and cs is not None:
            fr  = float(cs.get("frame_risk",0))
            yaw = float(cs.get("a1_prob_yawn",0) if cs.get("a1_prob_yawn") else 0)
            if yaw > 0.5 and fr < 0.45 and int(cs.get("eye_closed",0))==0:
                return None

        self.last_fired[alert_type] = ts
        self._reset_sustained(alert_type)
        self._alerts_in_window.append(ts)

        rec = _recommendation_from_state(
            cs.get("state","normal_driving") if cs else "normal_driving",
            self.esc_level, float(cs.get("perclos",0)) if cs else 0.0)

        alert = {"alert_id":str(uuid.uuid4())[:16],"alert_type":alert_type,
                 "severity":severity,"severity_score":severity_score,
                 "alert_timestamp_seconds":ts,"alert_wall_time":utc_now_iso(),
                 "message":message,"confidence":round(confidence,3),
                 "fatigue_score":round(fatigue_score,3),"offroad_prob":round(offroad_prob,3),
                 "zone_at_alert":zone,"escalation_level":self.esc_level,
                 "recommendation":rec,"consensus_signals":"|".join(consensus_signals),
                 "lighting_at_alert":lighting}
        self.alert_history.append(alert)
        if severity in ["medium","high","critical"]:
            self._esc_ts.append(ts); self._safe_since = None
            self._peak_before = max(self._peak_before, self.esc_level)
            self._update_escalation(ts)
        return alert

    def mark_safe_frame(self, ts, current_state):
        if self._prev_state not in ["normal_driving","mild_fatigue"] and current_state == "normal_driving":
            # just recovered
            if self.esc_level > 0 and self._safe_since is None:
                recent = [a for a in self.alert_history if ts-a["alert_timestamp_seconds"] < 300]
                self.recovery_events.append({
                    "recovery_id":str(uuid.uuid4())[:16],
                    "recovery_timestamp_seconds":ts,
                    "peak_escalation_before":self._peak_before,
                    "recovery_duration_seconds":0.0,
                    "previous_state":self._prev_state,
                    "alert_count_before":len(recent)})
        self._prev_state = current_state
        if current_state == "normal_driving":
            if self._safe_since is None: self._safe_since = ts
            elif ts - self._safe_since >= DE_ESCALATE_SAFE_S:
                if self.esc_level > 0: self.esc_level -= 1
                self._safe_since = ts; self._peak_before = 0

    def _update_escalation(self, now):
        while self._esc_ts and now - self._esc_ts[0] > 300: self._esc_ts.popleft()
        n = len(self._esc_ts)
        if   n >= 8: self.esc_level = 4
        elif n >= 5: self.esc_level = 3
        elif n >= 3: self.esc_level = 2
        elif n >= 1: self.esc_level = 1

    @property
    def total_alerts(self): return len(self.alert_history)

    def alerts_df(self): return pd.DataFrame(self.alert_history) if self.alert_history else pd.DataFrame()
    def recovery_df(self): return pd.DataFrame(self.recovery_events) if self.recovery_events else pd.DataFrame()

# =============================================================================
# ROLLING BUFFER
# =============================================================================
class RollingBuffer:
    def __init__(self, max_size=300):
        self._a: deque = deque(maxlen=max_size)
        self._b: deque = deque(maxlen=max_size)
    def push_a(self, r): self._a.append(r)
    def push_b(self, r): self._b.append(r)
    def df_a(self): return pd.DataFrame(list(self._a)) if self._a else pd.DataFrame()
    def df_b(self): return pd.DataFrame(list(self._b)) if self._b else pd.DataFrame()

# =============================================================================
# CHUNK STATE COMPUTATION
# =============================================================================
def compute_current_state(roll_a, roll_b, cfg) -> dict:
    out = {"state":"normal_driving","frame_risk":0.0,"perclos":0.0,"eye_closed":0,
           "offroad_prob":0.0,"zone":"Unknown","confidence":0.5,"ear":None,
           "fatigue_conf":0.5,"attn_uncertain":False,"a1_prob_yawn":0.0,"ts":0.0}
    fps = max(1, cfg["fps"])
    if not roll_a.empty:
        df=roll_a.copy()
        for c in ["a1_prob_drowsy","a2_prob_eye_closed","a1_confidence","a2_confidence",
                  "quality_score","face_confidence","a1_prob_yawn"]:
            if c not in df.columns: df[c]=0.0
            df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
        sw=max(1,int(0.6*fps))
        df["d_sm"]=df["a1_prob_drowsy"].rolling(sw,min_periods=1).mean()
        df["e_sm"]=df["a2_prob_eye_closed"].rolling(sw,min_periods=1).mean()
        df["ear_"]=pd.to_numeric(df.get("ear",np.nan),errors="coerce")
        df["ec_bin"]=((df["e_sm"]>=cfg["t_eye"])|(df["ear_"].notna()&(df["ear_"]<=cfg["ear_closed"]))).astype(int)
        win_m=max(1,int(cfg["win_m"]*fps))
        perclos=float(df["ec_bin"].rolling(win_m,min_periods=1).mean().iloc[-1])
        last=df.iloc[-1]
        pc=float(float(last.get("quality_score",0))*0.6+float(last.get("face_confidence",0))*0.4)
        fr=float(np.clip(cfg["w_drowsy"]*float(last.get("d_sm",0))+
                         cfg["w_eye_closed"]*float(last.get("e_sm",0))+cfg["w_perclos"]*perclos,0,1))
        fr_cal=float(np.clip(fr*(0.65+0.35*pc),0,1))
        fc=float((float(last.get("a1_confidence",0.5))*0.5+float(last.get("a2_confidence",0.5))*0.5))
        out.update({"frame_risk":fr_cal,"perclos":perclos,"eye_closed":int(df["ec_bin"].iloc[-1]),
                    "ear":float(last.get("ear_")) if pd.notna(last.get("ear_",None)) else None,
                    "fatigue_conf":float(np.clip(pc*0.4+fc*0.6,0,1)),
                    "a1_prob_yawn":float(last.get("a1_prob_yawn",0))})
    if not roll_b.empty:
        df2=roll_b.copy()
        for c in ["offroad_prob","confidence","uncertainty_score"]:
            if c not in df2.columns: df2[c]=0.0
            df2[c]=pd.to_numeric(df2[c],errors="coerce").fillna(0.0)
        last2=df2.iloc[-1]
        out.update({"offroad_prob":float(last2.get("offroad_prob",0)),
                    "zone":str(last2.get("zone_pred","Unknown")),
                    "confidence":float(last2.get("confidence",0.5)),
                    "attn_uncertain":bool(float(last2.get("uncertainty_score",0))>0.65 or
                                         float(last2.get("confidence",1))<MIN_ALERT_CONF)})
    fr=out["frame_risk"]
    if out.get("attn_uncertain") and out.get("eye_closed"):     out["state"]="needs_review"
    elif out["eye_closed"] and fr>=0.55:                        out["state"]="prolonged_eye_closure"
    elif fr>=cfg["t_alert"]:                                    out["state"]="high_fatigue"
    elif fr>=cfg["t_caution"]:                                  out["state"]="mild_fatigue"
    elif out["offroad_prob"]>=OFFROAD_THRESHOLD and not out.get("attn_uncertain"):
        out["state"]="repeated_distraction" if get_risk_group(out["zone"])=="HighRisk" else "offroad_glance"
    return out

# =============================================================================
# FRAME ANNOTATION  (v5: lighting badge, blink indicator, before/after helper)
# =============================================================================
def annotate_frame(frame_bgr, state, cs, esc_level, alert_active, light_method="normal", blink_active=False):
    if not CV2_AVAILABLE or frame_bgr is None: return frame_bgr
    f=frame_bgr.copy(); h,w=f.shape[:2]; font=cv2.FONT_HERSHEY_SIMPLEX
    border_bgr={"normal_driving":(0,200,0),"mild_fatigue":(0,165,255),"high_fatigue":(0,0,255),
                "offroad_glance":(0,130,255),"repeated_distraction":(0,0,220),
                "prolonged_eye_closure":(0,0,255),"needs_review":(128,128,128)}.get(state,(0,200,0))
    cv2.rectangle(f,(0,0),(w-1,h-1),border_bgr,7 if alert_active else 4)
    ov=f.copy(); cv2.rectangle(ov,(0,0),(min(w,310),200),(0,0,0),-1)
    cv2.addWeighted(ov,0.50,f,0.50,0,f)
    sc={"normal_driving":(50,220,50),"mild_fatigue":(0,180,255),"high_fatigue":(0,60,255),
        "prolonged_eye_closure":(0,0,255),"repeated_distraction":(0,0,200)}.get(state,(220,220,220))
    cv2.putText(f,state.upper().replace("_"," "),(8,22),font,0.46,sc,1,cv2.LINE_AA)
    cv2.putText(f,f"FATIGUE:{cs['frame_risk']:.2f}  CONF:{cs['fatigue_conf']:.2f}",(8,44),font,0.40,(220,220,220),1,cv2.LINE_AA)
    cv2.putText(f,f"OFFROAD:{cs['offroad_prob']:.2f}  ZONE:{cs['zone']}",(8,64),font,0.40,(220,220,220),1,cv2.LINE_AA)
    cv2.putText(f,f"PERCLOS:{cs['perclos']:.1%}",(8,84),font,0.40,(220,220,220),1,cv2.LINE_AA)
    ear_txt=f"EAR:{cs['ear']:.3f}" if cs.get("ear") is not None else "EAR:N/A"
    cv2.putText(f,ear_txt,(8,104),font,0.40,(220,220,220),1,cv2.LINE_AA)
    esc_bgr=[(20,180,20),(20,165,0),(0,130,255),(0,0,220),(0,0,160)][min(esc_level,4)]
    esc_txt=["NORMAL","LOW","MEDIUM","HIGH","CRITICAL"][min(esc_level,4)]
    cv2.rectangle(f,(8,110),(120,128),esc_bgr,-1)
    cv2.putText(f,f"ESC:{esc_txt}",(12,124),font,0.36,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(f,f"T:{cs.get('ts',0):.1f}s",(8,148),font,0.36,(180,180,180),1,cv2.LINE_AA)
    # Lighting badge
    badge=LIGHT_BADGE.get(light_method,"")
    if badge and light_method!="normal":
        badge_bgr={"overexposed_correction":(0,165,255),"underexposed_correction":(130,0,130),
                   "low_contrast_correction":(180,180,0)}.get(light_method,(100,100,100))
        cv2.rectangle(f,(8,152),(240,170),badge_bgr,-1)
        cv2.putText(f,badge[:28],(12,166),font,0.32,(255,255,255),1,cv2.LINE_AA)
    # Blink indicator
    if blink_active:
        cv2.circle(f,(w-24,24),12,(0,255,255),-1)
        cv2.putText(f,"B",(w-31,30),font,0.45,(0,0,0),1,cv2.LINE_AA)
    if alert_active:
        ov2=f.copy(); cv2.rectangle(ov2,(0,h-50),(w,h),(0,0,200),-1)
        cv2.addWeighted(ov2,0.65,f,0.35,0,f)
        cv2.putText(f,"ALERT — DRIVER NEEDS ATTENTION",(max(4,w//2-175),h-16),font,0.60,(255,255,255),2,cv2.LINE_AA)
    return f

def make_lighting_comparison(original_bgr, corrected_bgr, method):
    """Stack original and corrected frames side by side for before/after display."""
    if not CV2_AVAILABLE or original_bgr is None or corrected_bgr is None: return None
    try:
        th = 200
        w1 = int(original_bgr.shape[1] * th / original_bgr.shape[0])
        w2 = int(corrected_bgr.shape[1] * th / corrected_bgr.shape[0])
        o  = cv2.resize(original_bgr,(w1,th))
        c  = cv2.resize(corrected_bgr,(w2,th))
        ww = max(w1,w2)
        cv2.putText(o,"BEFORE",(4,18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)
        lbl = LIGHT_BADGE.get(method,"Corrected")
        cv2.putText(c,f"AFTER: {lbl}",(4,18),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,128),1)
        if o.shape[1]<ww: o=cv2.copyMakeBorder(o,0,0,0,ww-o.shape[1],cv2.BORDER_CONSTANT)
        if c.shape[1]<ww: c=cv2.copyMakeBorder(c,0,0,0,ww-c.shape[1],cv2.BORDER_CONSTANT)
        return cv2.hconcat([o,c])
    except Exception: return None

# =============================================================================
# ANALYTICS BUILDERS
# =============================================================================
def build_unified_frame_predictions(df_a, df_b, sid, driver_id, trip_id, src):
    if df_a.empty and df_b.empty: return pd.DataFrame()
    if not df_a.empty and not df_b.empty:
        mg=pd.merge(df_a,df_b,left_on="frame_id",right_on="frame_index",how="outer",suffixes=("_a","_b"))
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
    for col,dflt in [("risk_level","unknown"),("zone_pred","Unknown"),("risk_group_pred","Unknown")]:
        if col not in mg.columns: mg[col]=dflt
    for col in ["uncertain_frame","attn_uncertain"]:
        if col not in mg.columns: mg[col]=1
    mg["overall_risk_score"]=(0.50*mg["frame_risk"]+0.35*mg["offroad_prob"]+0.15*(1.0-mg["confidence"])).clip(0,1)
    mg["needs_review"]=((mg["uncertain_frame"].astype(int)==1)|(mg["attn_uncertain"].astype(int)==1)).astype(int)
    mg["overall_risk_label"]=np.select([mg["needs_review"]==1,mg["overall_risk_score"]>=0.70,mg["overall_risk_score"]>=0.40],
                                        ["needs_review","high","medium"],default="low")
    for k,v in [("session_id",sid),("driver_id",driver_id),("trip_id",trip_id),("source_file_name",src),("created_at_client",utc_now_iso())]:
        mg[k]=v
    keep=["session_id","driver_id","trip_id","source_file_name","frame_id","frame_index","timestamp_seconds",
          "overall_risk_score","overall_risk_label","needs_review","frame_risk","risk_level","offroad_prob",
          "offroad_pred","zone_pred","risk_group_pred","confidence","uncertainty_score","uncertain_frame","attn_uncertain",
          "a1_prob_drowsy","a2_prob_eye_closed","perclos_30s","blink_rate_per_min","quality_score","face_confidence",
          "preprocess_method","inference_mode_a","inference_mode_b","created_at_client"
          ]+[f"zone_prob_{z.lower().replace(' ','_')}" for z in ZONE_CLASSES]
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
                           "avg_frame_quality":qa,"avg_face_confidence":fa,"fatigue_uncertain_ratio":ua,
                           "attn_uncertain_ratio":ub,"dominant_preprocess_method":pm,
                           "lighting_corrections_applied":str(lm_counts),
                           "overall_quality_score":float(np.clip(0.4*qa+0.3*fa+0.3*(1-max(ua,ub)),0,1)),
                           "created_at_client":utc_now_iso()}])

# =============================================================================
# HTML TIMELINE + PULSING ALERT BANNER
# =============================================================================
def render_timeline_html(timeline_rows, total_dur):
    if not timeline_rows or total_dur <= 0: return ""
    bars=""
    for r in timeline_rows:
        dur=r.get("duration_seconds",0); pct=max(0.1,dur/total_dur*100)
        col=STATE_COLORS.get(r.get("timeline_state","normal_driving"),"#22c55e")
        s=r.get("start_ts",0); e=r.get("end_ts",0)
        lbl=r.get("timeline_state","").replace("_"," ").title()
        tip=f"{lbl} | {_fmt_ts(s)} → {_fmt_ts(e)} ({dur:.1f}s)"
        bars+=f'<div title="{tip}" style="display:inline-block;width:{pct:.3f}%;height:44px;background:{col};cursor:pointer;vertical-align:top;border-right:1px solid rgba(0,0,0,0.15);"></div>'
    legend="".join([f'<span style="display:inline-flex;align-items:center;margin-right:12px;font-size:11px;">'
                    f'<span style="background:{col};display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;"></span>'
                    f'{lbl.replace("_"," ").title()}</span>' for lbl,col in STATE_COLORS.items()])
    return (f'<div style="margin:8px 0 4px 0;">'
            f'<div style="width:100%;border-radius:4px;overflow:hidden;border:1px solid #333;">{bars}</div>'
            f'<div style="margin-top:8px;display:flex;flex-wrap:wrap;">{legend}</div></div>')

def _pulsing_alert_html(msg, severity, esc_level):
    bg={"critical":"#7f1d1d","high":"#ef4444","medium":"#f97316","low":"#f59e0b"}.get(severity,"#ef4444")
    icon={"critical":"🆘","high":"🚨","medium":"🟠","low":"⚠️"}.get(severity,"🚨")
    return f"""
<style>
@keyframes pulse{{0%{{opacity:1}}50%{{opacity:0.55}}100%{{opacity:1}}}}
.alert-pulse{{animation:pulse 1.2s ease-in-out infinite;}}
</style>
<div class="alert-pulse" style="background:{bg};color:white;padding:10px 16px;border-radius:8px;
font-weight:bold;font-size:15px;margin:6px 0;">
{icon} {msg}<br>
<span style="font-size:12px;opacity:0.85;">Escalation: {ESC_LABELS[min(esc_level,4)]}</span>
</div>"""

# =============================================================================
# FRAME EXPLORER
# =============================================================================
def render_frame_explorer(frame_map, unified_df):
    st.subheader("🔍 Frame-Level Explorer")
    if unified_df.empty or not frame_map: st.caption("No data available."); return
    fdf=unified_df.copy()
    fdf["frame_id"]=pd.to_numeric(fdf["frame_id"],errors="coerce").fillna(-1).astype(int)
    fdf=fdf[fdf["frame_id"]>=0].sort_values("frame_id").reset_index(drop=True)
    sel=st.multiselect("Filter by risk label",["needs_review","high","medium","low"],
                       default=["needs_review","high","medium","low"],key="fe_risk")
    fdf=fdf[fdf["overall_risk_label"].isin(sel)]
    if fdf.empty: st.caption("No frames match."); return
    idx=st.slider("Pick frame",0,len(fdf)-1,0,key="fe_slider")
    r=fdf.iloc[idx]; fid=int(r["frame_id"])
    c1,c2=st.columns([2,3])
    with c1:
        if fid in frame_map:
            st.image(cv2.cvtColor(frame_map[fid],cv2.COLOR_BGR2RGB),
                     caption=f"Frame {fid} | {float(r['timestamp_seconds']):.2f}s",use_container_width=True)
    with c2:
        a,b,c,d=st.columns(4)
        a.metric("Overall Risk",f"{float(r['overall_risk_score']):.2f}")
        b.metric("Fatigue Risk",f"{float(r['frame_risk']):.2f}")
        c.metric("Off-road",f"{float(r['offroad_prob']):.2f}")
        d.metric("Confidence",f"{float(r['confidence']):.2f}")
        st.caption(f"Label: **{r['overall_risk_label']}** | Zone: **{r.get('zone_pred','?')}** | Review: {int(r.get('needs_review',0))}")
        zpc=[f"zone_prob_{z.lower().replace(' ','_')}" for z in ZONE_CLASSES if f"zone_prob_{z.lower().replace(' ','_')}" in fdf.columns]
        if zpc:
            zr=pd.DataFrame({"Zone":[c.replace("zone_prob_","").replace("_"," ").title() for c in zpc],
                             "Prob":[float(r[c]) for c in zpc]}).sort_values("Prob",ascending=False).head(4)
            st.bar_chart(zr,x="Zone",y="Prob")
    view_cols=["frame_id","timestamp_seconds","overall_risk_label","overall_risk_score","frame_risk",
               "risk_level","offroad_prob","zone_pred","risk_group_pred","confidence","uncertainty_score"]
    st.dataframe(fdf[[c for c in view_cols if c in fdf.columns]],hide_index=True,use_container_width=True,height=260)
    st.subheader("📸 Frame Gallery")
    pg_size=st.select_slider("Per page",[4,6,9,12],6,key="fe_ps")
    total_pg=max(1,int(np.ceil(len(fdf)/pg_size)))
    pg=st.number_input("Page",1,total_pg,1,key="fe_pg")
    sub=fdf.iloc[(pg-1)*pg_size:pg*pg_size]; cols=st.columns(3)
    for i,(_,row) in enumerate(sub.iterrows()):
        with cols[i%3]:
            fid2=int(row["frame_id"])
            if fid2 in frame_map: st.image(cv2.cvtColor(frame_map[fid2],cv2.COLOR_BGR2RGB),use_container_width=True)
            st.caption(f"F{fid2}|{float(row['timestamp_seconds']):.1f}s|{row['overall_risk_label']}|{row.get('zone_pred','?')}")

# =============================================================================
# VIDEO UTILITIES
# =============================================================================
def open_video(video_bytes, file_ext):
    tmp=tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False)
    tmp.write(video_bytes); tmp.close()
    cap=cv2.VideoCapture(tmp.name)
    vfps=cap.get(cv2.CAP_PROP_FPS) or 30.0; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return tmp.name,{"video_fps":float(vfps),"total_frames":int(total),
                     "duration_seconds":float(total/vfps) if vfps>0 else 0.0,
                     "width":W,"height":H,"resolution":f"{W}x{H}"}

# =============================================================================
# SNOWFLAKE BATCH WRITE
# =============================================================================
def _write_all_to_sf(df_a, df_b, a_events, b_events, a_sum, b_sum, scorecard,
                     uf_df, zt_df, quality_df, tl_rows, unified_sum, alert_mgr: AlertManager,
                     sid, driver_id, trip_id, ss_ts, se_ts, uploaded_name):
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
    def _ev_row(ev,mod):
        base={"event_id":str(uuid.uuid4())[:16],"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
              "event_type":ev["type"],"event_start_ts":ev["start"],"event_end_ts":ev["end"],
              "duration_seconds":ev["dur"],"severity":ev["severity"],"severity_score":ev["severity_score"],
              "confidence":ev["confidence"],"alert_sent":ev["alert_sent"],"alert_type":ev["alert_type"],
              "event_confirmation_status":ev["event_confirmation_status"],"risk_group":ev.get("risk_group",""),
              "created_at_client":utc_now_iso()}
        if mod=="a": base.update({"dominant_zone":"","explanation":f"{ev['type']} for {ev['dur']:.2f}s"})
        else: base.update({"dominant_zone":ev.get("dominant_zone","")})
        return base
    if a_events: write_to_snowflake(pd.DataFrame([_ev_row(e,"a") for e in a_events]),"MODULE_A_EVENTS")
    if b_events: write_to_snowflake(pd.DataFrame([_ev_row(e,"b") for e in b_events]),"MODULE_B_EVENTS")
    write_to_snowflake(pd.DataFrame([{
        "session_id":sid,"driver_id":driver_id,"trip_id":trip_id,"source_name":uploaded_name,"source_type":"video",
        "session_start_ts":ss_ts,"session_end_ts":se_ts,"duration_seconds":a_sum["total_duration"],
        "total_frames_processed":a_sum["total_frames"],"total_duration_seconds":a_sum["total_duration"],
        "avg_a1_prob_drowsy":a_sum["avg_drowsy"],"max_a1_prob_drowsy":a_sum["max_drowsy"],
        "avg_a2_prob_eye_closed":a_sum["avg_eye"],"eye_closure_burden":a_sum["eye_closure_burden"],
        "perclos":a_sum["perclos"],"blink_count":a_sum["blink_count"],"blink_freq_per_min":a_sum["blink_freq_per_min"],
        "avg_blink_duration_seconds":a_sum["avg_blink_duration"],"avg_ear":a_sum["avg_ear"],
        "prolonged_closure_count":a_sum["closure_count"],"yawn_support_score":a_sum["yawn_sup"],
        "nod_support_score":a_sum["nod_sup"],"total_high_risk_duration":a_sum["hr_dur"],
        "total_caution_duration":a_sum["caut_dur"],"mean_confidence":a_sum["mean_confidence"],
        "uncertain_ratio":a_sum["uncertain_ratio"],"poor_quality_session":a_sum["poor_quality_session"],
        "final_session_risk":a_sum["final"],"model_version_a1":MODEL_VER_A1,"model_version_a2":MODEL_VER_A2,
        "app_version":APP_VERSION,"created_at_client":utc_now_iso()}]),"MODULE_A_SESSION_SUMMARY")
    write_to_snowflake(pd.DataFrame([{
        "session_id":sid,"driver_id":driver_id,"trip_id":trip_id,"source_file_name":uploaded_name,
        "session_start_ts":ss_ts,"session_end_ts":se_ts,"duration_seconds":b_sum["td"],
        "total_frames":b_sum["nf"],"total_duration_seconds":b_sum["td"],"offroad_ratio":b_sum["or"],
        "max_offroad_streak_seconds":b_sum["mos"],"offroad_events_per_min":b_sum["oepm"],
        "highrisk_ratio":b_sum["hr"],"max_highrisk_streak_seconds":b_sum["mhs"],
        "highrisk_events_per_min":b_sum["hepm"],"mirror_glance_frequency_per_min":b_sum["mfpm"],
        "avg_mirror_glance_duration_seconds":b_sum["amd"],"safe_forward_ratio":b_sum["sfr"],
        "mean_confidence":b_sum["mc"],"mean_entropy":b_sum["me"],"uncertain_ratio":b_sum["uncertain_ratio"],
        "poor_quality_session":b_sum["poor_quality_session"],
        "repeated_distraction_events":b_sum["repeated_distraction_events"],
        "model_version_b":MODEL_VER_B,"app_version":APP_VERSION,"created_at_client":utc_now_iso()}]),"MODULE_B_SESSION_SUMMARY")
    write_to_snowflake(pd.DataFrame([unified_sum]),"UNIFIED_DRIVER_SESSION_SUMMARY")
    write_to_snowflake(pd.DataFrame([scorecard]),"DRIVER_SCORECARDS")
    write_to_snowflake(uf_df,"UNIFIED_FRAME_PREDICTIONS")
    if not zt_df.empty: write_to_snowflake(zt_df,"MODULE_B_ZONE_TRANSITIONS")
    write_to_snowflake(quality_df,"SESSION_DATA_QUALITY_SUMMARY")
    if tl_rows:
        tld=pd.DataFrame(tl_rows); tld["session_id"]=sid; tld["driver_id"]=driver_id
        tld["trip_id"]=trip_id; tld["created_at_client"]=utc_now_iso()
        write_to_snowflake(tld,"DRIVER_TIMELINE")
    rev_q=uf_df[uf_df["needs_review"]==1].copy() if not uf_df.empty else pd.DataFrame()
    if not rev_q.empty: write_to_snowflake(rev_q,"FRAME_REVIEW_QUEUE")
    # Recovery events
    rec_df = alert_mgr.recovery_df()
    if not rec_df.empty:
        rec_df["session_id"]=sid; rec_df["driver_id"]=driver_id; rec_df["trip_id"]=trip_id
        rec_df["created_at_client"]=utc_now_iso()
        write_to_snowflake(rec_df,"DRIVER_RECOVERY_EVENTS")

# =============================================================================
# STREAMING INFERENCE  (v5 core loop)
# =============================================================================
def run_streaming_inference(video_path, video_meta, cfg_a, fps_b, sw_b, offroad_thr,
                             driver_id, trip_id, source_name, sim_a_mode, sim_b_mode, sid):
    m1,m2,bm = load_a1(),load_a2(),load_b()
    vfps      = video_meta["video_fps"]
    total_vf  = video_meta["total_frames"]
    afps      = cfg_a["fps"]
    interval  = max(1,int(vfps/max(1,afps)))
    chunk_raw = max(interval,int(CHUNK_SECONDS*vfps))
    roll_size = max(60,int(60*afps))
    rolling   = RollingBuffer(max_size=roll_size)
    alert_mgr = AlertManager(cooldown_s=ALERT_COOLDOWN_S)
    all_rows_a: list=[]; all_rows_b: list=[]; frame_map: dict={}
    chunk_start_wall = time.time()
    chunks_done = 0

    # ── UI LAYOUT ──────────────────────────────────────────────────────────────
    st.markdown("### 🎬 Live Inference Dashboard")
    prog_bar    = st.progress(0,"Initialising…")
    alert_ph    = st.empty()

    # Main dashboard: frame left, 6-metric grid right
    col_frame, col_dash = st.columns([3,2])
    with col_frame:
        frame_ph   = st.empty()
        light_ph   = st.empty()   # before/after lighting panel
    with col_dash:
        st.markdown("#### 📊 Live Metrics")
        row1c1,row1c2 = st.columns(2)
        row2c1,row2c2 = st.columns(2)
        row3c1,row3c2 = st.columns(2)
        m_fat  = row1c1.empty(); m_esc  = row1c2.empty()
        m_off  = row2c1.empty(); m_zone = row2c2.empty()
        m_perc = row3c1.empty(); m_ear  = row3c2.empty()
        st.markdown("---")
        st.markdown("##### 📈 Progress")
        prog_stats = st.empty()

    st.markdown("---")
    st.markdown("#### 🚨 Real-time Alert Log")
    log_ph  = st.empty()
    st.markdown("#### 📉 Rolling Signals (last 60 s)")
    chart_ph = st.empty()

    # ── MAIN LOOP ──────────────────────────────────────────────────────────────
    cap=cv2.VideoCapture(video_path)
    global_raw_idx=0; global_samp_idx=0; chunk_idx=0
    last_light_method = "normal"
    last_original_frame = None

    while cap.isOpened():
        chunk_frames=[]
        for _ in range(chunk_raw):
            ret,frame=cap.read()
            if not ret: break
            if global_raw_idx%interval==0:
                ts=global_raw_idx/vfps
                chunk_frames.append((global_raw_idx,ts,frame,global_samp_idx))
                global_samp_idx+=1
            global_raw_idx+=1
        if not chunk_frames: break

        chunk_new_alerts=[]
        for raw_idx,ts,frame_bgr,samp_idx in chunk_frames:
            frame_norm,light_method=normalize_lighting(frame_bgr)
            last_light_method=light_method
            if light_method!="normal": last_original_frame=frame_bgr.copy()
            meta=detect_face_eyes(frame_norm)
            rng=_make_rng(source_name,samp_idx)
            # Module A
            if sim_a_mode: ra=sim_a(rng); a_src="simulation"
            else:
                o1=infer_a1(m1,meta["face"]); o2=infer_a2(m2,meta["eye_strip"])
                if o1 or o2:
                    fb=sim_a(rng); ra={**fb,**(o1 or {}),**(o2 or {})}
                    a_src="hybrid_model" if not(o1 and o2) else "model"
                else: ra=sim_a(rng); a_src="simulation_fallback"
            # Module B
            if sim_b_mode: rb=sim_b(rng,offroad_thr); b_src="simulation"
            else:
                rb=infer_b(bm,frame_bgr,offroad_thr)
                if rb is None: rb=sim_b(rng,offroad_thr); b_src="simulation_fallback"
                else: b_src="model"
            row_a={"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
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
            row_b={"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,
                   "input_type":"video","source_file_name":source_name,
                   "frame_index":samp_idx,"timestamp_seconds":ts,"inference_mode":b_src,**rb}
            rolling.push_a(row_a); rolling.push_b(row_b)
            all_rows_a.append(row_a); all_rows_b.append(row_b)
            if samp_idx%max(1,int(afps*5))==0: frame_map[samp_idx]=frame_bgr.copy()

        # ── chunk state ────────────────────────────────────────────────────────
        cs=compute_current_state(rolling.df_a(),rolling.df_b(),cfg_a)
        cs["ts"]=chunk_frames[-1][1]; state=cs["state"]; ts_last=chunk_frames[-1][1]
        fr=cs["frame_risk"]; op=cs["offroad_prob"]; conf=cs["fatigue_conf"]; zone=cs["zone"]

        # Detect blink in this chunk
        recent_a=rolling.df_a()
        blink_active=False
        if not recent_a.empty and "a2_prob_eye_closed" in recent_a.columns:
            last_few=recent_a["a2_prob_eye_closed"].tail(3).values
            blink_active=bool(np.any(last_few>cfg_a["t_eye"]) and np.any(last_few<cfg_a["t_eye"]))

        # ── alert firing ───────────────────────────────────────────────────────
        common_kw=dict(cs=cs,lighting=last_light_method)
        if state in ["high_fatigue","prolonged_eye_closure"]:
            a=alert_mgr.try_fire("drowsiness_alert","high",fr,ts_last,
                f"🚨 DROWSINESS @ {_fmt_ts(ts_last)} | Fatigue:{fr:.2f} | Conf:{conf:.2f}",
                conf,fr,op,zone,**common_kw)
            if a: chunk_new_alerts.append(a)
        if cs["perclos"]>=0.25:
            a=alert_mgr.try_fire("perclos_alert","high",cs["perclos"],ts_last,
                f"😴 HIGH PERCLOS @ {_fmt_ts(ts_last)} | {cs['perclos']:.1%} | Conf:{conf:.2f}",
                conf,fr,op,zone,**common_kw)
            if a: chunk_new_alerts.append(a)
        if state in ["offroad_glance","repeated_distraction"] and conf>=MIN_ALERT_CONF:
            sev="high" if get_risk_group(zone)=="HighRisk" else "medium"
            a=alert_mgr.try_fire("distraction_alert",sev,op,ts_last,
                f"👁 DISTRACTION @ {_fmt_ts(ts_last)} | Zone:{zone} | Off-road:{op:.2f}",
                conf,fr,op,zone,**common_kw)
            if a: chunk_new_alerts.append(a)
        if alert_mgr.esc_level==4:
            a=alert_mgr.try_fire("escalation_critical","critical",1.0,ts_last,
                f"🆘 CRITICAL @ {_fmt_ts(ts_last)} — IMMEDIATE ACTION REQUIRED",
                conf,fr,op,zone,**common_kw)
            if a: chunk_new_alerts.append(a)
        alert_mgr.mark_safe_frame(ts_last,state)

        # ── annotate display frame ─────────────────────────────────────────────
        disp_raw=chunk_frames[-1][2]
        disp_ann=annotate_frame(disp_raw,state,cs,alert_mgr.esc_level,bool(chunk_new_alerts),last_light_method,blink_active)
        dw=DISPLAY_WIDTH; dh=int(dw*disp_ann.shape[0]/max(1,disp_ann.shape[1]))
        disp_small=cv2.resize(disp_ann,(dw,dh))

        # ── ETA + progress stats ───────────────────────────────────────────────
        pct=min(global_raw_idx/max(total_vf,1),1.0)
        elapsed=time.time()-chunk_start_wall; chunks_done+=1
        secs_per_chunk=elapsed/max(chunks_done,1)
        chunks_total=max(1,total_vf//max(chunk_raw,1))
        remaining_chunks=max(0,chunks_total-chunks_done)
        eta_s=remaining_chunks*secs_per_chunk
        eta_str=_fmt_ts(eta_s) if eta_s>0 else "—"
        alerts_hr=(alert_mgr.total_alerts/(elapsed/3600)) if elapsed>0 else 0
        prog_bar.progress(pct,f"Processing {_fmt_ts(ts_last)} / {_fmt_ts(video_meta['duration_seconds'])}")

        with prog_stats.container():
            pa,pb,pc=st.columns(3)
            pa.metric("ETA",eta_str)
            pb.metric("Alerts/hr",f"{alerts_hr:.0f}",delta="↑" if alerts_hr>10 else None,delta_color="inverse")
            pc.metric("Conservative mode","ON 🛡️" if alert_mgr.conservative_mode else "OFF")

        # ── live metric grid ───────────────────────────────────────────────────
        fat_col="🔴" if fr>=cfg_a["t_alert"] else ("🟠" if fr>=cfg_a["t_caution"] else "🟢")
        esc_col=ESC_HEX[min(alert_mgr.esc_level,4)]
        with m_fat.container():
            st.markdown(f"<div style='background:{'#1a1a1a'};padding:8px;border-radius:6px;border-left:4px solid {'#ef4444' if fr>=cfg_a['t_alert'] else '#f59e0b' if fr>=cfg_a['t_caution'] else '#22c55e'};'>"
                        f"<div style='font-size:11px;color:#aaa;'>Fatigue Risk</div>"
                        f"<div style='font-size:22px;font-weight:bold;color:{'#ef4444' if fr>=cfg_a['t_alert'] else '#f59e0b' if fr>=cfg_a['t_caution'] else '#22c55e'};'>{fr:.2f} {fat_col}</div></div>",unsafe_allow_html=True)
        with m_esc.container():
            st.markdown(f"<div style='background:{esc_col};padding:8px;border-radius:6px;'>"
                        f"<div style='font-size:11px;color:rgba(255,255,255,0.7);'>Escalation</div>"
                        f"<div style='font-size:16px;font-weight:bold;color:white;'>{ESC_LABELS[min(alert_mgr.esc_level,4)]}</div></div>",unsafe_allow_html=True)
        with m_off.container():
            c_op="#ef4444" if op>=offroad_thr else "#22c55e"
            st.markdown(f"<div style='background:#1a1a1a;padding:8px;border-radius:6px;border-left:4px solid {c_op};'>"
                        f"<div style='font-size:11px;color:#aaa;'>Off-road Prob</div>"
                        f"<div style='font-size:22px;font-weight:bold;color:{c_op};'>{op:.2f}</div></div>",unsafe_allow_html=True)
        with m_zone.container():
            risk_g=get_risk_group(zone); zc={"Safe":"#22c55e","LowRisk":"#f59e0b","HighRisk":"#ef4444"}.get(risk_g,"#888")
            st.markdown(f"<div style='background:#1a1a1a;padding:8px;border-radius:6px;border-left:4px solid {zc};'>"
                        f"<div style='font-size:11px;color:#aaa;'>Gaze Zone</div>"
                        f"<div style='font-size:16px;font-weight:bold;color:{zc};'>{zone}</div></div>",unsafe_allow_html=True)
        with m_perc.container():
            pc_c="#ef4444" if cs["perclos"]>=0.25 else "#22c55e"
            st.markdown(f"<div style='background:#1a1a1a;padding:8px;border-radius:6px;border-left:4px solid {pc_c};'>"
                        f"<div style='font-size:11px;color:#aaa;'>PERCLOS</div>"
                        f"<div style='font-size:22px;font-weight:bold;color:{pc_c};'>{cs['perclos']:.1%}</div></div>",unsafe_allow_html=True)
        with m_ear.container():
            ear_val=f"{cs['ear']:.3f}" if cs.get("ear") is not None else "N/A"
            ear_c="#ef4444" if cs.get("ear") is not None and cs["ear"]<EAR_CLOSED_THRESHOLD else "#22c55e"
            st.markdown(f"<div style='background:#1a1a1a;padding:8px;border-radius:6px;border-left:4px solid {ear_c};'>"
                        f"<div style='font-size:11px;color:#aaa;'>EAR {'👁' if not blink_active else '😑'}</div>"
                        f"<div style='font-size:22px;font-weight:bold;color:{ear_c};'>{ear_val}</div></div>",unsafe_allow_html=True)

        # ── annotated frame ────────────────────────────────────────────────────
        with frame_ph.container():
            st.image(cv2.cvtColor(disp_small,cv2.COLOR_BGR2RGB),
                     caption=f"{state.replace('_',' ').title()} | {_fmt_ts(ts_last)} | Alerts: {alert_mgr.total_alerts}",
                     use_container_width=True)

        # ── before/after lighting comparison ──────────────────────────────────
        if last_light_method not in ["normal","none","error"] and last_original_frame is not None:
            comp=make_lighting_comparison(last_original_frame,disp_raw,last_light_method)
            if comp is not None:
                with light_ph.container():
                    st.image(cv2.cvtColor(comp,cv2.COLOR_BGR2RGB),
                             caption=f"Lighting: {LIGHT_BADGE.get(last_light_method,last_light_method)}",
                             use_container_width=True)

        # ── pulsing alert banner ───────────────────────────────────────────────
        if chunk_new_alerts:
            top=chunk_new_alerts[0]
            html_banner=_pulsing_alert_html(top["message"],top["severity"],alert_mgr.esc_level)
            with alert_ph.container():
                st.markdown(html_banner,unsafe_allow_html=True)
                if top.get("recommendation"):
                    st.info(f"💡 **Recommendation:** {top['recommendation']}")
        elif state=="normal_driving" and chunk_idx%5==0:
            alert_ph.success(f"✅ Normal driving | {_fmt_ts(ts_last)}")

        # ── alert log ─────────────────────────────────────────────────────────
        if alert_mgr.alert_history:
            adf=alert_mgr.alerts_df()
            adf["time"]=adf["alert_timestamp_seconds"].apply(_fmt_ts)
            show_cols=[c for c in ["time","severity","alert_type","message","confidence","recommendation"] if c in adf.columns]
            with log_ph.container():
                st.dataframe(adf[show_cols].tail(6).iloc[::-1],hide_index=True,use_container_width=True)

        # ── rolling chart ──────────────────────────────────────────────────────
        recent_n=min(len(all_rows_a),int(60*afps))
        if recent_n>=3:
            rec=all_rows_a[-recent_n:]
            cdf=pd.DataFrame({"Time":[r["timestamp_seconds"] for r in rec],
                              "Fatigue":[float(r.get("a1_prob_drowsy") or 0) for r in rec],
                              "Eye Closed":[float(r.get("a2_prob_eye_closed") or 0) for r in rec]})
            with chart_ph.container(): st.line_chart(cdf,x="Time",y=["Fatigue","Eye Closed"])

        # ── incremental SF writes ──────────────────────────────────────────────
        n_chunk=len(chunk_frames)
        if all_rows_a: write_to_snowflake(pd.DataFrame(all_rows_a[-n_chunk:]),"MODULE_A_FRAME_PREDICTIONS")
        if all_rows_b: write_to_snowflake(pd.DataFrame(all_rows_b[-n_chunk:]),"MODULE_B_FRAME_PREDICTIONS")
        if chunk_new_alerts:
            al_rows=[{**a,"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,"created_at_client":utc_now_iso()}
                     for a in chunk_new_alerts]
            write_to_snowflake(pd.DataFrame(al_rows),"REALTIME_ALERTS")
        chunk_idx+=1

    cap.release()
    prog_bar.progress(1.0,f"✅ Complete — {alert_mgr.total_alerts} alerts fired")
    time.sleep(0.8); prog_bar.empty()
    return all_rows_a,all_rows_b,alert_mgr,frame_map

# =============================================================================
# POST-INFERENCE ANALYTICS
# =============================================================================
def render_post_analytics(df_a,df_b,cfg_a,afps,offroad_thr,sw_b,
                           sid,driver_id,trip_id,uploaded_name,ss_ts,se_ts,
                           alert_mgr:AlertManager,frame_map:dict):
    st.markdown("---"); st.markdown("## 📊 Full Session Analytics")
    with st.spinner("Running final temporal analysis…"):
        df_a,a_sum,a_events=temporal_a(df_a,cfg_a)
        df_b,b_sum,b_events=temporal_b(df_b,afps,sw_b,offroad_thr)

    tl_df=build_timeline(df_a,df_b)
    tl_df["session_id"]=sid; tl_df["driver_id"]=driver_id; tl_df["trip_id"]=trip_id
    tl_rows=summarize_timeline(tl_df,afps)
    scorecard=build_driver_scorecard(driver_id,trip_id,sid,a_sum,b_sum)
    summary_text=_explain_unified(a_sum,b_sum,scorecard)
    uf_df=build_unified_frame_predictions(df_a,df_b,sid,driver_id,trip_id,uploaded_name)
    zt_df=build_zone_transitions(df_b,sid,driver_id,trip_id)
    qsum_df=build_quality_summary(df_a,df_b,sid,driver_id,trip_id,uploaded_name)
    total_alrts=alert_mgr.total_alerts
    esc_level=min(4,alert_mgr.esc_level)
    cds=scorecard["combined_driver_safety_score"]
    esc_final=min(3,max(esc_level,len([e for e in a_events+b_events if e.get("alert_sent")])))
    unified_sum={"session_id":sid,"driver_id":driver_id,"trip_id":trip_id,"session_start_ts":ss_ts,
                 "session_end_ts":se_ts,"duration_seconds":float(a_sum.get("total_duration",0)),
                 "source_file_name":uploaded_name,"model_version_a1":MODEL_VER_A1,
                 "model_version_a2":MODEL_VER_A2,"model_version_b":MODEL_VER_B,
                 "app_version":APP_VERSION,"fatigue_score":scorecard["fatigue_risk_score"],
                 "distraction_score":scorecard["distraction_risk_score"],
                 "combined_driver_safety_score":cds,"driver_rating":scorecard["driver_rating"],
                 "summary_text":summary_text,"escalation_level":esc_final,
                 "mean_model_confidence":scorecard["mean_model_confidence"],
                 "total_realtime_alerts":total_alrts,"trip_start_ts":ss_ts,"trip_end_ts":se_ts,
                 "created_at_client":utc_now_iso()}

    # Banner
    r_color={"A":"#22c55e","B":"#22c55e","C":"#f59e0b","D":"#ef4444","E":"#7f1d1d"}.get(scorecard["driver_rating"],"#888")
    st.markdown(f"<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;'>"
                f"<div style='background:{ESC_HEX[esc_level]};padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>Escalation: {ESC_LABELS[esc_level]}</div>"
                f"<div style='background:#1e293b;padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>Rating: <span style='color:{r_color};font-size:22px;'>{scorecard['driver_rating']}</span></div>"
                f"<div style='background:#1e293b;padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>Safety Score: {cds:.2f}</div>"
                f"<div style='background:#1e293b;padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>Alerts: {total_alrts}</div>"
                f"<div style='background:#1e293b;padding:8px 16px;border-radius:8px;color:white;font-weight:bold;'>Recoveries: {len(alert_mgr.recovery_events)}</div>"
                f"</div>",unsafe_allow_html=True)
    st.caption(summary_text)

    tabs=st.tabs(["Overview","Fatigue","Attentiveness","Driver Timeline","Real-time Alerts",
                  "Recovery Events","Frame Explorer","Events","Scorecard","Data Export"])

    # Tab 0: Overview
    with tabs[0]:
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Fatigue Score",   f"{scorecard['fatigue_risk_score']:.2f}")
        c2.metric("Distraction",     f"{scorecard['distraction_risk_score']:.2f}")
        c3.metric("Safety Score",    f"{cds:.2f}")
        c4.metric("Driver Rating",   scorecard["driver_rating"])
        c5.metric("RT Alerts",       total_alrts)
        c6,c7,c8,c9=st.columns(4)
        c6.metric("PERCLOS",         f"{a_sum['perclos']:.1%}")
        c7.metric("Blinks/min",      f"{a_sum['blink_freq_per_min']:.1f}")
        c8.metric("Off-road Ratio",  f"{b_sum['or']:.1%}")
        c9.metric("Safe Forward",    f"{b_sum['sfr']:.1%}")
        if frame_map:
            sample=sorted(frame_map.keys())[:6]
            gcols=st.columns(min(3,len(sample)))
            for i,k in enumerate(sample[:6]):
                with gcols[i%3]:
                    st.image(cv2.cvtColor(frame_map[k],cv2.COLOR_BGR2RGB),caption=f"Frame {k}",use_container_width=True)
        st.caption("ℹ️ Lower safety score = safer driver. Rating A–E.")

    # Tab 1: Fatigue
    with tabs[1]:
        st.subheader("Fatigue Signal Over Time")
        fat_cols=[c for c in ["timestamp_seconds","frame_risk","a1_prob_drowsy_sm","a2_prob_eye_closed_sm","perclos_30s","blink_rate_per_min"] if c in df_a.columns]
        if len(fat_cols)>1:
            fdf2=df_a[fat_cols].rename(columns={"timestamp_seconds":"Time (s)","frame_risk":"Fatigue Risk",
                "a1_prob_drowsy_sm":"Drowsy (sm)","a2_prob_eye_closed_sm":"Eye Closed (sm)","perclos_30s":"PERCLOS","blink_rate_per_min":"Blink/min"})
            st.line_chart(fdf2,x="Time (s)")
        if alert_mgr.alert_history:
            fa=[a for a in alert_mgr.alert_history if "drowsiness" in a["alert_type"] or "perclos" in a["alert_type"]]
            if fa:
                fadf=pd.DataFrame(fa); fadf["time"]=fadf["alert_timestamp_seconds"].apply(_fmt_ts)
                st.markdown("**⚠️ Fatigue Alerts:**")
                st.dataframe(fadf[[c for c in ["time","severity","message","confidence","recommendation","consensus_signals"] if c in fadf.columns]].iloc[::-1],hide_index=True,use_container_width=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Avg EAR",       f"{a_sum['avg_ear']:.3f}" if a_sum.get("avg_ear") else "N/A")
        c2.metric("Blink Count",   a_sum["blink_count"])
        c3.metric("Avg Blink Dur", f"{a_sum['avg_blink_duration']:.2f}s")
        c4.metric("Uncertain",     f"{a_sum['uncertain_ratio']:.1%}")

    # Tab 2: Attentiveness
    with tabs[2]:
        st.subheader("Attentiveness Signal Over Time")
        att_cols=[c for c in ["timestamp_seconds","offroad_prob_sm","confidence_sm","uncertainty_score_sm"] if c in df_b.columns]
        if len(att_cols)>1:
            adf2=df_b[att_cols].rename(columns={"timestamp_seconds":"Time (s)","offroad_prob_sm":"Off-road (sm)","confidence_sm":"Confidence (sm)","uncertainty_score_sm":"Uncertainty (sm)"})
            st.line_chart(adf2,x="Time (s)")
        st.subheader("Gaze Zone Distribution")
        zd=df_b["zone_pred"].value_counts().reset_index(); zd.columns=["Zone","Count"]
        st.bar_chart(zd,x="Zone",y="Count")
        if alert_mgr.alert_history:
            da=[a for a in alert_mgr.alert_history if "distraction" in a["alert_type"]]
            if da:
                dadf=pd.DataFrame(da); dadf["time"]=dadf["alert_timestamp_seconds"].apply(_fmt_ts)
                st.markdown("**👁 Distraction Alerts:**")
                st.dataframe(dadf[[c for c in ["time","severity","zone_at_alert","message","confidence","recommendation"] if c in dadf.columns]].iloc[::-1],hide_index=True,use_container_width=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Off-road Ratio",f"{b_sum['or']:.1%}"); c2.metric("Max Streak",f"{b_sum['mos']:.1f}s")
        c3.metric("Mirror Freq/min",f"{b_sum['mfpm']:.1f}"); c4.metric("HighRisk Ratio",f"{b_sum['hr']:.1%}")

    # Tab 3: Driver Timeline
    with tabs[3]:
        st.subheader("Driver State Timeline")
        if tl_rows:
            total_dur=sum(r["duration_seconds"] for r in tl_rows)
            st.markdown(render_timeline_html(tl_rows,total_dur),unsafe_allow_html=True)
            tshow=pd.DataFrame(tl_rows).copy()
            tshow["start"]=tshow["start_ts"].apply(_fmt_ts); tshow["end"]=tshow["end_ts"].apply(_fmt_ts)
            tshow["duration_min"]=(tshow["duration_seconds"]/60).round(2)
            st.dataframe(tshow[["timeline_state","start","end","duration_seconds","duration_min"]],hide_index=True,use_container_width=True)
            sa=tshow.groupby("timeline_state")["duration_seconds"].sum().reset_index()
            sa.columns=["State","Duration (s)"]; st.bar_chart(sa,x="State",y="Duration (s)")
        else: st.caption("No timeline segments.")

    # Tab 4: Real-time Alerts
    with tabs[4]:
        st.subheader(f"Real-time Alert History ({total_alrts} alerts)")
        if alert_mgr.alert_history:
            aldf=alert_mgr.alerts_df(); aldf["time"]=aldf["alert_timestamp_seconds"].apply(_fmt_ts)
            esc_chart=aldf[["alert_timestamp_seconds","escalation_level"]].copy()
            esc_chart.columns=["Time (s)","Escalation Level"]
            st.markdown("**Escalation Progression:**"); st.line_chart(esc_chart,x="Time (s)",y="Escalation Level")
            tc=aldf["alert_type"].value_counts().reset_index(); tc.columns=["Alert Type","Count"]
            st.bar_chart(tc,x="Alert Type",y="Count")
            sc=aldf["severity"].value_counts().reset_index(); sc.columns=["Severity","Count"]
            st.bar_chart(sc,x="Severity",y="Count")
            st.markdown("**Full Alert Log:**")
            dcols=[c for c in ["time","severity","alert_type","message","confidence","recommendation",
                                "consensus_signals","lighting_at_alert","escalation_level","zone_at_alert"] if c in aldf.columns]
            st.dataframe(aldf[dcols].iloc[::-1],hide_index=True,use_container_width=True,height=380)
        else: st.success("No alerts fired — excellent session!")

    # Tab 5: Recovery Events
    with tabs[5]:
        st.subheader(f"Driver Recovery Events ({len(alert_mgr.recovery_events)})")
        st.info("A recovery event is logged whenever escalation drops after sustained safe driving. It shows how quickly the driver self-corrected.")
        rec_df=alert_mgr.recovery_df()
        if not rec_df.empty:
            rec_df["time"]=rec_df["recovery_timestamp_seconds"].apply(_fmt_ts)
            st.dataframe(rec_df[[c for c in ["time","previous_state","peak_escalation_before","alert_count_before"] if c in rec_df.columns]],hide_index=True,use_container_width=True)
            c1,c2=st.columns(2)
            c1.metric("Total Recoveries",len(rec_df))
            c2.metric("Avg Peak Before Recovery",f"{rec_df['peak_escalation_before'].mean():.1f}" if "peak_escalation_before" in rec_df.columns else "N/A")
        else: st.caption("No recovery events — driver either maintained safety or never escalated.")

    # Tab 6: Frame Explorer
    with tabs[6]: render_frame_explorer(frame_map,uf_df)

    # Tab 7: Events
    with tabs[7]:
        st.subheader("Detected Events")
        ev_rows=[]
        for ev in a_events:
            ev_rows.append({"module":"A","event_type":ev["type"],"start":_fmt_ts(ev["start"]),"end":_fmt_ts(ev["end"]),
                            "duration_s":round(ev["dur"],2),"severity":ev["severity"],"severity_score":round(ev["severity_score"],3),
                            "confidence":round(ev["confidence"],3),"alert_sent":ev["alert_sent"],"risk_group":ev.get("risk_group","")})
        for ev in b_events:
            ev_rows.append({"module":"B","event_type":ev["type"],"start":_fmt_ts(ev["start"]),"end":_fmt_ts(ev["end"]),
                            "duration_s":round(ev["dur"],2),"severity":ev["severity"],"severity_score":round(ev["severity_score"],3),
                            "confidence":round(ev["confidence"],3),"alert_sent":ev["alert_sent"],
                            "risk_group":ev.get("risk_group",""),"dominant_zone":ev.get("dominant_zone","")})
        if ev_rows:
            evdf=pd.DataFrame(ev_rows)
            st.dataframe(evdf,hide_index=True,use_container_width=True,height=360)
            c1,c2,c3=st.columns(3)
            c1.metric("Total Events",len(evdf)); c2.metric("Actionable Alerts",int(evdf["alert_sent"].astype(int).sum()))
            c3.metric("Eye Closures",int((evdf["event_type"]=="prolonged_eye_closure").sum()))
            sb=evdf["severity"].value_counts().reset_index(); sb.columns=["Severity","Count"]
            st.bar_chart(sb,x="Severity",y="Count")

            # Alert precision: real-time fired vs post-hoc confirmed
            confirmed=len([e for e in a_events+b_events if e.get("alert_sent") and e.get("event_confirmation_status")=="auto_confirmed"])
            rt_total=alert_mgr.total_alerts
            if rt_total>0:
                precision=min(1.0,confirmed/rt_total)
                st.markdown(f"**Alert Precision Retrospective:** {confirmed} post-hoc confirmed events vs {rt_total} real-time alerts fired → **Precision: {precision:.1%}**")
                st.caption("Higher precision means the real-time system closely matches the full temporal analysis. Low precision in a session may indicate difficult lighting or poor video quality.")
        else: st.caption("No events detected.")

    # Tab 8: Scorecard
    with tabs[8]:
        sc_d={k:(f"{v:.3f}" if isinstance(v,float) else v) for k,v in scorecard.items()}
        st.dataframe(pd.DataFrame([sc_d]),hide_index=True,use_container_width=True)
        bd={"Component":["Fatigue (55%)","Distraction (45%)","Uncertainty Penalty"],
            "Score":[scorecard["fatigue_risk_score"],scorecard["distraction_risk_score"],
                     round(0.08*max(0.0,0.6-scorecard["mean_model_confidence"]),4)]}
        st.bar_chart(pd.DataFrame(bd),x="Component",y="Score")
        st.info("A (≤0.20): Excellent | B (≤0.40): Good | C (≤0.60): Caution | D (≤0.80): Poor | E (>0.80): Critical")
        st.download_button("⬇ Scorecard CSV",pd.DataFrame([scorecard]).to_csv(index=False),f"scorecard_{sid}.csv","text/csv")
        st.dataframe(qsum_df,hide_index=True,use_container_width=True)

        # Lighting quality impact
        st.subheader("Lighting Quality Impact Report")
        if not df_a.empty and "lighting_method" in df_a.columns:
            lm=df_a.groupby("lighting_method").agg(frame_count=("frame_id","count"),
               avg_quality=("quality_score","mean") if "quality_score" in df_a.columns else ("frame_id","count"),
               avg_confidence=("a1_confidence","mean") if "a1_confidence" in df_a.columns else ("frame_id","count")).reset_index()
            lm.columns=[c.replace("_"," ").title() for c in lm.columns]
            st.dataframe(lm,hide_index=True,use_container_width=True)

    # Tab 9: Data Export
    with tabs[9]:
        st.subheader("Snowflake Export Summary")
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Module A Frames",len(df_a)); c2.metric("Module B Frames",len(df_b))
        c3.metric("Unified Frames",len(uf_df)); c4.metric("RT Alerts",total_alrts)
        c5.metric("Recovery Events",len(alert_mgr.recovery_events))
        comb=pd.merge(df_a,df_b,on="timestamp_seconds",how="outer",suffixes=("_A","_B"))
        st.download_button("⬇ Combined Frame CSV",comb.to_csv(index=False),f"combined_{sid}.csv","text/csv")
        st.download_button("⬇ Unified Predictions CSV",uf_df.to_csv(index=False),f"unified_{sid}.csv","text/csv")
        if not alert_mgr.alerts_df().empty:
            st.download_button("⬇ Alerts CSV",alert_mgr.alerts_df().to_csv(index=False),f"alerts_{sid}.csv","text/csv")
        if tl_rows:
            st.download_button("⬇ Timeline CSV",pd.DataFrame(tl_rows).to_csv(index=False),f"timeline_{sid}.csv","text/csv")
        if not alert_mgr.recovery_df().empty:
            st.download_button("⬇ Recovery Events CSV",alert_mgr.recovery_df().to_csv(index=False),f"recovery_{sid}.csv","text/csv")

    # Final SF write
    with st.spinner("Writing all data to Snowflake…"):
        _write_all_to_sf(df_a,df_b,a_events,b_events,a_sum,b_sum,scorecard,uf_df,zt_df,qsum_df,tl_rows,
                         unified_sum,alert_mgr,sid,driver_id,trip_id,ss_ts,se_ts,uploaded_name)
    st.success("✅ All data written to Snowflake successfully.")

# =============================================================================
# HOME PAGE
# =============================================================================
def home_page():
    st.title(":material/directions_car: Driver Safety Analytics Platform  v5")
    st.markdown("**Real-time streaming · Live HUD · Smart alerts · Snowflake-native analytics**")
    c1,c2,c3=st.columns(3)
    with c1:
        st.metric("A1 Model","Loaded ✅" if load_a1() else "Simulation mode")
        st.metric("A2 Model","Loaded ✅" if load_a2() else "Simulation mode")
    with c2:
        st.metric("Module B","Loaded ✅" if load_b() else "Simulation mode")
        st.metric("MediaPipe","Available ✅" if MP_AVAILABLE else "Fallback (Haar/crop)")
    with c3:
        for pkg,ok in [("PyTorch",TORCH_AVAILABLE),("OpenCV",CV2_AVAILABLE),("Pillow",PIL_AVAILABLE)]:
            st.write(f"{'✅' if ok else '❌'} {pkg}")
        st.caption(f"Snowflake: {'Connected ✅' if get_session() else 'Not connected ⚠️'}")
    st.divider()
    with st.container(border=True):
        st.subheader("What's New in v5.0.0")
        st.markdown("""
**Real-time UI**
- Structured dashboard: annotated frame (left) + colour-coded 2×3 metric grid (right) always visible during processing
- Progress bar now shows ETA, running alerts/hour, and conservative-mode status
- Lighting badge on every annotated frame — ☀️ Glare, 🌙 Night, 🌫️ Contrast Boost
- Side-by-side before/after lighting correction panel when a correction activates
- Pulsing animated CSS alert banner coloured by severity — pulses at 1.2 s rhythm
- Blink event indicator (👁 / 😑) on HUD and metric grid

**Alert System**
- **Sustained signal gate**: signal must be elevated for ≥ 3 s before any alert fires — eliminates single-frame spikes
- **Consensus check**: ≥ 2 of 3 signals (A1 drowsy, A2 eye-closed, EAR) must agree before drowsiness alert fires
- **Dynamic confidence floor**: raised to 0.65 when lighting correction is active — avoids false positives under glare
- **Mirror cadence suppression**: regular brief mirror checks are suppressed as safe driving behaviour
- **Isolated yawn suppression**: yawns without accompanying eye closure or high drowsy probability do not trigger alerts
- **Alert fatigue guard**: >15 alerts in 10 min → conservative mode (Critical-level alerts only), announced to user
- **Graded output**: every alert now includes a plain-language recommendation ("Take a rest break within 15 min")
- **Consensus signals logged**: which signals agreed is stored in REALTIME_ALERTS for post-hoc audit
- **Recovery events**: logged to new DRIVER_RECOVERY_EVENTS Snowflake table whenever driver de-escalates

**Analytics**
- New Recovery Events tab: shows how many times driver self-corrected and their peak escalation before recovery
- Alert Precision Retrospective: compares real-time alerts vs post-hoc confirmed events
- Lighting Quality Impact Report: per-lighting-method frame count, avg quality score, avg confidence
- Dedicated consensus_signals and lighting_at_alert columns in all alert exports
        """)
    st.divider()
    st.subheader("📋 Snowflake Schema Changes Required for v5")
    st.code("""
-- From v4 (run if not already done):
CREATE TABLE IF NOT EXISTS DEMO_DB.PUBLIC.REALTIME_ALERTS (
    ALERT_ID VARCHAR(64), SESSION_ID VARCHAR(64), DRIVER_ID VARCHAR(64),
    TRIP_ID VARCHAR(64), ALERT_TIMESTAMP_SECONDS FLOAT,
    ALERT_WALL_TIME VARCHAR(64), ALERT_TYPE VARCHAR(64),
    SEVERITY VARCHAR(16), SEVERITY_SCORE FLOAT, ESCALATION_LEVEL INT,
    FATIGUE_SCORE FLOAT, OFFROAD_PROB FLOAT, ZONE_AT_ALERT VARCHAR(32),
    CONFIDENCE FLOAT, MESSAGE VARCHAR(512), RECOMMENDATION VARCHAR(512),
    CONSENSUS_SIGNALS VARCHAR(128), LIGHTING_AT_ALERT VARCHAR(32),
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
ALTER TABLE DEMO_DB.PUBLIC.MODULE_A_FRAME_PREDICTIONS
    ADD COLUMN IF NOT EXISTS LIGHTING_METHOD VARCHAR(32);
ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
    ADD COLUMN IF NOT EXISTS MEAN_MODEL_CONFIDENCE FLOAT;
ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
    ADD COLUMN IF NOT EXISTS TOTAL_REALTIME_ALERTS INT;

-- New in v5:
CREATE TABLE IF NOT EXISTS DEMO_DB.PUBLIC.DRIVER_RECOVERY_EVENTS (
    RECOVERY_ID VARCHAR(64), SESSION_ID VARCHAR(64), DRIVER_ID VARCHAR(64),
    TRIP_ID VARCHAR(64), RECOVERY_TIMESTAMP_SECONDS FLOAT,
    PEAK_ESCALATION_BEFORE INT, RECOVERY_DURATION_SECONDS FLOAT,
    PREVIOUS_STATE VARCHAR(64), ALERT_COUNT_BEFORE INT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Snowflake Dynamic Table (auto-refreshing leaderboard):
CREATE OR REPLACE DYNAMIC TABLE DEMO_DB.PUBLIC.DRIVER_RISK_LEADERBOARD
    TARGET_LAG = '1 hour'
    WAREHOUSE = COMPUTE_WH
AS
SELECT DRIVER_ID,
       AVG(COMBINED_DRIVER_SAFETY_SCORE)  AS AVG_SAFETY_SCORE,
       COUNT(DISTINCT SESSION_ID)          AS TOTAL_SESSIONS,
       MAX(ESCALATION_LEVEL)               AS PEAK_ESCALATION,
       SUM(TOTAL_REALTIME_ALERTS)          AS TOTAL_ALERTS
FROM DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
GROUP BY DRIVER_ID;

-- Snowflake View (weekly trend):
CREATE OR REPLACE VIEW DEMO_DB.PUBLIC.DRIVER_WEEKLY_TREND AS
SELECT DRIVER_ID,
       DATE_TRUNC('week', TO_TIMESTAMP(SESSION_START_TS)) AS WEEK,
       AVG(COMBINED_DRIVER_SAFETY_SCORE) AS AVG_SCORE,
       SUM(TOTAL_REALTIME_ALERTS)        AS WEEKLY_ALERTS,
       COUNT(DISTINCT SESSION_ID)        AS TRIPS
FROM DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
GROUP BY 1, 2;
""", language="sql")

# =============================================================================
# REAL-TIME ANALYSIS PAGE
# =============================================================================
def unified_page():
    st.title(":material/security: Unified Driver Safety — Real-time Analysis")
    with st.sidebar:
        st.subheader("🪪 Driver & Trip")
        driver_id = st.text_input("Driver ID","DRV_001")
        trip_id   = st.text_input("Trip ID","TRIP_001")
        trip_start= st.text_input("Trip Start",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        trip_end  = st.text_input("Trip End","")
        st.subheader("⚙️ Sampling")
        afps = st.number_input("Target FPS",1,30,DEFAULT_A["fps"])
        st.subheader("🔥 Fatigue Weights")
        wd=st.slider("Drowsy",0.0,1.0,DEFAULT_A["w_drowsy"],0.05)
        we=st.slider("Eye Closed",0.0,1.0,DEFAULT_A["w_eye_closed"],0.05)
        wn=st.slider("Nod",0.0,1.0,DEFAULT_A["w_nod"],0.05)
        wy=st.slider("Yawn",0.0,1.0,DEFAULT_A["w_yawn"],0.05)
        wp=st.slider("PERCLOS",0.0,1.0,DEFAULT_A["w_perclos"],0.05)
        wb=st.slider("Blink",0.0,1.0,DEFAULT_A["w_blink"],0.05)
        st.subheader("🎯 Thresholds")
        te =st.slider("Eye closed thr",0.0,1.0,DEFAULT_A["t_eye"],0.05)
        tc =st.slider("Caution thr",0.0,1.0,DEFAULT_A["t_caution"],0.05)
        ta =st.slider("Alert thr",0.0,1.0,DEFAULT_A["t_alert"],0.05)
        pcf=st.number_input("Min closure frames",2,30,DEFAULT_A["closure_frames"])
        ear_c=st.slider("EAR closed",0.10,0.40,DEFAULT_A["ear_closed"],0.01)
        st.subheader("👁 Attentiveness")
        sw_b=st.number_input("Smoothing window",1,10,3)
        bort=st.slider("Off-road threshold",0.0,1.0,OFFROAD_THRESHOLD,0.05)
        st.subheader("🚨 Alert Settings (v5)")
        st.caption(f"Cooldown: {ALERT_COOLDOWN_S}s | Min sustained: {SUSTAINED_ALERT_S}s")
        st.caption(f"Chunk: {CHUNK_SECONDS}s | Fatigue guard: {ALERT_FATIGUE_LIMIT} alerts / {ALERT_FATIGUE_WINDOW//60} min")

    cfg_a=dict(w_drowsy=wd,w_eye_closed=we,w_nod=wn,w_yawn=wy,w_perclos=wp,w_blink=wb,
               t_eye=te,t_caution=tc,t_alert=ta,closure_frames=pcf,fps=afps,
               win_s=DEFAULT_A["win_s"],win_m=DEFAULT_A["win_m"],win_l=DEFAULT_A["win_l"],ear_closed=ear_c)
    sim_a_mode=load_a1() is None or load_a2() is None
    sim_b_mode=load_b() is None
    if sim_a_mode or sim_b_mode:
        st.info(":material/science: One or more models not loaded — deterministic simulation will be used.")

    uploaded=st.file_uploader("📂 Upload driver video",type=["mp4","avi","mov","mkv"])
    if not uploaded: st.caption("Upload a video file to begin."); return
    ext=uploaded.name.rsplit(".",1)[-1].lower()
    sid=str(uuid.uuid4())[:12]
    ss_ts=trip_start if trip_start else utc_now_iso()
    se_ts=trip_end if trip_end else ""
    if not CV2_AVAILABLE: st.error("OpenCV not available."); return

    st.subheader("📹 Uploaded Video")
    st.video(uploaded.getvalue())

    if st.button("🚀 Start Real-time Analysis",type="primary",use_container_width=True):
        video_bytes=uploaded.getvalue()
        with st.spinner("Preparing video…"):
            video_path,video_meta=open_video(video_bytes,ext)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Original FPS",f"{video_meta['video_fps']:.2f}")
        c2.metric("Total Frames",int(video_meta["total_frames"]))
        c3.metric("Duration",f"{video_meta['duration_seconds']:.1f}s")
        c4.metric("Resolution",video_meta["resolution"])
        si=max(1,int(video_meta["video_fps"]/max(1,afps)))
        c5.metric("Sample Interval",f"every {si} frame(s)")
        try:
            all_rows_a,all_rows_b,alert_mgr,frame_map=run_streaming_inference(
                video_path,video_meta,cfg_a,afps,sw_b,bort,driver_id,trip_id,uploaded.name,sim_a_mode,sim_b_mode,sid)
        finally:
            try: os.unlink(video_path)
            except Exception: pass
        if not all_rows_a: st.error("No frames processed."); return
        df_a=pd.DataFrame(all_rows_a); df_b=pd.DataFrame(all_rows_b)
        df_a["session_id"]=sid; df_b["session_id"]=sid
        render_post_analytics(df_a,df_b,cfg_a,afps,bort,sw_b,sid,driver_id,trip_id,
                               uploaded.name,ss_ts,se_ts,alert_mgr,frame_map)

# =============================================================================
# HISTORICAL ANALYTICS PAGE  (Snowflake-native showcase)
# =============================================================================
def analytics_page():
    st.title(":material/analytics: Historical Analytics")
    tabs=st.tabs(["Fleet Overview","Driver Trends","Sessions","Scorecards","Driver Timeline",
                  "Real-time Alerts","Recovery Events","Frame Risk","Zone Transitions",
                  "Lighting Report","Review Queue","Snowflake Advanced"])

    # Tab 0: Fleet Overview
    with tabs[0]:
        st.subheader("Fleet-Level Risk Overview")
        st.info("This tab queries the DRIVER_RISK_LEADERBOARD Dynamic Table which auto-refreshes every hour inside Snowflake — no ETL pipeline needed.")
        lb=query_sf(f"SELECT * FROM {DATABASE}.{SCHEMA}.DRIVER_RISK_LEADERBOARD ORDER BY AVG_SAFETY_SCORE DESC LIMIT 50")
        if lb.empty:
            lb=fetch_recent("UNIFIED_DRIVER_SESSION_SUMMARY",200)
            if not lb.empty and {"DRIVER_ID","COMBINED_DRIVER_SAFETY_SCORE"}.issubset(lb.columns):
                lb=lb.groupby("DRIVER_ID",as_index=False).agg(
                    AVG_SAFETY_SCORE=("COMBINED_DRIVER_SAFETY_SCORE","mean"),
                    TOTAL_SESSIONS=("SESSION_ID","count"),
                    TOTAL_ALERTS=("TOTAL_REALTIME_ALERTS","sum")).rename(columns=str)
        if not lb.empty:
            st.dataframe(lb,hide_index=True,use_container_width=True)
            if "DRIVER_ID" in lb.columns and "AVG_SAFETY_SCORE" in lb.columns:
                st.bar_chart(lb[["DRIVER_ID","AVG_SAFETY_SCORE"]],x="DRIVER_ID",y="AVG_SAFETY_SCORE")
        else: st.caption("No fleet data yet. Run a few sessions first.")

        # Time-of-day heatmap proxy
        ra=fetch_recent("REALTIME_ALERTS",2000)
        if not ra.empty and "ALERT_WALL_TIME" in ra.columns:
            try:
                ra["hour"]=pd.to_datetime(ra["ALERT_WALL_TIME"],errors="coerce").dt.hour.dropna()
                hour_cnt=ra.groupby("hour").size().reset_index(name="Alert Count")
                st.subheader("Alert Frequency by Hour of Day")
                st.bar_chart(hour_cnt,x="hour",y="Alert Count")
                st.caption("Peaks around 02:00–04:00 and 14:00–16:00 match known circadian drowsiness windows.")
            except Exception: pass

    # Tab 1: Driver Trends
    with tabs[1]:
        st.subheader("Driver Week-over-Week Safety Trend")
        st.info("Queries DRIVER_WEEKLY_TREND view — a Snowflake view that aggregates sessions by ISO week.")
        wt=query_sf(f"SELECT * FROM {DATABASE}.{SCHEMA}.DRIVER_WEEKLY_TREND ORDER BY WEEK DESC LIMIT 200")
        if wt.empty: wt=fetch_recent("UNIFIED_DRIVER_SESSION_SUMMARY",200)
        if not wt.empty:
            st.dataframe(wt.head(100),hide_index=True,use_container_width=True)
            if {"DRIVER_ID","AVG_SCORE"}.issubset(wt.columns):
                st.line_chart(wt[["WEEK","AVG_SCORE"]].dropna() if "WEEK" in wt.columns else wt[["DRIVER_ID","AVG_SCORE"]])
        else: st.caption("No trend data yet.")
        # Personal baseline vs latest
        df_s=fetch_recent("UNIFIED_DRIVER_SESSION_SUMMARY",200)
        if not df_s.empty and {"DRIVER_ID","COMBINED_DRIVER_SAFETY_SCORE"}.issubset(df_s.columns):
            baseline=df_s.groupby("DRIVER_ID")["COMBINED_DRIVER_SAFETY_SCORE"].mean().reset_index()
            baseline.columns=["Driver","Baseline Safety Score"]
            latest=df_s.sort_values("CREATED_AT" if "CREATED_AT" in df_s.columns else df_s.columns[0],ascending=False)
            latest=latest.groupby("DRIVER_ID")["COMBINED_DRIVER_SAFETY_SCORE"].first().reset_index()
            latest.columns=["Driver","Latest Score"]
            cmp=pd.merge(baseline,latest,on="Driver",how="inner")
            cmp["Delta"]=cmp["Latest Score"]-cmp["Baseline Safety Score"]
            st.subheader("Personal Baseline vs Latest Session")
            st.dataframe(cmp,hide_index=True,use_container_width=True)
            st.caption("Positive Delta = performance declined vs baseline. Negative = improvement.")

    # Tab 2: Sessions
    with tabs[2]:
        df=fetch_recent("UNIFIED_DRIVER_SESSION_SUMMARY",200)
        if df.empty: st.caption("No sessions yet.")
        else:
            st.dataframe(df,hide_index=True,use_container_width=True)
            for col,label in [("COMBINED_DRIVER_SAFETY_SCORE","Safety Score"),("FATIGUE_SCORE","Fatigue"),("DISTRACTION_SCORE","Distraction")]:
                if {"DRIVER_ID",col}.issubset(df.columns):
                    agg=df.groupby("DRIVER_ID",as_index=False)[col].mean(); agg.columns=["Driver",label]
                    st.bar_chart(agg,x="Driver",y=label)
            if "DRIVER_RATING" in df.columns:
                rc=df["DRIVER_RATING"].value_counts().reset_index(); rc.columns=["Rating","Count"]
                st.bar_chart(rc,x="Rating",y="Count")

    # Tab 3: Scorecards
    with tabs[3]:
        sc=fetch_recent("DRIVER_SCORECARDS",200)
        if sc.empty: st.caption("No scorecards yet.")
        else:
            st.dataframe(sc,hide_index=True,use_container_width=True)
            for col in ["FATIGUE_RISK_SCORE","DISTRACTION_RISK_SCORE","COMBINED_DRIVER_SAFETY_SCORE"]:
                if {"DRIVER_ID",col}.issubset(sc.columns):
                    agg=sc.groupby("DRIVER_ID",as_index=False)[col].mean(); agg.columns=["Driver",col]
                    st.bar_chart(agg,x="Driver",y=col)

    # Tab 4: Driver Timeline
    with tabs[4]:
        tl=fetch_recent("DRIVER_TIMELINE",2000)
        if tl.empty: st.caption("No timeline data.")
        else:
            st.dataframe(tl.head(400),hide_index=True,use_container_width=True)
            if {"TIMELINE_STATE","DURATION_SECONDS"}.issubset(tl.columns):
                sd=tl.groupby("TIMELINE_STATE",as_index=False)["DURATION_SECONDS"].sum()
                sd.columns=["State","Total Duration (s)"]; st.bar_chart(sd,x="State",y="Total Duration (s)")

    # Tab 5: Real-time Alerts
    with tabs[5]:
        ra=fetch_recent("REALTIME_ALERTS",500)
        if ra.empty: st.caption("No real-time alerts yet.")
        else:
            st.dataframe(ra.head(300),hide_index=True,use_container_width=True)
            for col,label in [("SEVERITY","Severity"),("ALERT_TYPE","Alert Type"),("LIGHTING_AT_ALERT","Lighting at Alert")]:
                if col in ra.columns:
                    vc=ra[col].value_counts().reset_index(); vc.columns=[label,"Count"]
                    st.bar_chart(vc,x=label,y="Count")
            if {"SESSION_ID","ESCALATION_LEVEL"}.issubset(ra.columns):
                pk=ra.groupby("SESSION_ID",as_index=False)["ESCALATION_LEVEL"].max()
                pk.columns=["Session","Peak Escalation"]; st.bar_chart(pk,x="Session",y="Peak Escalation")

    # Tab 6: Recovery Events
    with tabs[6]:
        rv=fetch_recent("DRIVER_RECOVERY_EVENTS",500)
        if rv.empty: st.caption("No recovery events yet.")
        else:
            st.dataframe(rv,hide_index=True,use_container_width=True)
            if {"DRIVER_ID","PEAK_ESCALATION_BEFORE"}.issubset(rv.columns):
                agg=rv.groupby("DRIVER_ID",as_index=False)["PEAK_ESCALATION_BEFORE"].mean()
                agg.columns=["Driver","Avg Peak Before Recovery"]; st.bar_chart(agg,x="Driver",y="Avg Peak Before Recovery")

    # Tab 7: Frame Risk
    with tabs[7]:
        uf=fetch_recent("UNIFIED_FRAME_PREDICTIONS",3000)
        if uf.empty: st.caption("No frame predictions yet.")
        else:
            st.dataframe(uf.head(300),hide_index=True,use_container_width=True)
            for col,label in [("OVERALL_RISK_LABEL","Risk Label"),("ZONE_PRED","Gaze Zone"),("RISK_GROUP_PRED","Risk Group")]:
                if col in uf.columns:
                    rc=uf[col].value_counts().reset_index(); rc.columns=[label,"Count"]
                    st.bar_chart(rc,x=label,y="Count")

    # Tab 8: Zone Transitions
    with tabs[8]:
        zt=fetch_recent("MODULE_B_ZONE_TRANSITIONS",1000)
        if zt.empty: st.caption("No zone transitions yet.")
        else:
            st.dataframe(zt.head(300),hide_index=True,use_container_width=True)
            if {"FROM_ZONE","TO_ZONE"}.issubset(zt.columns):
                pair=(zt["FROM_ZONE"]+" → "+zt["TO_ZONE"]).value_counts().head(10).reset_index()
                pair.columns=["Transition","Count"]; st.bar_chart(pair,x="Transition",y="Count")

    # Tab 9: Lighting Report
    with tabs[9]:
        st.subheader("Lighting Quality Impact Report")
        af=fetch_recent("MODULE_A_FRAME_PREDICTIONS",5000)
        if not af.empty and "LIGHTING_METHOD" in af.columns:
            lm=af.groupby("LIGHTING_METHOD").agg(
                Frames=("FRAME_ID","count"),
                Avg_Quality=("QUALITY_SCORE","mean") if "QUALITY_SCORE" in af.columns else ("FRAME_ID","count"),
                Avg_Confidence=("A1_CONFIDENCE","mean") if "A1_CONFIDENCE" in af.columns else ("FRAME_ID","count")
            ).reset_index()
            st.dataframe(lm,hide_index=True,use_container_width=True)
            st.bar_chart(lm,x="LIGHTING_METHOD",y="Frames")
            st.caption("Sessions with overexposed or underexposed corrections should be cross-referenced with alert confidence to validate correction effectiveness.")
        else: st.caption("No frame data with lighting method yet.")

    # Tab 10: Review Queue
    with tabs[10]:
        rq=fetch_recent("FRAME_REVIEW_QUEUE",1000)
        if rq.empty: st.caption("No review queue items.")
        else:
            st.dataframe(rq.head(300),hide_index=True,use_container_width=True)
            if "DRIVER_ID" in rq.columns:
                bd=rq["DRIVER_ID"].value_counts().reset_index(); bd.columns=["Driver","Review Frames"]
                st.bar_chart(bd,x="Driver",y="Review Frames")

    # Tab 11: Snowflake Advanced
    with tabs[11]:
        st.subheader("⚡ Snowflake Advanced Features")
        st.markdown("Use these queries directly in Snowflake to demonstrate platform-native ML and audit capabilities.")

        with st.expander("🔮 Cortex FORECAST — Predict next session safety score"):
            st.code("""
-- Snowflake Cortex ML: forecast next 4 weeks of safety scores per driver
SELECT * FROM TABLE(
  FORECAST(
    INPUT_DATA => TABLE(
      SELECT TO_TIMESTAMP(SESSION_START_TS) AS ts,
             COMBINED_DRIVER_SAFETY_SCORE   AS y,
             DRIVER_ID                       AS series
      FROM DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
    ),
    SERIES_COLNAME => 'SERIES',
    TIMESTAMP_COLNAME => 'TS',
    TARGET_COLNAME => 'Y',
    FORECASTING_PERIODS => 4
  )
);
""",language="sql")

        with st.expander("🔍 Cortex ANOMALY DETECTION — Flag unusual alert sessions"):
            st.code("""
-- Detect sessions with statistically anomalous alert counts
SELECT * FROM TABLE(
  ANOMALY_DETECTION(
    INPUT_DATA => TABLE(
      SELECT TO_TIMESTAMP(SESSION_START_TS) AS ts,
             CAST(TOTAL_REALTIME_ALERTS AS FLOAT) AS y
      FROM DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
    ),
    TIMESTAMP_COLNAME => 'TS',
    TARGET_COLNAME => 'Y'
  )
);
""",language="sql")

        with st.expander("🕰️ Time Travel — Audit driver scores at any past point"):
            driver_tt=st.text_input("Driver ID for Time Travel","DRV_001",key="tt_driver")
            ts_tt=st.text_input("Point in time (UTC)","2024-06-01 00:00:00",key="tt_ts")
            if st.button("Run Time Travel Query"):
                tt_sql=f"""
SELECT SESSION_ID, SESSION_START_TS, COMBINED_DRIVER_SAFETY_SCORE, DRIVER_RATING, ESCALATION_LEVEL
FROM DEMO_DB.PUBLIC.UNIFIED_DRIVER_SESSION_SUMMARY
AT (TIMESTAMP => '{ts_tt}'::TIMESTAMP_NTZ)
WHERE DRIVER_ID = '{driver_tt}'
ORDER BY SESSION_START_TS DESC
LIMIT 20;
"""
                ttdf=query_sf(tt_sql)
                if ttdf.empty: st.caption("No data at that point in time (or Time Travel period expired).")
                else: st.dataframe(ttdf,hide_index=True,use_container_width=True)

        with st.expander("🏆 Dynamic Table — Driver Risk Leaderboard (auto-refreshing)"):
            lb_df=query_sf(f"SELECT * FROM {DATABASE}.{SCHEMA}.DRIVER_RISK_LEADERBOARD ORDER BY AVG_SAFETY_SCORE DESC LIMIT 50")
            if lb_df.empty: st.caption("Dynamic Table not created yet. Run the DDL from the Home page SQL block.")
            else:
                st.dataframe(lb_df,hide_index=True,use_container_width=True)
                if "DRIVER_ID" in lb_df.columns and "AVG_SAFETY_SCORE" in lb_df.columns:
                    st.bar_chart(lb_df,x="DRIVER_ID",y="AVG_SAFETY_SCORE")
            st.caption("This table refreshes every hour automatically inside Snowflake — no external scheduler or ETL pipeline needed.")

        with st.expander("📊 Custom Snowflake Query"):
            custom_sql=st.text_area("Enter SQL",f"SELECT * FROM {DATABASE}.{SCHEMA}.UNIFIED_DRIVER_SESSION_SUMMARY LIMIT 10",height=100)
            if st.button("Run Query"):
                cdf=query_sf(custom_sql)
                if cdf.empty: st.caption("No results.")
                else: st.dataframe(cdf,hide_index=True,use_container_width=True)

# =============================================================================
# NAVIGATION
# =============================================================================
pages=[
    st.Page(home_page,      title="Home",               icon=":material/home:"),
    st.Page(unified_page,   title="Real-time Analysis", icon=":material/security:"),
    st.Page(analytics_page, title="Historical Analytics",icon=":material/analytics:"),
]
pg=st.navigation(pages)
pg.run()
