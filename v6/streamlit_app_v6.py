# streamlit_app_v6.py — Driver Safety Analytics Platform v6.0.0
#
# CHANGES FROM v3:
#   1. EfficientNet-B0 backbone for A1 and A2 (was ResNet18)
#   2. Ensemble inference: A1 (folds 1,2,4), A2 (folds 1,3,4), B (folds 1-6)
#   3. YuNet face detection via cv2.FaceDetectorYN — no MediaPipe needed
#   4. Updated thresholds from training: offroad=0.30, zone_T=1.034
#   5. A2 input: 224x224 square (was 64x224 — mismatch with training)
#   6. A1 yawn/nod heads now produce non-zero predictions (Focal Loss training)
#   7. Per-fold temperature calibration for Module B ensemble

import io
import os
import tempfile
import time
import uuid
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
    import torch.nn as nn
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
    from snowflake.snowpark.context import get_active_session
    _session = get_active_session()
except Exception:
    try:
        from snowflake.snowpark import Session
        _session = Session.builder.config("connection_name", "default").create()
    except Exception:
        _session = None

# ─── Identity ─────────────────────────────────────────────────────────────────
DATABASE    = "DEMO_DB"
SCHEMA      = "PUBLIC"
APP_VERSION = "6.0.0"

MODEL_VERSION_A1 = "a1_v4_efficientnet_b0_ensemble_folds_1_2_4"
MODEL_VERSION_A2 = "a2_v4_efficientnet_b0_ensemble_folds_1_3_4"
MODEL_VERSION_B  = "b_v4_efficientnet_b0_ensemble_folds_1_to_6"

_MODEL_DIR = tempfile.mkdtemp()

# ─── Zone / risk constants ────────────────────────────────────────────────────
ZONE_CLASSES = [
    "Forward",       # 0
    "Lap",           # 1
    "Left Mirror",   # 2
    "Radio",         # 3
    "Rearview",      # 4
    "Right Mirror",  # 5
    "Shoulder",      # 6
    "Speedometer",   # 7
]
RISK_GROUPS = {
    "Safe":     ["Forward"],
    "LowRisk":  ["Left Mirror", "Right Mirror", "Rearview"],
    "HighRisk": ["Lap", "Radio", "Speedometer", "Shoulder"],
}

# ─── Inference thresholds (from v4 training) ──────────────────────────────────
# Module B — from b_ensemble_meta.json
OFFROAD_THRESHOLD = 0.30      # v3 was 0.35
ZONE_TEMPERATURE  = 1.034     # v3 was 2.601 (model now much better calibrated)

# Per-fold temperatures and thresholds for B ensemble (from b_ensemble_meta.json)
_B_FOLD_TEMPERATURES = {1: 1.000, 2: 1.000, 3: 1.000, 4: 1.000, 5: 1.000, 6: 1.201}
_B_FOLD_OFFROAD_THRS = {1: 0.300, 2: 0.305, 3: 0.300, 4: 0.300, 5: 0.300, 6: 0.300}

# Module A1 — mean thresholds from a1_ensemble_meta.json
A1_THR_DROWSY = 0.25
A1_THR_YAWN   = 0.27
A1_THR_NOD    = 0.47

# Module A2 — mean threshold from a2_ensemble_meta.json
A2_THR_EYE_CLOSED = 0.421

# Ensemble fold file lists
_A1_FOLD_FILES = ["a1_fold_1_best.pt", "a1_fold_2_best.pt", "a1_fold_4_best.pt"]
_A2_FOLD_FILES = ["a2_fold_1_best.pt", "a2_fold_3_best.pt", "a2_fold_4_best.pt"]
_B_FOLD_FILES  = [f"b_fold_{i}_best.pt" for i in range(1, 7)]

# YuNet face detector model (upload to @DRIVER_SAFETY_MODELS stage)
_YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"

# ─── Image normalisation ──────────────────────────────────────────────────────
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
IMG_SIZE  = 224

# ─── Quality / alert constants ────────────────────────────────────────────────
MIN_ALERT_CONF             = 0.45
MAX_UNCERTAIN_ENTROPY      = 1.55
MIN_FRAME_QUALITY          = 0.35
EAR_CLOSED_THRESHOLD       = 0.21
POOR_QUALITY_SESSION_RATIO = 0.35
OFFROAD_EXIT_HYSTERESIS    = 0.08
MAX_EXTRACTION_FRAMES      = 4000

# ─── Default fatigue weighting ────────────────────────────────────────────────
DEFAULT_A = {
    "w_drowsy":      0.30,
    "w_eye_closed":  0.30,
    "w_nod":         0.15,
    "w_yawn":        0.10,
    "w_perclos":     0.10,
    "w_blink":       0.05,
    "t_eye":         A2_THR_EYE_CLOSED,
    "t_caution":     0.40,
    "t_alert":       0.70,
    "closure_frames":5,
    "fps":           5,
    "win_s":         10,
    "win_m":         30,
    "win_l":         60,
    "ear_closed":    EAR_CLOSED_THRESHOLD,
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp01(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float(default)
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return float(default)


def _make_rng(source_name: str, frame_idx: int):
    key  = f"{source_name}:{frame_idx}".encode("utf-8")
    seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def get_session():
    return _session


def _download_model(filename: str):
    path = os.path.join(_MODEL_DIR, filename)
    if os.path.exists(path):
        return path
    s = get_session()
    if s is None:
        return None
    try:
        s.file.get(
            f"@{DATABASE}.{SCHEMA}.DRIVER_SAFETY_MODELS/{filename}", _MODEL_DIR)
        if os.path.exists(path):
            return path
        gz_path = path + ".gz"
        if os.path.exists(gz_path):
            import gzip, shutil
            with gzip.open(gz_path, "rb") as fi, open(path, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(gz_path)
            return path
        return None
    except Exception:
        return None


def write_to_snowflake(df: pd.DataFrame, table_name: str, retries: int = 2):
    s = get_session()
    if s is None or df.empty:
        return False
    upload = df.copy()
    if "created_at_client" not in upload.columns:
        upload["created_at_client"] = utc_now_iso()
    upload.columns = [c.upper() for c in upload.columns]
    for attempt in range(retries + 1):
        try:
            s.write_pandas(upload, table_name,
                           database=DATABASE, schema=SCHEMA, overwrite=False)
            return True
        except Exception as e:
            if attempt == retries:
                st.toast(f"DB write warning for {table_name}: {e}",
                         icon=":material/warning:")
                return False
            time.sleep(0.3 * (attempt + 1))
    return False


def query_sf(sql: str):
    s = get_session()
    if s is None:
        return pd.DataFrame()
    try:
        return s.sql(sql).to_pandas()
    except Exception:
        return pd.DataFrame()


def fetch_recent(table_name: str, limit: int = 100):
    q1 = (f"SELECT * FROM {DATABASE}.{SCHEMA}.{table_name} "
          f"ORDER BY CREATED_AT_CLIENT DESC LIMIT {limit}")
    df = query_sf(q1)
    if not df.empty:
        return df
    q2 = (f"SELECT * FROM {DATABASE}.{SCHEMA}.{table_name} "
          f"ORDER BY CREATED_AT DESC LIMIT {limit}")
    return query_sf(q2)


def get_risk_group(zone: str):
    for group, zones in RISK_GROUPS.items():
        if zone in zones:
            return group
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# All three modules now use tf_efficientnet_b0.ns_jft_in1k backbone.
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class _A1Model(nn.Module):
        """
        EfficientNet-B0 + 3 binary heads: drowsy / yawn / nod.
        Forward returns sigmoid probabilities (same shape as app expects).
        Matches mod_a1_training_v4.py architecture exactly.
        """
        def __init__(self):
            super().__init__()
            import timm
            self.bb      = timm.create_model(
                "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=0)
            nf           = self.bb.num_features
            self.dropout = nn.Dropout(p=0.30)
            self.h_d     = nn.Linear(nf, 1)
            self.h_y     = nn.Linear(nf, 1)
            self.h_n     = nn.Linear(nf, 1)

        def forward(self, x):
            f = self.dropout(self.bb(x))
            return {
                "drowsy": torch.sigmoid(self.h_d(f)).squeeze(-1),
                "yawn":   torch.sigmoid(self.h_y(f)).squeeze(-1),
                "nod":    torch.sigmoid(self.h_n(f)).squeeze(-1),
            }

    class _A2Model(nn.Module):
        """
        EfficientNet-B0 + 1 binary head: eye_closed.
        Matches mod_a2_training_v4.py architecture exactly.
        """
        def __init__(self):
            super().__init__()
            import timm
            self.bb      = timm.create_model(
                "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=0)
            self.dropout = nn.Dropout(p=0.30)
            self.head    = nn.Linear(self.bb.num_features, 1)

        def forward(self, x):
            f = self.dropout(self.bb(x))
            return {"eye_closed": torch.sigmoid(self.head(f)).squeeze(-1)}

    class _BModel(nn.Module):
        """
        EfficientNet-B0 + zone head (8-class) + offroad head (binary).
        riskgrp_head is present in checkpoint weights but NOT defined here.
        strict=False at loading time silently ignores those extra weights.
        """
        def __init__(self, bb, nf):
            super().__init__()
            self.bb           = bb
            self.zone_head    = nn.Linear(nf, 8)
            self.offroad_head = nn.Linear(nf, 1)

        def forward(self, x):
            f = self.bb(x)
            return {
                "zone_logits":   self.zone_head(f),
                "offroad_logit": self.offroad_head(f),
            }


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_a1():
    """Load A1 ensemble (folds 1, 2, 4). Returns list of models or None."""
    if not TORCH_AVAILABLE:
        return None
    models = []
    for fname in _A1_FOLD_FILES:
        p = _download_model(fname)
        if not p:
            continue
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            mdl  = _A1Model()
            sd   = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            mdl.load_state_dict(sd, strict=True)
            mdl.eval()
            models.append(mdl)
        except Exception:
            pass
    return models if models else None


@st.cache_resource
def load_a2():
    """Load A2 ensemble (folds 1, 3, 4). Returns list of models or None."""
    if not TORCH_AVAILABLE:
        return None
    models = []
    for fname in _A2_FOLD_FILES:
        p = _download_model(fname)
        if not p:
            continue
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            mdl  = _A2Model()
            sd   = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            mdl.load_state_dict(sd, strict=True)
            mdl.eval()
            models.append(mdl)
        except Exception:
            pass
    return models if models else None


@st.cache_resource
def load_b():
    """
    Load B ensemble (folds 1-6).
    Returns list of (model, temperature, offroad_threshold) tuples or None.
    Uses strict=False to ignore riskgrp_head weights from training checkpoints.
    """
    if not TORCH_AVAILABLE:
        return None
    ensemble = []
    for fold_num, fname in enumerate(_B_FOLD_FILES, start=1):
        p = _download_model(fname)
        if not p:
            continue
        try:
            import timm
            bb  = timm.create_model(
                "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=0)
            mdl = _BModel(bb, bb.num_features)
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            sd   = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            mdl.load_state_dict(sd, strict=False)   # ignores riskgrp_head
            mdl.eval()
            T   = float(ckpt.get("temperature",
                                  _B_FOLD_TEMPERATURES.get(fold_num, 1.0)))
            thr = float(ckpt.get("offroad_threshold",
                                  _B_FOLD_OFFROAD_THRS.get(fold_num, 0.30)))
            ensemble.append((mdl, T, thr))
        except Exception:
            pass
    return ensemble if ensemble else None


# ══════════════════════════════════════════════════════════════════════════════
# FACE DETECTION — YuNet via cv2.FaceDetectorYN
# YuNet model: face_detection_yunet_2023mar.onnx (~70 KB)
# Upload to @DEMO_DB.PUBLIC.DRIVER_SAFETY_MODELS stage.
# Falls back to Haar cascade if model is unavailable.
# cv2.FaceDetectorYN is part of opencv-python >= 4.5.4
# (available on Snowflake Anaconda channel).
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_yunet_detector():
    if not CV2_AVAILABLE:
        return None
    model_path = _download_model(_YUNET_MODEL_FILE)
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        fd = cv2.FaceDetectorYN.create(
            model=model_path, config="",
            input_size=(320, 320),
            score_threshold=0.60,
            nms_threshold=0.30,
            top_k=1,
        )
        return fd
    except Exception:
        return None


def _yunet_detect(frame_bgr, detector):
    """
    Run YuNet on frame. Returns (face_224, eye_strip, score) or (None,None,0).
    YuNet detection row layout (15 values):
      [0]=x [1]=y [2]=w [3]=h
      [4]=re_x [5]=re_y  (right eye center)
      [6]=le_x [7]=le_y  (left eye center)
      [8]=nose_x [9]=nose_y
      [10]=rm_x [11]=rm_y  (right mouth)
      [12]=lm_x [13]=lm_y  (left mouth)
      [14]=score
    """
    h, w = frame_bgr.shape[:2]
    try:
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame_bgr)
    except Exception:
        return None, None, 0.0

    if faces is None or len(faces) == 0:
        return None, None, 0.0

    det   = faces[0]
    bx, by, bw, bh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
    score = float(det[14]) if len(det) > 14 else 0.0

    # Face crop with 20% padding (matches training face_pad=0.20)
    px = int(bw * 0.20); py = int(bh * 0.20)
    fx1 = max(0, bx - px);      fy1 = max(0, by - py)
    fx2 = min(w, bx + bw + px); fy2 = min(h, by + bh + py)
    if fx2 <= fx1 or fy2 <= fy1:
        return None, None, score

    face_crop = cv2.resize(frame_bgr[fy1:fy2, fx1:fx2],
                           (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Eye strip from YuNet 5-point landmarks
    re_x, re_y = float(det[4]), float(det[5])   # right eye center
    le_x, le_y = float(det[6]), float(det[7])   # left eye center
    eye_dist   = abs(le_x - re_x)
    eye_radius = max(bh * 0.10, 12)
    sx1 = max(0, int(re_x - eye_dist * 0.45))
    sx2 = min(w, int(le_x + eye_dist * 0.45))
    sy1 = max(0, int(min(re_y, le_y) - eye_radius * 1.8))
    sy2 = min(h, int(max(re_y, le_y) + eye_radius * 1.8))

    eye_strip = frame_bgr[sy1:sy2, sx1:sx2] if sy2 > sy1 and sx2 > sx1 else None
    if eye_strip is None or eye_strip.size == 0:
        # Fallback: use upper 45% of face crop as eye strip
        eye_strip = face_crop[:int(IMG_SIZE * 0.45), :]

    return face_crop, eye_strip, score


def _haar_fallback(frame_bgr, gray):
    """Haar cascade / centre-crop fallback when YuNet unavailable."""
    h, w = frame_bgr.shape[:2]
    cascade_path = None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    except Exception:
        pass

    if cascade_path and os.path.exists(cascade_path):
        fc    = cv2.CascadeClassifier(cascade_path)
        faces = fc.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            face  = cv2.resize(frame_bgr[fy:fy+fh, fx:fx+fw],
                               (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            h1, h2 = int(fh * 0.15), int(fh * 0.45)
            strip = frame_bgr[fy+h1:fy+h2, fx:fx+fw]
            return face, strip if strip.size > 0 else face[:int(IMG_SIZE*0.45), :], True

    # Centre-crop last resort
    cx, cy = w // 2, h // 2
    fwc, fhc = int(w * 0.5), int(h * 0.6)
    x1 = max(0, cx - fwc // 2); y1 = max(0, cy - fhc // 2)
    face = cv2.resize(frame_bgr[y1:y1+fhc, x1:x1+fwc],
                      (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    strip = face[:int(IMG_SIZE * 0.45), :]
    return face, strip, False


def _normalized_variance_laplacian(gray_img):
    if gray_img is None or gray_img.size == 0:
        return 0.0
    return float(np.clip(cv2.Laplacian(gray_img, cv2.CV_64F).var() / 300.0, 0.0, 1.0))


def _brightness_score(gray_img):
    if gray_img is None or gray_img.size == 0:
        return 0.0
    return float(np.clip(1.0 - abs(float(np.mean(gray_img)) - 128.0) / 128.0, 0.0, 1.0))


def _safe_crop(img, x1, y1, x2, y2):
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def detect_face_eyes(frame_bgr):
    """
    Detect face and extract crops for A1 (face) and A2 (eye strip).

    Priority:
      1. YuNet ONNX via cv2.FaceDetectorYN (requires face_detection_yunet_2023mar.onnx in stage)
      2. Haar cascade (built-in OpenCV — no extra files needed)
      3. Centre-crop last resort

    MediaPipe is no longer used. YuNet is fully compatible with Snowflake's
    opencv-python package and produces consistent crops matching training.

    EAR is not available from YuNet's 5-point landmarks (needs 6 per eye).
    temporal_a handles ear=None gracefully — A2 model becomes the primary
    eye-closure signal rather than a complement to EAR.
    """
    default = {
        "face": None, "eye_strip": None,
        "face_detected": False, "preprocess_method": "none",
        "quality_score": 0.0, "face_confidence": 0.0,
        "brightness_score": 0.0, "blur_score": 0.0,
        "ear": None, "left_ear": None, "right_ear": None,
        "head_pitch": None, "head_yaw": None, "head_roll": None,
    }
    if not CV2_AVAILABLE:
        return default

    gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = _brightness_score(gray)
    blur       = _normalized_variance_laplacian(gray)

    # ── Try YuNet ──────────────────────────────────────────────────────────────
    detector = _get_yunet_detector()
    if detector is not None:
        face_crop, eye_strip, score = _yunet_detect(frame_bgr, detector)
        if face_crop is not None:
            quality = float(np.clip(
                0.40 * blur + 0.30 * brightness + 0.30 * min(score, 1.0), 0.0, 1.0))
            return {
                "face": face_crop, "eye_strip": eye_strip,
                "face_detected": True, "preprocess_method": "yunet",
                "quality_score": quality,
                "face_confidence": float(np.clip(score, 0.0, 1.0)),
                "brightness_score": brightness, "blur_score": blur,
                # EAR not available from 5-point landmarks — handled gracefully downstream
                "ear": None, "left_ear": None, "right_ear": None,
                "head_pitch": None, "head_yaw": None, "head_roll": None,
            }

    # ── Haar / centre-crop fallback ────────────────────────────────────────────
    face_crop, eye_strip, haar_ok = _haar_fallback(frame_bgr, gray)
    quality = float(np.clip(0.40*blur + 0.35*brightness + (0.10 if haar_ok else 0.0), 0.0, 1.0))
    return {
        "face": face_crop, "eye_strip": eye_strip,
        "face_detected": haar_ok,
        "preprocess_method": "haar" if haar_ok else "center_crop",
        "quality_score": quality,
        "face_confidence": 0.55 if haar_ok else 0.20,
        "brightness_score": brightness, "blur_score": blur,
        "ear": None, "left_ear": None, "right_ear": None,
        "head_pitch": None, "head_yaw": None, "head_roll": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE — ENSEMBLE AVERAGING
# ══════════════════════════════════════════════════════════════════════════════

def _certainty_from_prob(prob):
    if prob is None:
        return 0.0
    return float(np.clip(abs(prob - 0.5) * 2.0, 0.0, 1.0))


def _make_tensor_224(img_bgr):
    """Convert BGR crop to normalised 224x224 float tensor (B,C,H,W)."""
    if img_bgr is None or not PIL_AVAILABLE:
        return None
    try:
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        t   = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
        return t(img).unsqueeze(0)
    except Exception:
        return None


def infer_a1(models, face):
    """
    Ensemble A1 inference.
    Averages sigmoid outputs across all fold models.
    Returns per-head probabilities and ensemble size.
    """
    if not models or face is None or not TORCH_AVAILABLE:
        return None
    tensor = _make_tensor_224(face)
    if tensor is None:
        return None
    try:
        probs = {"drowsy": [], "yawn": [], "nod": []}
        for mdl in models:
            with torch.no_grad():
                o = mdl(tensor)
            for h in ("drowsy", "yawn", "nod"):
                probs[h].append(float(o[h].item()))

        drowsy = _clamp01(float(np.mean(probs["drowsy"])))
        yawn   = _clamp01(float(np.mean(probs["yawn"])))
        nod    = _clamp01(float(np.mean(probs["nod"])))
        return {
            "a1_prob_drowsy":   drowsy,
            "a1_prob_yawn":     yawn,
            "a1_prob_nod":      nod,
            "a1_confidence":    float(np.mean([
                _certainty_from_prob(drowsy),
                _certainty_from_prob(yawn),
                _certainty_from_prob(nod),
            ])),
            "a1_ensemble_size": len(models),
        }
    except Exception:
        return None


def infer_a2(models, strip):
    """
    Ensemble A2 inference.
    Input is resized to 224x224 (critical fix — v3 used 64x224).
    Averages sigmoid outputs across fold models.
    """
    if not models or strip is None or not TORCH_AVAILABLE:
        return None
    tensor = _make_tensor_224(strip)   # 224x224 square — matches training
    if tensor is None:
        return None
    try:
        probs = []
        for mdl in models:
            with torch.no_grad():
                o = mdl(tensor)
            probs.append(float(o["eye_closed"].item()))
        eye_closed = _clamp01(float(np.mean(probs)))
        return {
            "a2_prob_eye_closed":    eye_closed,
            "a2_eye_openness_score": _clamp01(1.0 - eye_closed),
            "a2_confidence":         _certainty_from_prob(eye_closed),
            "a2_ensemble_size":      len(models),
        }
    except Exception:
        return None


def infer_b(ensemble, frame):
    """
    Ensemble B inference (6 folds).
    Zone: average softmax(logits / T) across folds.
    Off-road: average sigmoid(logit) across folds, then threshold.
    Each fold contributes its own calibrated T from training.
    """
    if not ensemble or not PIL_AVAILABLE or not CV2_AVAILABLE:
        return None
    tensor = _make_tensor_224(frame)
    if tensor is None:
        return None
    try:
        zone_probs_list = []
        offroad_probs   = []
        for mdl, T, thr in ensemble:
            with torch.no_grad():
                o = mdl(tensor)
            zl = o["zone_logits"][0] / max(float(T), 0.01)
            zp = torch.softmax(zl, dim=0).detach().cpu().numpy()
            zone_probs_list.append(zp)
            op = float(torch.sigmoid(o["offroad_logit"][0]).item())
            offroad_probs.append(op)

        zp_avg     = np.mean(zone_probs_list, axis=0)   # shape (8,)
        op_avg     = float(np.mean(offroad_probs))
        sorted_idx = np.argsort(zp_avg)[::-1]
        zi         = int(sorted_idx[0])
        confidence = float(zp_avg.max())
        entropy    = float(-np.sum(zp_avg * np.log(zp_avg + 1e-8)))
        margin     = float(zp_avg[sorted_idx[0]] - zp_avg[sorted_idx[1]]) \
                     if len(sorted_idx) > 1 else confidence
        uncertainty= float(np.clip(
            (entropy / np.log(len(ZONE_CLASSES))) * 0.6 + (1.0 - confidence) * 0.4, 0.0, 1.0))
        top2       = [ZONE_CLASSES[int(i)] for i in sorted_idx[:2]]

        return {
            "zone_pred":         ZONE_CLASSES[zi],
            "risk_group_pred":   get_risk_group(ZONE_CLASSES[zi]),
            "offroad_prob":      _clamp01(op_avg),
            "offroad_pred":      1 if op_avg >= OFFROAD_THRESHOLD else 0,
            "confidence":        confidence,
            "entropy":           entropy,
            "margin":            margin,
            "uncertainty_score": uncertainty,
            "zone_top2":         " | ".join(top2),
            "b_ensemble_size":   len(ensemble),
            **{f"zone_prob_{z.lower().replace(' ', '_')}": float(zp_avg[i])
               for i, z in enumerate(ZONE_CLASSES)},
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION FALLBACKS (deterministic — same input gives same output)
# ══════════════════════════════════════════════════════════════════════════════

def sim_a(rng):
    eye_closed = float(rng.beta(2, 5))
    drowsy     = float(rng.beta(2, 5))
    return {
        "a1_prob_drowsy":        drowsy,
        "a1_prob_yawn":          float(rng.beta(1.5, 8)),
        "a1_prob_nod":           float(rng.beta(1.5, 6)),
        "a2_prob_eye_closed":    eye_closed,
        "a2_eye_openness_score": float(1.0 - eye_closed),
        "a1_confidence":         float(rng.uniform(0.45, 0.85)),
        "a2_confidence":         float(rng.uniform(0.45, 0.85)),
    }


def sim_b(rng, offroad_threshold=OFFROAD_THRESHOLD):
    pr        = rng.dirichlet([5, 1, 1.5, 0.8, 1.5, 1.5, 0.5, 0.8])
    zi        = int(np.argmax(pr))
    zone_name = ZONE_CLASSES[zi]
    op        = float(np.clip(1 - pr[0] + rng.normal(0, 0.05), 0, 1))
    confidence= float(pr.max())
    entropy   = float(-np.sum(pr * np.log(pr + 1e-8)))
    sidx      = np.argsort(pr)[::-1]
    sprobs    = pr[sidx]
    margin    = float(sprobs[0] - sprobs[1]) if len(sprobs) > 1 else confidence
    uncertainty = float(np.clip(
        (entropy / np.log(len(ZONE_CLASSES))) * 0.6 + (1.0 - confidence) * 0.4, 0.0, 1.0))
    return {
        "zone_pred":         zone_name,
        "risk_group_pred":   get_risk_group(zone_name),
        "offroad_prob":      op,
        "offroad_pred":      1 if op >= offroad_threshold else 0,
        "confidence":        confidence,
        "entropy":           entropy,
        "margin":            margin,
        "uncertainty_score": uncertainty,
        "zone_top2":         " | ".join([ZONE_CLASSES[int(i)] for i in sidx[:2]]),
        **{f"zone_prob_{z.lower().replace(' ', '_')}": float(pr[i])
           for i, z in enumerate(ZONE_CLASSES)},
    }


# ============================================================================
# TEMPORAL PROCESSING AND PIPELINE — unchanged from v3
# ============================================================================

def _streak_events(binary, fps, min_dur, label):
    events = []
    start = None
    for i, v in enumerate(binary):
        if int(v) == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                dur = (i - start) / fps
                if dur >= min_dur:
                    events.append({"type": label, "start": start / fps, "end": (i - 1) / fps, "dur": dur})
                start = None
    if start is not None:
        dur = (len(binary) - start) / fps
        if dur >= min_dur:
            events.append({"type": label, "start": start / fps, "end": (len(binary) - 1) / fps, "dur": dur})
    return events


def _derive_blinks(df, fps):
    blink_events = []
    blink_binary = []
    min_frames = max(1, int(0.08 * fps))
    max_frames = max(min_frames + 1, int(0.8 * fps))
    start = None
    src = df["eye_closed_binary"].fillna(0).astype(int).tolist()
    for i, v in enumerate(src):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            length = i - start
            if min_frames <= length <= max_frames:
                dur = length / fps
                blink_events.append({"type": "blink", "start": start / fps, "end": (i - 1) / fps, "dur": dur})
                blink_binary.extend([1] * length)
            else:
                blink_binary.extend([0] * length)
            start = None
        elif v == 0:
            blink_binary.append(0)
    if start is not None:
        length = len(src) - start
        if min_frames <= length <= max_frames:
            dur = length / fps
            blink_events.append({"type": "blink", "start": start / fps, "end": (len(src) - 1) / fps, "dur": dur})
            blink_binary.extend([1] * length)
        else:
            blink_binary.extend([0] * length)
    if len(blink_binary) < len(df):
        blink_binary.extend([0] * (len(df) - len(blink_binary)))
    return blink_events, np.array(blink_binary[: len(df)], dtype=int)


def _severity_bucket(score, duration):
    combined = float(score + min(duration / 4.0, 1.0) * 0.25)
    if combined >= 0.9:
        return "critical", min(1.0, combined)
    if combined >= 0.7:
        return "high", combined
    if combined >= 0.45:
        return "medium", combined
    return "low", combined


def _alert_type_from_severity(severity):
    return {
        "critical": "dashboard_critical",
        "high": "dashboard_alert",
        "medium": "dashboard_warning",
        "low": "none",
    }.get(severity, "none")


def temporal_a(df, cfg):
    if df.empty:
        return df, {}, []
    df = df.copy()
    fps = max(1, int(cfg["fps"]))

    for c in ["a1_prob_drowsy", "a1_prob_yawn", "a1_prob_nod", "a2_prob_eye_closed", "a1_confidence", "a2_confidence"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(0, 1)
    df["ear"] = pd.to_numeric(df.get("ear", np.nan), errors="coerce")
    df["quality_score"] = pd.to_numeric(df.get("quality_score", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    df["face_confidence"] = pd.to_numeric(df.get("face_confidence", 0.0), errors="coerce").fillna(0.0).clip(0, 1)

    smooth_w = max(1, int(0.6 * fps))
    df["a1_prob_drowsy_sm"] = df["a1_prob_drowsy"].rolling(smooth_w, min_periods=1).mean()
    df["a1_prob_yawn_sm"] = df["a1_prob_yawn"].rolling(smooth_w, min_periods=1).mean()
    df["a1_prob_nod_sm"] = df["a1_prob_nod"].rolling(smooth_w, min_periods=1).mean()
    df["a2_prob_eye_closed_sm"] = df["a2_prob_eye_closed"].rolling(smooth_w, min_periods=1).mean()

    df["preprocess_confidence"] = (df["quality_score"] * 0.6 + df["face_confidence"] * 0.4).clip(0, 1)
    df["eye_closed_binary"] = (
        (df["a2_prob_eye_closed_sm"] >= cfg["t_eye"]) | (df["ear"].notna() & (df["ear"] <= cfg["ear_closed"]))
    ).astype(int)

    blink_events, blink_binary = _derive_blinks(df, fps)
    df["blink_binary"] = blink_binary

    win_s = max(1, int(cfg["win_s"] * fps))
    win_m = max(1, int(cfg["win_m"] * fps))
    win_l = max(1, int(cfg["win_l"] * fps))

    df["perclos_30s"] = df["eye_closed_binary"].rolling(win_m, min_periods=1).mean()
    df["perclos_60s"] = df["eye_closed_binary"].rolling(win_l, min_periods=1).mean()
    df["blink_rate_per_min"] = df["blink_binary"].rolling(win_m, min_periods=1).sum() / max(cfg["win_m"] / 60.0, 1e-6)
    df["avg_blink_duration_s"] = 0.0
    for evt in blink_events:
        s = int(evt["start"] * fps)
        e = int(evt["end"] * fps) + 1
        df.loc[s : e - 1, "avg_blink_duration_s"] = evt["dur"]
    df["avg_blink_duration_s"] = df["avg_blink_duration_s"].rolling(win_m, min_periods=1).mean()

    for wn, ws in [("s", cfg["win_s"]), ("m", cfg["win_m"]), ("l", cfg["win_l"])]:
        w = max(1, int(ws * fps))
        df[f"rd_{wn}"] = df["a1_prob_drowsy_sm"].rolling(w, min_periods=1).mean()
        df[f"rdmx_{wn}"] = df["a1_prob_drowsy_sm"].rolling(w, min_periods=1).max()
        df[f"re_{wn}"] = df["a2_prob_eye_closed_sm"].rolling(w, min_periods=1).mean()
        df[f"ry_{wn}"] = df["a1_prob_yawn_sm"].rolling(w, min_periods=1).mean()
        df[f"rn_{wn}"] = df["a1_prob_nod_sm"].rolling(w, min_periods=1).mean()

    df["fatigue_signal_confidence"] = (
        0.35 * df["preprocess_confidence"] + 0.35 * df["a1_confidence"] + 0.30 * df["a2_confidence"]
    ).clip(0, 1)
    df["uncertain_frame"] = (
        (df["fatigue_signal_confidence"] < MIN_ALERT_CONF)
        | (df["quality_score"] < MIN_FRAME_QUALITY)
        | (df.get("preprocess_method", "none") == "center_crop")
    ).astype(int)
    df["review_flag"] = np.where(df["uncertain_frame"] == 1, "needs_review", "clear")

    normalized_blink = np.clip(df["blink_rate_per_min"] / 20.0, 0.0, 1.0)
    df["fatigue_score_raw"] = (
        cfg["w_drowsy"] * df["a1_prob_drowsy_sm"]
        + cfg["w_eye_closed"] * df["a2_prob_eye_closed_sm"]
        + cfg["w_nod"] * df["a1_prob_nod_sm"]
        + cfg["w_yawn"] * df["a1_prob_yawn_sm"]
        + cfg["w_perclos"] * df["perclos_30s"]
        + cfg["w_blink"] * normalized_blink
    ).clip(0, 1)
    df["fatigue_score_calibrated"] = (df["fatigue_score_raw"] * (0.65 + 0.35 * df["preprocess_confidence"])).clip(0, 1)
    df["frame_risk"] = np.where(
        df["uncertain_frame"] == 1,
        df["fatigue_score_calibrated"] * 0.70,
        df["fatigue_score_calibrated"],
    ).clip(0, 1)

    alert_threshold_adj = np.clip(cfg["t_alert"] + (0.5 - df["preprocess_confidence"]) * 0.10, 0.5, 0.9)
    caution_threshold_adj = np.clip(cfg["t_caution"] + (0.5 - df["preprocess_confidence"]) * 0.08, 0.3, 0.8)
    df["risk_level"] = np.where(
        df["frame_risk"] >= alert_threshold_adj,
        "high",
        np.where(df["frame_risk"] >= caution_threshold_adj, "medium", "low"),
    )
    df.loc[df["uncertain_frame"] == 1, "risk_level"] = df.loc[df["uncertain_frame"] == 1, "risk_level"] + "_uncertain"

    closure_events = []
    for evt in _streak_events(df["eye_closed_binary"].values, fps, cfg["closure_frames"] / fps, "prolonged_eye_closure"):
        mask = (df["timestamp_seconds"] >= evt["start"]) & (df["timestamp_seconds"] <= evt["end"])
        conf = float(df.loc[mask, "fatigue_signal_confidence"].mean()) if mask.any() else 0.0
        sev, sev_score = _severity_bucket(max(df.loc[mask, "frame_risk"].mean(), 0.55) if mask.any() else 0.55, evt["dur"])
        closure_events.append(
            {
                **evt,
                "severity": sev,
                "severity_score": sev_score,
                "confidence": conf,
                "alert_sent": conf >= MIN_ALERT_CONF and sev in ["high", "critical"],
                "alert_type": _alert_type_from_severity(sev),
                "event_confirmation_status": "pending_review" if conf < MIN_ALERT_CONF else "auto_confirmed",
                "risk_group": "fatigue",
            }
        )

    high_binary = (df["frame_risk"] >= cfg["t_alert"]).astype(int)
    high_fatigue_events = []
    for evt in _streak_events(high_binary.values, fps, 2.0, "high_fatigue"):
        mask = (df["timestamp_seconds"] >= evt["start"]) & (df["timestamp_seconds"] <= evt["end"])
        conf = float(df.loc[mask, "fatigue_signal_confidence"].mean()) if mask.any() else 0.0
        sev, sev_score = _severity_bucket(max(df.loc[mask, "frame_risk"].mean(), 0.60) if mask.any() else 0.60, evt["dur"])
        high_fatigue_events.append(
            {
                **evt,
                "severity": sev,
                "severity_score": sev_score,
                "confidence": conf,
                "alert_sent": conf >= MIN_ALERT_CONF and sev in ["high", "critical"],
                "alert_type": _alert_type_from_severity(sev),
                "event_confirmation_status": "pending_review" if conf < MIN_ALERT_CONF else "auto_confirmed",
                "risk_group": "fatigue",
            }
        )

    uncertain_events = _streak_events(df["uncertain_frame"].values, fps, 2.0, "uncertain_segment_review")
    uncertain_event_rows = []
    for evt in uncertain_events:
        uncertain_event_rows.append(
            {
                **evt,
                "severity": "medium",
                "severity_score": 0.50,
                "confidence": 0.20,
                "alert_sent": False,
                "alert_type": "review_queue",
                "event_confirmation_status": "pending_review",
                "risk_group": "review",
            }
        )

    all_events = closure_events + high_fatigue_events + uncertain_event_rows

    total_duration = len(df) / fps if fps > 0 else 0
    blink_count = len(blink_events)
    avg_blink_dur = float(np.mean([e["dur"] for e in blink_events])) if blink_events else 0.0
    high_risk_dur = float((df["frame_risk"] >= cfg["t_alert"]).sum() / fps)
    caution_dur = float((((df["frame_risk"] >= cfg["t_caution"]) & (df["frame_risk"] < cfg["t_alert"]))).sum() / fps)
    uncertain_ratio = float(df["uncertain_frame"].mean())
    poor_quality_session = uncertain_ratio >= POOR_QUALITY_SESSION_RATIO
    avg_conf = float(df["fatigue_signal_confidence"].mean())

    final = (
        "high"
        if high_risk_dur > 5 or len(closure_events) >= 2
        else ("medium" if high_risk_dur > 2 or df["perclos_30s"].mean() > 0.15 or len(closure_events) >= 1 else "low")
    )
    if poor_quality_session:
        final = final + "_review"

    summary = {
        "total_frames": int(len(df)),
        "total_duration": float(total_duration),
        "avg_drowsy": float(df["a1_prob_drowsy_sm"].mean()),
        "max_drowsy": float(df["a1_prob_drowsy_sm"].max()),
        "avg_eye": float(df["a2_prob_eye_closed_sm"].mean()),
        "perclos": float(df["perclos_30s"].mean()),
        "eye_closure_burden": float(df["eye_closed_binary"].mean()),
        "closure_count": int(len(closure_events)),
        "blink_count": int(blink_count),
        "blink_freq_per_min": float(blink_count / total_duration * 60) if total_duration > 0 else 0.0,
        "avg_blink_duration": avg_blink_dur,
        "avg_ear": float(df["ear"].dropna().mean()) if df["ear"].notna().any() else None,
        "yawn_sup": float(df["a1_prob_yawn_sm"].mean()),
        "nod_sup": float(df["a1_prob_nod_sm"].mean()),
        "hr_dur": high_risk_dur,
        "caut_dur": caution_dur,
        "uncertain_ratio": uncertain_ratio,
        "poor_quality_session": poor_quality_session,
        "mean_confidence": avg_conf,
        "final": final,
    }
    return df, summary, all_events


def temporal_b(df, fps, sw=3, offroad_threshold=OFFROAD_THRESHOLD):
    if df.empty:
        return df, {}, []
    df = df.copy()
    fps = max(1, int(fps))
    td = len(df) / fps if fps > 0 else 0

    for c in ["offroad_prob", "confidence", "entropy", "uncertainty_score"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["offroad_prob"] = df["offroad_prob"].clip(0, 1)
    df["confidence"] = df["confidence"].clip(0, 1)
    df["uncertainty_score"] = df["uncertainty_score"].clip(0, 1)

    if sw > 1:
        for c in ["offroad_prob", "confidence", "entropy", "uncertainty_score"]:
            df[f"{c}_sm"] = df[c].rolling(sw, min_periods=1).mean()
    else:
        for c in ["offroad_prob", "confidence", "entropy", "uncertainty_score"]:
            df[f"{c}_sm"] = df[c]

    norm_entropy = df["entropy_sm"] / np.log(len(ZONE_CLASSES))
    df["attn_uncertain"] = (
        (df["confidence_sm"] < MIN_ALERT_CONF)
        | (norm_entropy > 0.75)
        | (df["uncertainty_score_sm"] > 0.65)
        | (df["entropy_sm"] > MAX_UNCERTAIN_ENTROPY)
    ).astype(int)
    df["review_flag"] = np.where(df["attn_uncertain"] == 1, "needs_review", "clear")

    enter_t = offroad_threshold
    exit_t = max(0.05, offroad_threshold - OFFROAD_EXIT_HYSTERESIS)
    offroad_pred = []
    active = False
    for prob, uncertain in zip(df["offroad_prob_sm"].values, df["attn_uncertain"].values):
        if int(uncertain) == 1:
            active = False
            offroad_pred.append(0)
            continue
        if not active and prob >= enter_t:
            active = True
        elif active and prob <= exit_t:
            active = False
        offroad_pred.append(1 if active else 0)
    df["offroad_pred"] = offroad_pred

    ob = df["offroad_pred"].astype(int).values
    hb = (df["risk_group_pred"] == "HighRisk").astype(int).values
    mb = df["zone_pred"].isin(["Left Mirror", "Right Mirror", "Rearview"]).astype(int).values

    def max_streak(arr):
        mx, c = 0, 0
        for v in arr:
            c = c + 1 if v else 0
            mx = max(mx, c)
        return mx

    def transitions(arr):
        return sum(1 for i in range(1, len(arr)) if arr[i] == 1 and arr[i - 1] == 0)

    mos = max_streak(ob) / fps if fps > 0 else 0
    mhs = max_streak(hb) / fps if fps > 0 else 0
    ot = transitions(ob)
    ht = transitions(hb)
    mt = transitions(mb)

    md_list = []
    cc = 0
    for v in mb:
        if v:
            cc += 1
        else:
            if cc > 0:
                md_list.append(cc / fps)
            cc = 0
    if cc > 0:
        md_list.append(cc / fps)

    offroad_events = []
    for evt in _streak_events(ob, fps, 2.0, "offroad_glance"):
        mask = (df["timestamp_seconds"] >= evt["start"]) & (df["timestamp_seconds"] <= evt["end"])
        conf = float(df.loc[mask, "confidence_sm"].mean()) if mask.any() else 0.0
        sev, sev_score = _severity_bucket(max(df.loc[mask, "offroad_prob_sm"].mean(), 0.55) if mask.any() else 0.55, evt["dur"])
        offroad_events.append(
            {
                **evt,
                "dominant_zone": df.loc[mask, "zone_pred"].mode().iloc[0] if mask.any() else "",
                "risk_group": "distraction",
                "severity": sev,
                "severity_score": sev_score,
                "confidence": conf,
                "alert_sent": conf >= MIN_ALERT_CONF and sev in ["high", "critical"],
                "alert_type": _alert_type_from_severity(sev),
                "event_confirmation_status": "pending_review" if conf < MIN_ALERT_CONF else "auto_confirmed",
            }
        )

    repeated_distraction = []
    if td > 0 and ot >= 3:
        repeated_distraction.append(
            {
                "type": "repeated_distraction",
                "start": 0.0,
                "end": float(df["timestamp_seconds"].iloc[-1]),
                "dur": float(df["timestamp_seconds"].iloc[-1]),
                "dominant_zone": df[df["offroad_pred"] == 1]["zone_pred"].mode().iloc[0] if (df["offroad_pred"] == 1).any() else "",
                "risk_group": "distraction",
                "severity": "high" if ot >= 5 else "medium",
                "severity_score": 0.80 if ot >= 5 else 0.60,
                "confidence": float(df["confidence_sm"].mean()),
                "alert_sent": ot >= 4,
                "alert_type": "dashboard_escalation" if ot >= 4 else "dashboard_warning",
                "event_confirmation_status": "auto_confirmed",
            }
        )

    mirror_events = []
    for evt in _streak_events(mb, fps, 0.0, "mirror_glance"):
        mask = (df["timestamp_seconds"] >= evt["start"]) & (df["timestamp_seconds"] <= evt["end"])
        mirror_events.append(
            {
                **evt,
                "dominant_zone": df.loc[mask, "zone_pred"].mode().iloc[0] if mask.any() else "",
                "risk_group": "monitoring",
                "severity": "low",
                "severity_score": 0.20,
                "confidence": float(df.loc[mask, "confidence_sm"].mean()) if mask.any() else 0.0,
                "alert_sent": False,
                "alert_type": "none",
                "event_confirmation_status": "auto_confirmed",
            }
        )

    uncertain_events = []
    for evt in _streak_events(df["attn_uncertain"].values, fps, 2.0, "uncertain_segment_review"):
        uncertain_events.append(
            {
                **evt,
                "dominant_zone": "",
                "risk_group": "review",
                "severity": "medium",
                "severity_score": 0.50,
                "confidence": 0.20,
                "alert_sent": False,
                "alert_type": "review_queue",
                "event_confirmation_status": "pending_review",
            }
        )

    events = offroad_events + repeated_distraction + mirror_events + uncertain_events
    kpis = {
        "nf": int(len(df)),
        "td": float(td),
        "or": float(np.mean(ob)) if len(ob) else 0.0,
        "mos": float(mos),
        "oepm": float(ot / td * 60) if td > 0 else 0.0,
        "hr": float(np.mean(hb)) if len(hb) else 0.0,
        "mhs": float(mhs),
        "hepm": float(ht / td * 60) if td > 0 else 0.0,
        "mfpm": float(mt / td * 60) if td > 0 else 0.0,
        "amd": float(np.mean(md_list)) if md_list else 0.0,
        "sfr": float((df["zone_pred"] == "Forward").mean()),
        "mc": float(df["confidence_sm"].mean()),
        "me": float(df["entropy_sm"].mean()),
        "uncertain_ratio": float(df["attn_uncertain"].mean()),
        "poor_quality_session": float(df["attn_uncertain"].mean()) >= POOR_QUALITY_SESSION_RATIO,
        "repeated_distraction_events": int(len(repeated_distraction)),
    }
    return df, kpis, events


def build_timeline(df_a, df_b):
    if df_a.empty and df_b.empty:
        return pd.DataFrame()
    if not df_a.empty and not df_b.empty:
        left = df_a[["frame_id", "timestamp_seconds", "frame_risk", "eye_closed_binary", "uncertain_frame"]].copy()
        right = df_b[["frame_index", "timestamp_seconds", "offroad_pred", "risk_group_pred", "attn_uncertain"]].copy()
        left["ts_key"] = left["timestamp_seconds"].round(3)
        right["ts_key"] = right["timestamp_seconds"].round(3)
        timeline = pd.merge(left, right, on="ts_key", how="outer", suffixes=("_a", "_b")).sort_values("ts_key")
        timeline["timestamp_seconds"] = timeline["timestamp_seconds_a"].fillna(timeline["timestamp_seconds_b"])
        timeline["frame_id"] = timeline["frame_id"].fillna(timeline["frame_index"])
    elif not df_a.empty:
        timeline = df_a[["frame_id", "timestamp_seconds", "frame_risk", "eye_closed_binary", "uncertain_frame"]].copy()
        timeline["offroad_pred"] = 0
        timeline["risk_group_pred"] = "Unknown"
        timeline["attn_uncertain"] = 0
    else:
        timeline = df_b[["frame_index", "timestamp_seconds", "offroad_pred", "risk_group_pred", "attn_uncertain"]].copy()
        timeline = timeline.rename(columns={"frame_index": "frame_id"})
        timeline["frame_risk"] = 0.0
        timeline["eye_closed_binary"] = 0
        timeline["uncertain_frame"] = 0

    keep_cols = [
        "frame_id",
        "timestamp_seconds",
        "frame_risk",
        "eye_closed_binary",
        "uncertain_frame",
        "offroad_pred",
        "risk_group_pred",
        "attn_uncertain",
    ]
    timeline = timeline[[c for c in keep_cols if c in timeline.columns]].copy()

    def label_row(r):
        if int(r.get("uncertain_frame", 0)) == 1 or int(r.get("attn_uncertain", 0)) == 1:
            return "needs_review"
        if int(r.get("eye_closed_binary", 0)) == 1:
            return "prolonged_eye_closure" if float(r.get("frame_risk", 0)) >= 0.6 else "mild_fatigue"
        if int(r.get("offroad_pred", 0)) == 1:
            return "offroad_glance"
        if str(r.get("risk_group_pred", "")) == "HighRisk":
            return "repeated_distraction" if float(r.get("frame_risk", 0)) >= 0.4 else "offroad_glance"
        if float(r.get("frame_risk", 0)) >= 0.7:
            return "high_fatigue"
        if float(r.get("frame_risk", 0)) >= 0.4:
            return "mild_fatigue"
        return "normal_driving"

    timeline["timeline_label"] = timeline.apply(label_row, axis=1)
    timeline["overall_risk_score"] = (
        0.55 * pd.to_numeric(timeline.get("frame_risk", 0), errors="coerce").fillna(0)
        + 0.45 * pd.to_numeric(timeline.get("offroad_pred", 0), errors="coerce").fillna(0)
    ).clip(0, 1)
    return timeline.sort_values("timestamp_seconds").reset_index(drop=True)


def summarize_timeline(timeline_df, fps):
    if timeline_df.empty:
        return []
    labels = timeline_df["timeline_label"].tolist()
    times = timeline_df["timestamp_seconds"].tolist()
    rows = []
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            rows.append(
                {
                    "timeline_state": current,
                    "start_ts": times[start],
                    "end_ts": times[i - 1],
                    "duration_seconds": max(0.0, times[i - 1] - times[start] + (1 / fps if fps else 0)),
                }
            )
            start = i
            current = labels[i]
    rows.append(
        {
            "timeline_state": current,
            "start_ts": times[start],
            "end_ts": times[-1],
            "duration_seconds": max(0.0, times[-1] - times[start] + (1 / fps if fps else 0)),
        }
    )
    return rows


def build_driver_scorecard(driver_id, trip_id, sid, a_summary, b_summary):
    fatigue_score = float(
        np.clip(
            0.45 * a_summary.get("avg_drowsy", 0)
            + 0.25 * a_summary.get("perclos", 0)
            + 0.20 * min(a_summary.get("closure_count", 0) / 3.0, 1.0)
            + 0.10 * min(a_summary.get("hr_dur", 0) / 10.0, 1.0),
            0,
            1,
        )
    )
    distraction_score = float(
        np.clip(
            0.45 * b_summary.get("or", 0)
            + 0.25 * min(b_summary.get("oepm", 0) / 6.0, 1.0)
            + 0.20 * (1 - b_summary.get("sfr", 0))
            + 0.10 * min(b_summary.get("repeated_distraction_events", 0) / 2.0, 1.0),
            0,
            1,
        )
    )
    mean_conf = float(np.clip((a_summary.get("mean_confidence", 0.0) + b_summary.get("mc", 0.0)) / 2.0, 0.0, 1.0))
    uncertainty_penalty = 0.08 * max(0.0, 0.6 - mean_conf)
    combined = float(np.clip(0.55 * fatigue_score + 0.45 * distraction_score + uncertainty_penalty, 0, 1))
    if combined <= 0.20:
        rating = "A"
    elif combined <= 0.40:
        rating = "B"
    elif combined <= 0.60:
        rating = "C"
    elif combined <= 0.80:
        rating = "D"
    else:
        rating = "E"
    return {
        "driver_id": driver_id,
        "trip_id": trip_id,
        "session_id": sid,
        "fatigue_risk_score": fatigue_score,
        "distraction_risk_score": distraction_score,
        "combined_driver_safety_score": combined,
        "average_offroad_ratio": float(b_summary.get("or", 0)),
        "prolonged_eye_closure_count": int(a_summary.get("closure_count", 0)),
        "mirror_usage_pattern": "healthy" if b_summary.get("mfpm", 0) >= 1 else "low_mirror_usage",
        "safe_forward_ratio": float(b_summary.get("sfr", 0)),
        "highrisk_events_per_driving_hour": float(b_summary.get("hepm", 0)),
        "mean_model_confidence": mean_conf,
        "driver_rating": rating,
        "created_at_client": utc_now_iso(),
    }


def _explain_unified(a_summary, b_summary, scorecard):
    parts = []
    if a_summary.get("perclos", 0) > 0.15:
        parts.append(f"PERCLOS was elevated at {a_summary['perclos']:.1%}")
    if a_summary.get("closure_count", 0) > 0:
        parts.append(f"{a_summary['closure_count']} prolonged eye closure event(s) were found")
    if b_summary.get("or", 0) > 0.20:
        parts.append(f"off-road ratio reached {b_summary['or']:.1%}")
    if b_summary.get("repeated_distraction_events", 0) > 0:
        parts.append("repeated distraction pattern was detected")
    if a_summary.get("poor_quality_session") or b_summary.get("poor_quality_session"):
        parts.append("some frames were uncertain and need review")
    if not parts:
        return "Driver session looks stable. No major fatigue or distraction pattern sustained over time."
    return "Combined safety summary: " + "; ".join(parts) + f". Overall driver rating: {scorecard['driver_rating']}."


def extract_frames(video_bytes: bytes, fps_target: int, file_ext: str, max_frames: int):
    if not CV2_AVAILABLE:
        return [], {}
    with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
        tmp.write(video_bytes)
        tp = tmp.name
    cap = cv2.VideoCapture(tp)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = raw_total / vfps if vfps > 0 else 0.0
    interval = max(1, int(vfps / max(1, fps_target)))
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append((idx / vfps if vfps > 0 else 0.0, frame))
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    os.unlink(tp)
    meta = {
        "video_fps": float(vfps),
        "raw_frame_count": int(raw_total),
        "duration_seconds": float(duration),
        "sample_interval_frames": int(interval),
        "sampled_frame_count": int(len(frames)),
        "resolution": f"{width}x{height}" if width and height else "unknown",
    }
    return frames, meta


def analyze_video(
    frames,
    cfg_a,
    fps_b,
    bsw,
    offroad_threshold,
    driver_id,
    trip_id,
    source_name,
    sim_a_mode,
    sim_b_mode,
    progress_cb=None,
):
    m1, m2, bm = load_a1(), load_a2(), load_b()
    rows_a, rows_b, thumbs = [], [], []
    n = len(frames)
    tick = max(1, n // 100) if n else 1

    for i, (ts, frame_bgr) in enumerate(frames):
        meta = detect_face_eyes(frame_bgr)
        rng = _make_rng(source_name, i)

        a_source = "model"
        if sim_a_mode:
            ra = sim_a(rng)
            a_source = "simulation"
        else:
            o1 = infer_a1(m1, meta["face"])
            o2 = infer_a2(m2, meta["eye_strip"])
            if o1 or o2:
                fallback = sim_a(rng)
                ra = {**fallback, **(o1 or {}), **(o2 or {})}
                if o1 is None or o2 is None:
                    a_source = "hybrid_model_plus_simulation"
            else:
                ra = sim_a(rng)
                a_source = "simulation_fallback"

        b_source = "model"
        if sim_b_mode:
            rb = sim_b(rng, offroad_threshold)
            b_source = "simulation"
        else:
            rb = infer_b(bm, frame_bgr)
            if rb is None:
                rb = sim_b(rng, offroad_threshold)
                b_source = "simulation_fallback"

        row_a = {
            "session_id": None,
            "driver_id": driver_id,
            "trip_id": trip_id,
            "source_type": "video",
            "source_name": source_name,
            "frame_id": i,
            "timestamp_seconds": ts,
            "a1_prob_drowsy": ra.get("a1_prob_drowsy"),
            "a1_prob_yawn": ra.get("a1_prob_yawn"),
            "a1_prob_nod": ra.get("a1_prob_nod"),
            "a2_prob_eye_closed": ra.get("a2_prob_eye_closed"),
            "a2_eye_openness_score": ra.get("a2_eye_openness_score"),
            "a1_confidence": ra.get("a1_confidence"),
            "a2_confidence": ra.get("a2_confidence"),
            "ear": meta.get("ear"),
            "left_ear": meta.get("left_ear"),
            "right_ear": meta.get("right_ear"),
            "head_pitch": meta.get("head_pitch"),
            "head_yaw": meta.get("head_yaw"),
            "head_roll": meta.get("head_roll"),
            "quality_score": meta.get("quality_score"),
            "face_confidence": meta.get("face_confidence"),
            "brightness_score": meta.get("brightness_score"),
            "blur_score": meta.get("blur_score"),
            "preprocess_method": meta.get("preprocess_method"),
            "inference_mode": a_source,
        }
        rows_a.append(row_a)

        row_b = {
            "session_id": None,
            "driver_id": driver_id,
            "trip_id": trip_id,
            "input_type": "video",
            "source_file_name": source_name,
            "frame_index": i,
            "timestamp_seconds": ts,
            "inference_mode": b_source,
            "face_detect_method": meta.get("preprocess_method", "unknown"),
            **rb,
        }
        rows_b.append(row_b)

        if i % max(1, n // 12) == 0:
            thumbs.append((i, ts, frame_bgr.copy(), meta, ra, rb))

        if progress_cb is not None and (i % tick == 0 or i == n - 1):
            progress_cb(i + 1, n)

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    df_a, a_summary, a_events = temporal_a(df_a, cfg_a)
    df_b, b_summary, b_events = temporal_b(df_b, fps_b, bsw, offroad_threshold)
    return df_a, a_summary, a_events, df_b, b_summary, b_events, thumbs


def build_unified_frame_predictions(df_a, df_b, session_id, driver_id, trip_id, source_file_name):
    if df_a.empty and df_b.empty:
        return pd.DataFrame()
    if not df_a.empty and not df_b.empty:
        merged = pd.merge(
            df_a,
            df_b,
            left_on="frame_id",
            right_on="frame_index",
            how="outer",
            suffixes=("_a", "_b"),
        )
        merged["timestamp_seconds"] = pd.to_numeric(merged["timestamp_seconds_a"], errors="coerce").fillna(
            pd.to_numeric(merged["timestamp_seconds_b"], errors="coerce")
        )
    elif not df_a.empty:
        merged = df_a.copy()
        merged["frame_index"] = merged["frame_id"]
        merged["offroad_prob"] = 0.0
        merged["offroad_pred"] = 0
        merged["zone_pred"] = "Unknown"
        merged["risk_group_pred"] = "Unknown"
        merged["attn_uncertain"] = 0
        merged["confidence"] = 0.0
        merged["uncertainty_score"] = 1.0
        merged["inference_mode_b"] = "missing"
    else:
        merged = df_b.copy()
        merged["frame_id"] = merged["frame_index"]
        merged["frame_risk"] = 0.0
        merged["risk_level"] = "unknown"
        merged["uncertain_frame"] = 1
        merged["inference_mode_a"] = "missing"

    for col in ["frame_risk", "offroad_prob", "confidence", "uncertainty_score"]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).clip(0, 1)
    if "offroad_pred" not in merged.columns:
        merged["offroad_pred"] = (merged["offroad_prob"] >= OFFROAD_THRESHOLD).astype(int)
    if "risk_level" not in merged.columns:
        merged["risk_level"] = "unknown"
    if "uncertain_frame" not in merged.columns:
        merged["uncertain_frame"] = 1
    if "attn_uncertain" not in merged.columns:
        merged["attn_uncertain"] = 1
    if "zone_pred" not in merged.columns:
        merged["zone_pred"] = "Unknown"
    if "risk_group_pred" not in merged.columns:
        merged["risk_group_pred"] = "Unknown"

    merged["overall_risk_score"] = (
        0.50 * merged["frame_risk"]
        + 0.35 * merged["offroad_prob"]
        + 0.15 * (1.0 - merged["confidence"])
    ).clip(0, 1)
    merged["needs_review"] = ((merged["uncertain_frame"].astype(int) == 1) | (merged["attn_uncertain"].astype(int) == 1)).astype(int)
    merged["overall_risk_label"] = np.select(
        [
            merged["needs_review"] == 1,
            merged["overall_risk_score"] >= 0.70,
            merged["overall_risk_score"] >= 0.40,
        ],
        ["needs_review", "high", "medium"],
        default="low",
    )
    merged["session_id"] = session_id
    merged["driver_id"] = driver_id
    merged["trip_id"] = trip_id
    merged["source_file_name"] = source_file_name
    merged["created_at_client"] = utc_now_iso()

    keep_cols = [
        "session_id",
        "driver_id",
        "trip_id",
        "source_file_name",
        "frame_id",
        "frame_index",
        "timestamp_seconds",
        "overall_risk_score",
        "overall_risk_label",
        "needs_review",
        "frame_risk",
        "risk_level",
        "offroad_prob",
        "offroad_pred",
        "zone_pred",
        "risk_group_pred",
        "confidence",
        "uncertainty_score",
        "uncertain_frame",
        "attn_uncertain",
        "a1_prob_drowsy",
        "a2_prob_eye_closed",
        "perclos_30s",
        "blink_rate_per_min",
        "quality_score",
        "face_confidence",
        "preprocess_method",
        "inference_mode_a",
        "inference_mode_b",
        "created_at_client",
    ] + [f"zone_prob_{z.lower().replace(' ', '_')}" for z in ZONE_CLASSES]

    for c in ["inference_mode_a", "inference_mode_b"]:
        if c not in merged.columns:
            merged[c] = "unknown"
    out = merged[[c for c in keep_cols if c in merged.columns]].copy()
    out["frame_id"] = pd.to_numeric(out.get("frame_id", np.nan), errors="coerce").fillna(
        pd.to_numeric(out.get("frame_index", np.nan), errors="coerce")
    )
    out = out.sort_values("frame_id").reset_index(drop=True)
    return out


def build_zone_transition_summary(df_b, session_id, driver_id, trip_id):
    if df_b.empty:
        return pd.DataFrame()
    x = df_b[["frame_index", "timestamp_seconds", "zone_pred", "risk_group_pred", "confidence"]].copy()
    x = x.sort_values("frame_index").reset_index(drop=True)
    transition_rows = []
    prev_zone = None
    seg_start_ts = None
    seg_start_idx = None
    for _, row in x.iterrows():
        zone = row["zone_pred"]
        ts = float(row["timestamp_seconds"])
        idx = int(row["frame_index"])
        if prev_zone is None:
            prev_zone = zone
            seg_start_ts = ts
            seg_start_idx = idx
            continue
        if zone != prev_zone:
            transition_rows.append(
                {
                    "session_id": session_id,
                    "driver_id": driver_id,
                    "trip_id": trip_id,
                    "from_zone": prev_zone,
                    "to_zone": zone,
                    "start_ts": seg_start_ts,
                    "end_ts": ts,
                    "duration_seconds": max(0.0, ts - seg_start_ts),
                    "start_frame_index": seg_start_idx,
                    "end_frame_index": idx,
                    "created_at_client": utc_now_iso(),
                }
            )
            prev_zone = zone
            seg_start_ts = ts
            seg_start_idx = idx
    return pd.DataFrame(transition_rows)


def build_session_quality_summary(df_a, df_b, session_id, driver_id, trip_id, source_file_name):
    qa = float(pd.to_numeric(df_a.get("quality_score", 0.0), errors="coerce").fillna(0.0).mean()) if not df_a.empty else 0.0
    fa = float(pd.to_numeric(df_a.get("face_confidence", 0.0), errors="coerce").fillna(0.0).mean()) if not df_a.empty else 0.0
    aa = float(pd.to_numeric(df_a.get("uncertain_frame", 1), errors="coerce").fillna(1.0).mean()) if not df_a.empty else 1.0
    ab = float(pd.to_numeric(df_b.get("attn_uncertain", 1), errors="coerce").fillna(1.0).mean()) if not df_b.empty else 1.0
    p_method = "unknown"
    if not df_a.empty and "preprocess_method" in df_a.columns and df_a["preprocess_method"].notna().any():
        p_method = str(df_a["preprocess_method"].mode().iloc[0])
    return pd.DataFrame(
        [
            {
                "session_id": session_id,
                "driver_id": driver_id,
                "trip_id": trip_id,
                "source_file_name": source_file_name,
                "avg_frame_quality": qa,
                "avg_face_confidence": fa,
                "fatigue_uncertain_ratio": aa,
                "attn_uncertain_ratio": ab,
                "dominant_preprocess_method": p_method,
                "overall_quality_score": float(np.clip(0.4 * qa + 0.3 * fa + 0.3 * (1 - max(aa, ab)), 0, 1)),
                "created_at_client": utc_now_iso(),
            }
        ]
    )


def render_frame_explorer(frames, unified_frame_df):
    st.subheader("Frame-Level Prediction Explorer")
    if unified_frame_df.empty:
        st.caption("No frame-level rows available.")
        return

    frame_df = unified_frame_df.copy()
    frame_df["frame_id"] = pd.to_numeric(frame_df["frame_id"], errors="coerce").fillna(-1).astype(int)
    frame_df = frame_df[frame_df["frame_id"] >= 0].sort_values("frame_id").reset_index(drop=True)

    risk_opts = ["needs_review", "high", "medium", "low"]
    selected_risks = st.multiselect("Filter by overall risk label", options=risk_opts, default=risk_opts)
    filtered = frame_df[frame_df["overall_risk_label"].isin(selected_risks)].copy()
    if filtered.empty:
        st.caption("No frames match the selected filters.")
        return

    row_idx = st.slider("Pick frame from filtered set", min_value=0, max_value=len(filtered) - 1, value=0, step=1)
    r = filtered.iloc[row_idx]
    fid = int(r["frame_id"])

    frame_map = {i: frm for i, (_, frm) in enumerate(frames)}
    if fid in frame_map:
        st.image(cv2.cvtColor(frame_map[fid], cv2.COLOR_BGR2RGB), caption=f"Frame {fid} | {float(r['timestamp_seconds']):.2f}s", use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Risk", f"{float(r['overall_risk_score']):.2f}")
    c2.metric("Fatigue Risk", f"{float(r['frame_risk']):.2f}")
    c3.metric("Off-road Prob", f"{float(r['offroad_prob']):.2f}")
    c4.metric("Model Confidence", f"{float(r['confidence']):.2f}")

    st.caption(
        f"Label: {r['overall_risk_label']} | Zone: {r.get('zone_pred', 'Unknown')} | Risk Group: {r.get('risk_group_pred', 'Unknown')} | Needs Review: {int(r.get('needs_review', 0))}"
    )

    zone_prob_cols = [f"zone_prob_{z.lower().replace(' ', '_')}" for z in ZONE_CLASSES if f"zone_prob_{z.lower().replace(' ', '_')}" in filtered.columns]
    if zone_prob_cols:
        zr = pd.DataFrame({"zone": [c.replace("zone_prob_", "").replace("_", " ").title() for c in zone_prob_cols], "prob": [float(r[c]) for c in zone_prob_cols]})
        zr = zr.sort_values("prob", ascending=False).head(3)
        st.bar_chart(zr, x="zone", y="prob")

    view_cols = [
        "frame_id",
        "timestamp_seconds",
        "overall_risk_score",
        "overall_risk_label",
        "needs_review",
        "frame_risk",
        "risk_level",
        "offroad_prob",
        "offroad_pred",
        "zone_pred",
        "risk_group_pred",
        "confidence",
        "uncertainty_score",
        "uncertain_frame",
        "attn_uncertain",
    ]
    st.dataframe(filtered[[c for c in view_cols if c in filtered.columns]], hide_index=True, use_container_width=True, height=320)

    st.subheader("Frame Gallery (Prediction + Frame)")
    page_size = st.select_slider("Frames per page", options=[4, 6, 8, 12], value=6)
    total_pages = max(1, int(np.ceil(len(filtered) / page_size)))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(start + page_size, len(filtered))
    page_df = filtered.iloc[start:end]

    cols = st.columns(3)
    for i, (_, row) in enumerate(page_df.iterrows()):
        col = cols[i % 3]
        with col:
            frame_id = int(row["frame_id"])
            if frame_id in frame_map:
                col.image(cv2.cvtColor(frame_map[frame_id], cv2.COLOR_BGR2RGB), use_container_width=True)
            col.caption(
                f"F{frame_id} | {float(row['timestamp_seconds']):.1f}s | {row['overall_risk_label']} | zone={row.get('zone_pred', 'Unknown')}"
            )




# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def home_page():
    st.title(":material/directions_car: Driver Safety Analytics Platform")
    st.markdown(
        "Unified fatigue + attentiveness analytics — EfficientNet-B0 ensemble, "
        "YuNet face detection, Snowflake-ready outputs."
    )

    a1_models  = load_a1()
    a2_models  = load_a2()
    b_ensemble = load_b()

    c1, c2, c3 = st.columns(3)
    with c1:
        n1 = len(a1_models) if a1_models else 0
        n2 = len(a2_models) if a2_models else 0
        st.metric("A1 Ensemble (drowsy/yawn/nod)",
                  f"{n1} fold(s) loaded" if n1 else "Not loaded")
        st.metric("A2 Ensemble (eye closed)",
                  f"{n2} fold(s) loaded" if n2 else "Not loaded")
    with c2:
        nb = len(b_ensemble) if b_ensemble else 0
        st.metric("B Ensemble (gaze zone)",
                  f"{nb} fold(s) loaded" if nb else "Not loaded")
        yunet = _get_yunet_detector()
        st.metric("Face Detector",
                  "YuNet ONNX" if yunet is not None else "Haar Cascade (fallback)")
    with c3:
        for pkg, avail in [("PyTorch", TORCH_AVAILABLE),
                            ("OpenCV",  CV2_AVAILABLE),
                            ("Pillow",  PIL_AVAILABLE)]:
            st.write(f"- {'OK' if avail else 'MISSING'}: {pkg}")
        st.caption(
            f"Snowflake session: {'Connected' if get_session() else 'Not connected'}")

    st.divider()
    with st.container(border=True):
        st.subheader("What's New in v6.0.0")
        st.markdown(f"""
- **EfficientNet-B0 backbone** for A1 and A2 — replacing v3's ResNet18 (critical fix)
- **Ensemble inference**: A1 (folds 1,2,4), A2 (folds 1,3,4), B (folds 1-6) — averaging across folds reduces variance
- **YuNet face detector** (`cv2.FaceDetectorYN`) — no MediaPipe dependency, Snowflake-compatible
- **Recalibrated thresholds**: off-road threshold {OFFROAD_THRESHOLD} (was 0.35), zone T={ZONE_TEMPERATURE:.3f} (was 2.60)
- **A2 input fixed**: 224x224 square crop (was 64x224 — mismatch with training)
- **A1 yawn/nod now work**: Focal Loss training fixed the collapsed minority heads from v3
        """)

def unified_page():
    st.title(":material/security: Unified Driver Safety Analyzer")
    with st.sidebar:
        st.subheader("Driver & Trip Context")
        driver_id = st.text_input("Driver ID", value="DRV_001")
        trip_id = st.text_input("Trip ID", value="TRIP_001")
        trip_start_ts = st.text_input("Trip Start TS", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        trip_end_ts = st.text_input("Trip End TS", value="")

        st.subheader("Processing")
        afps = st.number_input("Sampling FPS", 1, 30, DEFAULT_A["fps"])
        max_frames = st.number_input("Max sampled frames", 100, MAX_EXTRACTION_FRAMES, 1500, 50)

        st.subheader("Fatigue Settings")
        wd = st.slider("A1 Drowsy weight", 0.0, 1.0, DEFAULT_A["w_drowsy"], 0.05)
        we = st.slider("A2 Eye Closed weight", 0.0, 1.0, DEFAULT_A["w_eye_closed"], 0.05)
        wn = st.slider("A1 Nod weight", 0.0, 1.0, DEFAULT_A["w_nod"], 0.05)
        wy = st.slider("A1 Yawn weight", 0.0, 1.0, DEFAULT_A["w_yawn"], 0.05)
        wp = st.slider("PERCLOS weight", 0.0, 1.0, DEFAULT_A["w_perclos"], 0.05)
        wb = st.slider("Blink weight", 0.0, 1.0, DEFAULT_A["w_blink"], 0.05)
        te = st.slider("Eye closed threshold", 0.0, 1.0, DEFAULT_A["t_eye"], 0.05)
        tc = st.slider("Caution threshold", 0.0, 1.0, DEFAULT_A["t_caution"], 0.05)
        ta = st.slider("Alert threshold", 0.0, 1.0, DEFAULT_A["t_alert"], 0.05)
        pcf = st.number_input("Closure min frames", 2, 30, DEFAULT_A["closure_frames"])
        ear_closed = st.slider("EAR closed threshold", 0.10, 0.40, DEFAULT_A["ear_closed"], 0.01)

        st.subheader("Attentiveness Settings")
        bsw = st.number_input("Smoothing window", 1, 10, 3)
        bort = st.slider("Off-road threshold", 0.0, 1.0, OFFROAD_THRESHOLD, 0.05)

    cfg_a = {
        "w_drowsy": wd,
        "w_eye_closed": we,
        "w_nod": wn,
        "w_yawn": wy,
        "w_perclos": wp,
        "w_blink": wb,
        "t_eye": te,
        "t_caution": tc,
        "t_alert": ta,
        "closure_frames": pcf,
        "fps": afps,
        "win_s": DEFAULT_A["win_s"],
        "win_m": DEFAULT_A["win_m"],
        "win_l": DEFAULT_A["win_l"],
        "ear_closed": ear_closed,
    }

    sim_a_mode = load_a1() is None or load_a2() is None
    sim_b_mode = load_b() is None
    if sim_a_mode or sim_b_mode:
        st.info(
            ":material/science: Some models are not loaded. The app will use deterministic simulation fallback where needed."
        )

    uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])
    if not uploaded:
        st.caption("Upload a file to begin analysis.")
        return

    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    is_video = ext in ("mp4", "avi", "mov", "mkv")
    sid = str(uuid.uuid4())[:12]

    if not is_video:
        st.warning("Unified scorecard and timeline are intended for video. For image, only preview is shown.")
        if PIL_AVAILABLE:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_container_width=True)
        else:
            st.error("Pillow is not available to preview image.")
        return

    video_bytes = uploaded.getvalue()
    st.subheader("Uploaded Video")
    st.video(video_bytes)

    frames, video_meta = extract_frames(video_bytes, int(afps), ext, int(max_frames))
    if not frames:
        st.error("Could not extract frames.")
        return

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Original FPS", f"{video_meta.get('video_fps', 0):.2f}")
    v2.metric("Raw Frames", int(video_meta.get("raw_frame_count", 0)))
    v3.metric("Sampled Frames", int(video_meta.get("sampled_frame_count", 0)))
    v4.metric("Resolution", video_meta.get("resolution", "unknown"))
    st.caption(
        f"Duration: {video_meta.get('duration_seconds', 0):.1f}s | Sampling interval: every {video_meta.get('sample_interval_frames', 1)} frame(s)"
    )

    prog = st.progress(0, "Running frame inference...")

    def progress_cb(done, total):
        prog.progress(min(done / max(total, 1), 1.0), f"Running frame inference... {done}/{total}")

    df_a, a_summary, a_events, df_b, b_summary, b_events, thumbs = analyze_video(
        frames,
        cfg_a,
        afps,
        bsw,
        bort,
        driver_id,
        trip_id,
        uploaded.name,
        sim_a_mode,
        sim_b_mode,
        progress_cb=progress_cb,
    )
    prog.empty()

    df_a["session_id"] = sid
    df_b["session_id"] = sid
    session_start_ts = trip_start_ts if trip_start_ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_end_ts = trip_end_ts if trip_end_ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    timeline_df = build_timeline(df_a, df_b)
    timeline_df["session_id"] = sid
    timeline_df["driver_id"] = driver_id
    timeline_df["trip_id"] = trip_id
    timeline_rows = summarize_timeline(timeline_df, afps)
    scorecard = build_driver_scorecard(driver_id, trip_id, sid, a_summary, b_summary)
    combined_summary_text = _explain_unified(a_summary, b_summary, scorecard)
    unified_frame_df = build_unified_frame_predictions(df_a, df_b, sid, driver_id, trip_id, uploaded.name)
    zone_transition_df = build_zone_transition_summary(df_b, sid, driver_id, trip_id)
    quality_summary_df = build_session_quality_summary(df_a, df_b, sid, driver_id, trip_id, uploaded.name)

    combined_driver_safety_score = scorecard["combined_driver_safety_score"]
    escalation_level = min(3, len([e for e in a_events + b_events if e.get("alert_sent")]))

    unified_summary = {
        "session_id": sid,
        "driver_id": driver_id,
        "trip_id": trip_id,
        "session_start_ts": session_start_ts,
        "session_end_ts": session_end_ts,
        "duration_seconds": float(a_summary.get("total_duration", 0)),
        "source_file_name": uploaded.name,
        "model_version_a1": MODEL_VERSION_A1,
        "model_version_a2": MODEL_VERSION_A2,
        "model_version_b": MODEL_VERSION_B,
        "app_version": APP_VERSION,
        "fatigue_score": scorecard["fatigue_risk_score"],
        "distraction_score": scorecard["distraction_risk_score"],
        "combined_driver_safety_score": combined_driver_safety_score,
        "driver_rating": scorecard["driver_rating"],
        "summary_text": combined_summary_text,
        "escalation_level": escalation_level,
        "trip_start_ts": trip_start_ts,
        "trip_end_ts": trip_end_ts,
        "mean_model_confidence": scorecard["mean_model_confidence"],
        "created_at_client": utc_now_iso(),
    }

    fdf_a_cols = [
        "session_id",
        "driver_id",
        "trip_id",
        "source_type",
        "source_name",
        "frame_id",
        "timestamp_seconds",
        "a1_prob_drowsy",
        "a1_prob_yawn",
        "a1_prob_nod",
        "a2_prob_eye_closed",
        "a2_eye_openness_score",
        "a1_prob_drowsy_sm",
        "a2_prob_eye_closed_sm",
        "frame_risk",
        "risk_level",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "ear",
        "left_ear",
        "right_ear",
        "quality_score",
        "brightness_score",
        "blur_score",
        "face_confidence",
        "preprocess_method",
        "fatigue_signal_confidence",
        "uncertain_frame",
        "review_flag",
        "eye_closed_binary",
        "blink_binary",
        "perclos_30s",
        "blink_rate_per_min",
        "inference_mode",
    ]
    write_to_snowflake(df_a[[c for c in fdf_a_cols if c in df_a.columns]].copy(), "MODULE_A_FRAME_PREDICTIONS")

    fdf_b_cols = [
        "session_id",
        "driver_id",
        "trip_id",
        "input_type",
        "source_file_name",
        "frame_index",
        "timestamp_seconds",
        "zone_pred",
        "risk_group_pred",
        "zone_top2",
        "offroad_prob",
        "offroad_prob_sm",
        "offroad_pred",
        "confidence",
        "confidence_sm",
        "entropy",
        "entropy_sm",
        "margin",
        "uncertainty_score",
        "uncertainty_score_sm",
        "attn_uncertain",
        "review_flag",
        "inference_mode",
    ] + [f"zone_prob_{z.lower().replace(' ', '_')}" for z in ZONE_CLASSES]
    write_to_snowflake(df_b[[c for c in fdf_b_cols if c in df_b.columns]].copy(), "MODULE_B_FRAME_PREDICTIONS")

    a_event_rows = []
    for ev in a_events:
        a_event_rows.append(
            {
                "event_id": str(uuid.uuid4())[:16],
                "session_id": sid,
                "driver_id": driver_id,
                "trip_id": trip_id,
                "event_type": ev["type"],
                "event_start_ts": ev["start"],
                "event_end_ts": ev["end"],
                "duration_seconds": ev["dur"],
                "severity": ev["severity"],
                "severity_score": ev["severity_score"],
                "confidence": ev["confidence"],
                "dominant_zone": "",
                "risk_group": ev.get("risk_group", "fatigue"),
                "alert_sent": ev["alert_sent"],
                "alert_type": ev["alert_type"],
                "event_confirmation_status": ev["event_confirmation_status"],
                "explanation": f"{ev['type']} for {ev['dur']:.2f}s",
                "created_at_client": utc_now_iso(),
            }
        )
    if a_event_rows:
        write_to_snowflake(pd.DataFrame(a_event_rows), "MODULE_A_EVENTS")

    b_event_rows = []
    for ev in b_events:
        b_event_rows.append(
            {
                "event_id": str(uuid.uuid4())[:16],
                "session_id": sid,
                "driver_id": driver_id,
                "trip_id": trip_id,
                "event_type": ev["type"],
                "event_start_ts": ev["start"],
                "event_end_ts": ev["end"],
                "duration_seconds": ev["dur"],
                "severity": ev["severity"],
                "severity_score": ev["severity_score"],
                "confidence": ev["confidence"],
                "dominant_zone": ev.get("dominant_zone", ""),
                "risk_group": ev.get("risk_group", "distraction"),
                "alert_sent": ev["alert_sent"],
                "alert_type": ev["alert_type"],
                "event_confirmation_status": ev["event_confirmation_status"],
                "created_at_client": utc_now_iso(),
            }
        )
    if b_event_rows:
        write_to_snowflake(pd.DataFrame(b_event_rows), "MODULE_B_EVENTS")

    a_summary_row = {
        "session_id": sid,
        "driver_id": driver_id,
        "trip_id": trip_id,
        "source_name": uploaded.name,
        "source_type": "video",
        "session_start_ts": session_start_ts,
        "session_end_ts": session_end_ts,
        "duration_seconds": a_summary["total_duration"],
        "total_frames_processed": a_summary["total_frames"],
        "total_duration_seconds": a_summary["total_duration"],
        "avg_a1_prob_drowsy": a_summary["avg_drowsy"],
        "max_a1_prob_drowsy": a_summary["max_drowsy"],
        "avg_a2_prob_eye_closed": a_summary["avg_eye"],
        "eye_closure_burden": a_summary["eye_closure_burden"],
        "perclos": a_summary["perclos"],
        "blink_count": a_summary["blink_count"],
        "blink_freq_per_min": a_summary["blink_freq_per_min"],
        "avg_blink_duration_seconds": a_summary["avg_blink_duration"],
        "avg_ear": a_summary["avg_ear"],
        "prolonged_closure_count": a_summary["closure_count"],
        "yawn_support_score": a_summary["yawn_sup"],
        "nod_support_score": a_summary["nod_sup"],
        "total_high_risk_duration": a_summary["hr_dur"],
        "total_caution_duration": a_summary["caut_dur"],
        "mean_confidence": a_summary["mean_confidence"],
        "uncertain_ratio": a_summary["uncertain_ratio"],
        "poor_quality_session": a_summary["poor_quality_session"],
        "final_session_risk": a_summary["final"],
        "model_version_a1": MODEL_VERSION_A1,
        "model_version_a2": MODEL_VERSION_A2,
        "app_version": APP_VERSION,
        "summary_text": combined_summary_text,
        "created_at_client": utc_now_iso(),
    }
    write_to_snowflake(pd.DataFrame([a_summary_row]), "MODULE_A_SESSION_SUMMARY")

    b_summary_row = {
        "session_id": sid,
        "driver_id": driver_id,
        "trip_id": trip_id,
        "source_file_name": uploaded.name,
        "session_start_ts": session_start_ts,
        "session_end_ts": session_end_ts,
        "duration_seconds": b_summary["td"],
        "total_frames": b_summary["nf"],
        "total_duration_seconds": b_summary["td"],
        "offroad_ratio": b_summary["or"],
        "max_offroad_streak_seconds": b_summary["mos"],
        "offroad_events_per_min": b_summary["oepm"],
        "highrisk_ratio": b_summary["hr"],
        "max_highrisk_streak_seconds": b_summary["mhs"],
        "highrisk_events_per_min": b_summary["hepm"],
        "mirror_glance_frequency_per_min": b_summary["mfpm"],
        "avg_mirror_glance_duration_seconds": b_summary["amd"],
        "safe_forward_ratio": b_summary["sfr"],
        "mean_confidence": b_summary["mc"],
        "mean_entropy": b_summary["me"],
        "uncertain_ratio": b_summary["uncertain_ratio"],
        "poor_quality_session": b_summary["poor_quality_session"],
        "repeated_distraction_events": b_summary["repeated_distraction_events"],
        "model_version_b": MODEL_VERSION_B,
        "app_version": APP_VERSION,
        "created_at_client": utc_now_iso(),
    }
    write_to_snowflake(pd.DataFrame([b_summary_row]), "MODULE_B_SESSION_SUMMARY")

    write_to_snowflake(pd.DataFrame([unified_summary]), "UNIFIED_DRIVER_SESSION_SUMMARY")
    write_to_snowflake(pd.DataFrame([scorecard]), "DRIVER_SCORECARDS")
    write_to_snowflake(unified_frame_df, "UNIFIED_FRAME_PREDICTIONS")
    if not zone_transition_df.empty:
        write_to_snowflake(zone_transition_df, "MODULE_B_ZONE_TRANSITIONS")
    write_to_snowflake(quality_summary_df, "SESSION_DATA_QUALITY_SUMMARY")

    review_queue_df = unified_frame_df[unified_frame_df["needs_review"] == 1].copy() if not unified_frame_df.empty else pd.DataFrame()
    if not review_queue_df.empty:
        write_to_snowflake(review_queue_df, "FRAME_REVIEW_QUEUE")

    if timeline_rows:
        timeline_upload = pd.DataFrame(timeline_rows)
        timeline_upload["session_id"] = sid
        timeline_upload["driver_id"] = driver_id
        timeline_upload["trip_id"] = trip_id
        timeline_upload["created_at_client"] = utc_now_iso()
        write_to_snowflake(timeline_upload, "DRIVER_TIMELINE")

    risk_color = "green" if combined_driver_safety_score < 0.35 else ("orange" if combined_driver_safety_score < 0.65 else "red")
    st.markdown(f"### Unified Session Score: :{risk_color}[{combined_driver_safety_score:.2f}] | Driver Rating: **{scorecard['driver_rating']}**")
    st.caption(combined_summary_text)

    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
        ["Overview", "Fatigue", "Attentiveness", "Timeline", "Frame Explorer", "Events", "Scorecard", "Data Exports"]
    )
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fatigue Score", f"{scorecard['fatigue_risk_score']:.2f}")
        c2.metric("Distraction Score", f"{scorecard['distraction_risk_score']:.2f}")
        c3.metric("Combined Safety Score", f"{scorecard['combined_driver_safety_score']:.2f}")
        c4.metric("Escalation Level", escalation_level)
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("PERCLOS", f"{a_summary['perclos']:.1%}")
        c6.metric("Blinks/min", f"{a_summary['blink_freq_per_min']:.1f}")
        c7.metric("Off-road Ratio", f"{b_summary['or']:.1%}")
        c8.metric("Safe Forward Ratio", f"{b_summary['sfr']:.1%}")
        if thumbs:
            last = thumbs[-1]
            st.image(cv2.cvtColor(last[2], cv2.COLOR_BGR2RGB), caption=f"Sample Frame {last[0]} @ {last[1]:.1f}s", use_container_width=True)
        st.caption("Interpretation: lower combined score is safer. Review sessions with high uncertainty ratio before operational action.")

    with t2:
        fatigue_trend = df_a[
            ["timestamp_seconds", "frame_risk", "a1_prob_drowsy_sm", "a2_prob_eye_closed_sm", "perclos_30s", "blink_rate_per_min"]
        ].rename(
            columns={
                "timestamp_seconds": "Time",
                "frame_risk": "Fatigue Risk",
                "a1_prob_drowsy_sm": "Drowsy (Smoothed)",
                "a2_prob_eye_closed_sm": "Eye Closed (Smoothed)",
                "perclos_30s": "PERCLOS",
                "blink_rate_per_min": "Blink/min",
            }
        )
        st.line_chart(fatigue_trend, x="Time")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg EAR", f"{a_summary['avg_ear']:.3f}" if a_summary["avg_ear"] is not None else "N/A")
        c2.metric("Blink Count", a_summary["blink_count"])
        c3.metric("Avg Blink Duration", f"{a_summary['avg_blink_duration']:.2f}s")
        c4.metric("Uncertain Ratio", f"{a_summary['uncertain_ratio']:.1%}")
        st.caption(f"Current thresholds -> Caution: {tc:.2f}, Alert: {ta:.2f}, Eye Closed: {te:.2f}, EAR Closed: {ear_closed:.2f}")

    with t3:
        att_trend = df_b[["timestamp_seconds", "offroad_prob_sm", "confidence_sm", "uncertainty_score_sm"]].rename(
            columns={
                "timestamp_seconds": "Time",
                "offroad_prob_sm": "Off-road Prob (Smoothed)",
                "confidence_sm": "Confidence (Smoothed)",
                "uncertainty_score_sm": "Uncertainty (Smoothed)",
            }
        )
        st.line_chart(att_trend, x="Time")
        zd = df_b["zone_pred"].value_counts().reset_index()
        zd.columns = ["Zone", "Count"]
        st.bar_chart(zd, x="Zone", y="Count")
        st.caption("Higher confidence and lower uncertainty indicate more reliable attentiveness predictions.")

    with t4:
        if timeline_df.empty:
            st.caption("No timeline available.")
        else:
            tshow = timeline_df[["timestamp_seconds", "timeline_label"]].copy()
            mapv = {
                "normal_driving": 0,
                "mild_fatigue": 1,
                "high_fatigue": 2,
                "offroad_glance": 3,
                "prolonged_eye_closure": 4,
                "repeated_distraction": 5,
                "needs_review": 6,
            }
            tshow["state_value"] = tshow["timeline_label"].map(mapv).fillna(0)
            st.bar_chart(tshow.rename(columns={"timestamp_seconds": "Time", "state_value": "Timeline State"}), x="Time", y="Timeline State")
            st.dataframe(pd.DataFrame(timeline_rows), hide_index=True, use_container_width=True)

    with t5:
        render_frame_explorer(frames, unified_frame_df)

    with t6:
        all_events_df = pd.DataFrame(a_event_rows + b_event_rows)
        if not all_events_df.empty:
            st.dataframe(all_events_df, hide_index=True, use_container_width=True)
            sev_counts = all_events_df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            st.bar_chart(sev_counts, x="Severity", y="Count")
            alert_count = int(all_events_df["alert_sent"].fillna(False).astype(int).sum())
            repeat_count = int((all_events_df["event_type"] == "repeated_distraction").sum())
            closure_count = int((all_events_df["event_type"] == "prolonged_eye_closure").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Actionable Alerts", alert_count)
            c2.metric("Repeated Distraction Events", repeat_count)
            c3.metric("Closure Events", closure_count)
        else:
            st.caption("No events detected.")

    with t7:
        st.dataframe(pd.DataFrame([scorecard]), hide_index=True, use_container_width=True)
        st.download_button("Download Scorecard CSV", pd.DataFrame([scorecard]).to_csv(index=False), f"driver_scorecard_{sid}.csv", "text/csv")
        st.dataframe(quality_summary_df, hide_index=True, use_container_width=True)

    with t8:
        st.subheader("Snowflake Export Preview")
        c1, c2, c3 = st.columns(3)
        c1.metric("A Frame Rows", len(df_a))
        c2.metric("B Frame Rows", len(df_b))
        c3.metric("Unified Frame Rows", len(unified_frame_df))
        if not review_queue_df.empty:
            st.metric("Review Queue Frames", len(review_queue_df))

        combined_export = pd.merge(
            df_a,
            df_b,
            left_on="timestamp_seconds",
            right_on="timestamp_seconds",
            how="outer",
            suffixes=("_A", "_B"),
        )
        st.download_button(
            "Download Combined Frame CSV",
            combined_export.to_csv(index=False),
            f"unified_driver_safety_{sid}.csv",
            "text/csv",
        )
        st.download_button(
            "Download Unified Frame Predictions CSV",
            unified_frame_df.to_csv(index=False),
            f"unified_frame_predictions_{sid}.csv",
            "text/csv",
        )
        if not review_queue_df.empty:
            st.download_button(
                "Download Review Queue CSV",
                review_queue_df.to_csv(index=False),
                f"review_queue_{sid}.csv",
                "text/csv",
            )




def analytics_page():
    st.title(":material/analytics: Historical Analytics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Unified Sessions", "Driver Scorecards", "Timeline States", "Frame Risk Analytics", "Review Queue"]
    )
    with tab1:
        df = fetch_recent("UNIFIED_DRIVER_SESSION_SUMMARY", 100)
        if df.empty:
            st.caption("No unified sessions yet.")
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)
            if {"DRIVER_ID", "COMBINED_DRIVER_SAFETY_SCORE"}.issubset(df.columns):
                score_df = df[["DRIVER_ID", "COMBINED_DRIVER_SAFETY_SCORE"]].groupby("DRIVER_ID", as_index=False).mean()
                st.bar_chart(score_df, x="DRIVER_ID", y="COMBINED_DRIVER_SAFETY_SCORE")
            if {"DRIVER_RATING"}.issubset(df.columns):
                rc = df["DRIVER_RATING"].value_counts().reset_index()
                rc.columns = ["Driver Rating", "Count"]
                st.bar_chart(rc, x="Driver Rating", y="Count")

    with tab2:
        sc = fetch_recent("DRIVER_SCORECARDS", 100)
        if sc.empty:
            st.caption("No driver scorecards yet.")
        else:
            st.dataframe(sc, hide_index=True, use_container_width=True)
            if {"DRIVER_ID", "DRIVER_RATING"}.issubset(sc.columns):
                rating_counts = sc["DRIVER_RATING"].value_counts().reset_index()
                rating_counts.columns = ["Driver Rating", "Count"]
                st.bar_chart(rating_counts, x="Driver Rating", y="Count")

    with tab3:
        tl = fetch_recent("DRIVER_TIMELINE", 500)
        if tl.empty:
            st.caption("No timeline rows yet.")
        else:
            st.dataframe(tl, hide_index=True, use_container_width=True)
            if {"TIMELINE_STATE", "DURATION_SECONDS"}.issubset(tl.columns):
                state_dur = tl.groupby("TIMELINE_STATE", as_index=False)["DURATION_SECONDS"].sum()
                st.bar_chart(state_dur, x="TIMELINE_STATE", y="DURATION_SECONDS")

    with tab4:
        uf = fetch_recent("UNIFIED_FRAME_PREDICTIONS", 3000)
        if uf.empty:
            st.caption("No unified frame predictions yet.")
        else:
            st.dataframe(uf.head(300), hide_index=True, use_container_width=True)
            if {"OVERALL_RISK_LABEL"}.issubset(uf.columns):
                rc = uf["OVERALL_RISK_LABEL"].value_counts().reset_index()
                rc.columns = ["Risk Label", "Count"]
                st.bar_chart(rc, x="Risk Label", y="Count")
            if {"ZONE_PRED"}.issubset(uf.columns):
                zc = uf["ZONE_PRED"].value_counts().reset_index()
                zc.columns = ["Zone", "Count"]
                st.bar_chart(zc, x="Zone", y="Count")

    with tab5:
        rq = fetch_recent("FRAME_REVIEW_QUEUE", 1000)
        if rq.empty:
            st.caption("No review queue frames yet.")
        else:
            st.dataframe(rq, hide_index=True, use_container_width=True)
            if {"DRIVER_ID"}.issubset(rq.columns):
                by_driver = rq["DRIVER_ID"].value_counts().reset_index()
                by_driver.columns = ["Driver ID", "Review Frames"]
                st.bar_chart(by_driver, x="Driver ID", y="Review Frames")




pages = [
    st.Page(home_page, title="Home", icon=":material/home:"),
    st.Page(unified_page, title="Unified Safety", icon=":material/security:"),
    st.Page(analytics_page, title="Historical Analytics", icon=":material/analytics:"),
]
pg = st.navigation(pages)
pg.run()
