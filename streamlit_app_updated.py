import streamlit as st
import numpy as np
import pandas as pd
import uuid
import os
import tempfile
from datetime import datetime

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

DATABASE = "DEMO_DB"
SCHEMA = "PUBLIC"
APP_VERSION = "2.0.0"
MODEL_VERSION_A1 = "a1_fold1.pt"
MODEL_VERSION_A2 = "a2_fold1.pt"
MODEL_VERSION_B = "best_lisa_fold5_calibrated.pt"

_MODEL_DIR = tempfile.mkdtemp()

ZONE_CLASSES = [
    "Forward", "Lap", "Left Mirror", "Radio",
    "Rearview", "Right Mirror", "Shoulder", "Speedometer",
]
RISK_GROUPS = {
    "Safe": ["Forward"],
    "LowRisk": ["Left Mirror", "Right Mirror", "Rearview"],
    "HighRisk": ["Lap", "Radio", "Speedometer", "Shoulder"],
}
OFFROAD_THRESHOLD = 0.35
ZONE_TEMPERATURE = 2.600605
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
MIN_ALERT_CONF = 0.45
MAX_UNCERTAIN_ENTROPY = 1.55
MIN_FRAME_QUALITY = 0.35
EAR_CLOSED_THRESHOLD = 0.21
POOR_QUALITY_SESSION_RATIO = 0.35

MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MP_FACE_POINTS = [1, 33, 263, 61, 291, 152]

DEFAULT_A = {
    "w_drowsy": 0.30,
    "w_eye_closed": 0.30,
    "w_nod": 0.15,
    "w_yawn": 0.10,
    "w_perclos": 0.10,
    "w_blink": 0.05,
    "t_eye": 0.30,
    "t_caution": 0.40,
    "t_alert": 0.70,
    "closure_frames": 5,
    "fps": 5,
    "win_s": 10,
    "win_m": 30,
    "win_l": 60,
    "ear_closed": EAR_CLOSED_THRESHOLD,
}


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
        s.file.get(f"@{DATABASE}.{SCHEMA}.DRIVER_SAFETY_MODELS/{filename}", _MODEL_DIR)
        if os.path.exists(path):
            return path
        gz_path = path + ".gz"
        if os.path.exists(gz_path):
            import gzip
            import shutil
            with gzip.open(gz_path, "rb") as f_in, open(path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
            return path
        return None
    except Exception:
        return None


def write_to_snowflake(df: pd.DataFrame, table_name: str):
    s = get_session()
    if s is None or df.empty:
        return False
    try:
        upload = df.copy()
        upload.columns = [c.upper() for c in upload.columns]
        s.write_pandas(upload, table_name, database=DATABASE, schema=SCHEMA, overwrite=False)
        return True
    except Exception as e:
        st.toast(f"DB write warning for {table_name}: {e}", icon=":material/warning:")
        return False


def query_sf(sql: str):
    s = get_session()
    if s is None:
        return pd.DataFrame()
    try:
        return s.sql(sql).to_pandas()
    except Exception:
        return pd.DataFrame()


def get_risk_group(zone: str):
    for group, zones in RISK_GROUPS.items():
        if zone in zones:
            return group
    return "Unknown"


@st.cache_resource
def get_face_mesh():
    if not MP_AVAILABLE:
        return None
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


if TORCH_AVAILABLE:
    class _A1Model(nn.Module):
        def __init__(self):
            super().__init__()
            import torchvision.models as m
            bb = m.resnet18(weights=None)
            nf = bb.fc.in_features
            bb.fc = nn.Identity()
            self.bb = bb
            self.h_d = nn.Linear(nf, 1)
            self.h_y = nn.Linear(nf, 1)
            self.h_n = nn.Linear(nf, 1)

        def forward(self, x):
            f = self.bb(x)
            return {
                "drowsy": torch.sigmoid(self.h_d(f)).squeeze(-1),
                "yawn": torch.sigmoid(self.h_y(f)).squeeze(-1),
                "nod": torch.sigmoid(self.h_n(f)).squeeze(-1),
            }

    class _A2Model(nn.Module):
        def __init__(self):
            super().__init__()
            import torchvision.models as m
            bb = m.resnet18(weights=None)
            nf = bb.fc.in_features
            bb.fc = nn.Identity()
            self.bb = bb
            self.h_e = nn.Linear(nf, 1)

        def forward(self, x):
            f = self.bb(x)
            return {"eye_closed": torch.sigmoid(self.h_e(f)).squeeze(-1)}

    class _BModel(nn.Module):
        def __init__(self, bb, nf):
            super().__init__()
            self.bb = bb
            self.zone_head = nn.Linear(nf, 8)
            self.offroad_head = nn.Linear(nf, 1)

        def forward(self, x):
            f = self.bb(x)
            return {"zone_logits": self.zone_head(f), "offroad_logit": self.offroad_head(f)}


@st.cache_resource
def load_a1():
    if not TORCH_AVAILABLE:
        return None
    p = _download_model(MODEL_VERSION_A1)
    if not p or not os.path.exists(p):
        return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, nn.Module):
            ckpt.eval()
            return ckpt
        mdl = _A1Model()
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        mdl.load_state_dict(sd, strict=False)
        mdl.eval()
        return mdl
    except Exception:
        return None


@st.cache_resource
def load_a2():
    if not TORCH_AVAILABLE:
        return None
    p = _download_model(MODEL_VERSION_A2)
    if not p or not os.path.exists(p):
        return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, nn.Module):
            ckpt.eval()
            return ckpt
        mdl = _A2Model()
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        mdl.load_state_dict(sd, strict=False)
        mdl.eval()
        return mdl
    except Exception:
        return None


@st.cache_resource
def load_b():
    if not TORCH_AVAILABLE:
        return None
    p = _download_model(MODEL_VERSION_B)
    if not p or not os.path.exists(p):
        return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, nn.Module):
            ckpt.eval()
            return ckpt
        try:
            import timm
            bb = timm.create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=0)
            nf = bb.num_features
        except ImportError:
            import torchvision.models as tm
            bb = tm.efficientnet_b0(weights=None)
            nf = bb.classifier[1].in_features
            bb.classifier = nn.Identity()
        mdl = _BModel(bb, nf)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        mdl.load_state_dict(sd, strict=False)
        mdl.eval()
        return mdl
    except Exception:
        return None


def _prep_tensor(img_pil, size):
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return t(img_pil).unsqueeze(0)


def _normalized_variance_laplacian(gray_img):
    if gray_img is None or gray_img.size == 0:
        return 0.0
    blur = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    return float(np.clip(blur / 300.0, 0.0, 1.0))


def _brightness_score(gray_img):
    if gray_img is None or gray_img.size == 0:
        return 0.0
    mean = float(np.mean(gray_img))
    # best around mid-range brightness
    return float(np.clip(1.0 - abs(mean - 128.0) / 128.0, 0.0, 1.0))


def _safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _landmark_xy(landmarks, idx, w, h):
    pt = landmarks[idx]
    return np.array([pt.x * w, pt.y * h], dtype=np.float32)


def _eye_aspect_ratio(points):
    p1, p2, p3, p4, p5, p6 = points
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4) + 1e-6
    return float((vertical1 + vertical2) / (2.0 * horizontal))


def _crop_from_points(frame_bgr, points, pad_ratio=0.25):
    xs = points[:, 0]
    ys = points[:, 1]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w = x2 - x1
    h = y2 - y1
    pad_x = max(4, w * pad_ratio)
    pad_y = max(4, h * pad_ratio)
    return _safe_crop(frame_bgr, x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)


def _estimate_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    image_points = np.array([
        _landmark_xy(landmarks, 1, w, h),
        _landmark_xy(landmarks, 152, w, h),
        _landmark_xy(landmarks, 33, w, h),
        _landmark_xy(landmarks, 263, w, h),
        _landmark_xy(landmarks, 61, w, h),
        _landmark_xy(landmarks, 291, w, h),
    ], dtype=np.float64)
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ], dtype=np.float64)
    focal = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    try:
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None, None
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        proj = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        pitch, yaw, roll = float(euler[0]), float(euler[1]), float(euler[2])
        return pitch, yaw, roll
    except Exception:
        return None, None, None


def detect_face_eyes(frame_bgr):
    if not CV2_AVAILABLE:
        return {
            "face": None,
            "eye_strip": None,
            "left_eye": None,
            "right_eye": None,
            "face_detected": False,
            "preprocess_method": "none",
            "quality_score": 0.0,
            "face_confidence": 0.0,
            "ear": None,
            "left_ear": None,
            "right_ear": None,
            "head_pitch": None,
            "head_yaw": None,
            "head_roll": None,
        }

    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = _brightness_score(gray)
    blur = _normalized_variance_laplacian(gray)
    mesh = get_face_mesh()

    if MP_AVAILABLE and mesh is not None:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pts_all = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
                x1, y1 = pts_all.min(axis=0)
                x2, y2 = pts_all.max(axis=0)
                face = _safe_crop(frame_bgr, x1 - 12, y1 - 12, x2 + 12, y2 + 12)

                left_pts = np.array([_landmark_xy(lm, idx, w, h) for idx in MP_LEFT_EYE], dtype=np.float32)
                right_pts = np.array([_landmark_xy(lm, idx, w, h) for idx in MP_RIGHT_EYE], dtype=np.float32)
                left_eye = _crop_from_points(frame_bgr, left_pts, pad_ratio=0.45)
                right_eye = _crop_from_points(frame_bgr, right_pts, pad_ratio=0.45)

                eye_strip = None
                if left_eye is not None and right_eye is not None:
                    max_h = max(left_eye.shape[0], right_eye.shape[0])
                    if left_eye.shape[0] != max_h:
                        left_eye = cv2.resize(left_eye, (left_eye.shape[1], max_h))
                    if right_eye.shape[0] != max_h:
                        right_eye = cv2.resize(right_eye, (right_eye.shape[1], max_h))
                    eye_strip = cv2.hconcat([left_eye, right_eye])
                elif left_eye is not None:
                    eye_strip = left_eye
                elif right_eye is not None:
                    eye_strip = right_eye

                left_ear = _eye_aspect_ratio(left_pts)
                right_ear = _eye_aspect_ratio(right_pts)
                ear = float((left_ear + right_ear) / 2.0)
                head_pitch, head_yaw, head_roll = _estimate_head_pose(lm, frame_bgr.shape)
                face_area = max(1.0, (x2 - x1) * (y2 - y1))
                face_area_ratio = float(np.clip(face_area / (w * h), 0.0, 1.0))
                quality = float(np.clip(0.40 * blur + 0.30 * brightness + 0.30 * min(face_area_ratio / 0.2, 1.0), 0.0, 1.0))
                return {
                    "face": face,
                    "eye_strip": eye_strip,
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "face_detected": face is not None,
                    "preprocess_method": "mediapipe",
                    "quality_score": quality,
                    "face_confidence": min(1.0, 0.5 + quality / 2.0),
                    "brightness_score": brightness,
                    "blur_score": blur,
                    "ear": ear,
                    "left_ear": left_ear,
                    "right_ear": right_ear,
                    "head_pitch": head_pitch,
                    "head_yaw": head_yaw,
                    "head_roll": head_roll,
                }
        except Exception:
            pass

    # fallback
    cascade_path = None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    except Exception:
        pass
    if cascade_path and os.path.exists(cascade_path):
        fc = cv2.CascadeClassifier(cascade_path)
        faces = fc.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            face = frame_bgr[y:y + fh, x:x + fw]
            eye_y1 = int(fh * 0.15)
            eye_y2 = int(fh * 0.45)
            strip = face[eye_y1:eye_y2, :]
            area_ratio = float(np.clip((fw * fh) / (w * h), 0.0, 1.0))
            quality = float(np.clip(0.45 * blur + 0.30 * brightness + 0.25 * min(area_ratio / 0.2, 1.0), 0.0, 1.0))
            return {
                "face": face,
                "eye_strip": strip,
                "left_eye": None,
                "right_eye": None,
                "face_detected": True,
                "preprocess_method": "haar",
                "quality_score": quality,
                "face_confidence": min(0.8, 0.35 + quality / 2.0),
                "brightness_score": brightness,
                "blur_score": blur,
                "ear": None,
                "left_ear": None,
                "right_ear": None,
                "head_pitch": None,
                "head_yaw": None,
                "head_roll": None,
            }

    # last fallback center crop
    cx, cy = w // 2, h // 2
    fw, fh = int(w * 0.5), int(h * 0.6)
    x1 = max(0, cx - fw // 2)
    y1 = max(0, cy - fh // 2)
    face = frame_bgr[y1:y1 + fh, x1:x1 + fw]
    ey1 = int(fh * 0.15)
    ey2 = int(fh * 0.45)
    strip = face[ey1:ey2, :]
    quality = float(np.clip(0.35 * blur + 0.35 * brightness + 0.15, 0.0, 0.55))
    return {
        "face": face,
        "eye_strip": strip,
        "left_eye": None,
        "right_eye": None,
        "face_detected": False,
        "preprocess_method": "center_crop",
        "quality_score": quality,
        "face_confidence": 0.20,
        "brightness_score": brightness,
        "blur_score": blur,
        "ear": None,
        "left_ear": None,
        "right_ear": None,
        "head_pitch": None,
        "head_yaw": None,
        "head_roll": None,
    }


def _certainty_from_prob(prob):
    if prob is None:
        return 0.0
    return float(np.clip(abs(prob - 0.5) * 2.0, 0.0, 1.0))


def infer_a1(mdl, face):
    if mdl is None or face is None or not PIL_AVAILABLE or not CV2_AVAILABLE:
        return None
    try:
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            o = mdl(_prep_tensor(img, 224))
        drowsy = float(o["drowsy"].item())
        yawn = float(o["yawn"].item())
        nod = float(o["nod"].item())
        return {
            "a1_prob_drowsy": drowsy,
            "a1_prob_yawn": yawn,
            "a1_prob_nod": nod,
            "a1_confidence": float(np.mean([_certainty_from_prob(drowsy), _certainty_from_prob(yawn), _certainty_from_prob(nod)])),
        }
    except Exception:
        return None


def infer_a2(mdl, strip):
    if mdl is None or strip is None or not PIL_AVAILABLE or not CV2_AVAILABLE:
        return None
    try:
        img = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
        t = transforms.Compose([
            transforms.Resize((64, 224)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
        with torch.no_grad():
            o = mdl(t(img).unsqueeze(0))
        eye_closed = float(o["eye_closed"].item())
        return {
            "a2_prob_eye_closed": eye_closed,
            "a2_eye_openness_score": float(1.0 - eye_closed),
            "a2_confidence": _certainty_from_prob(eye_closed),
        }
    except Exception:
        return None


def infer_b(mdl, frame, offroad_threshold=OFFROAD_THRESHOLD):
    if mdl is None or not PIL_AVAILABLE or not CV2_AVAILABLE:
        return None
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            o = mdl(_prep_tensor(img, IMG_SIZE))
        zl = o["zone_logits"][0] / ZONE_TEMPERATURE
        zp = torch.softmax(zl, dim=0).detach().cpu().numpy()
        sorted_probs = np.sort(zp)[::-1]
        zi = int(np.argmax(zp))
        op = float(torch.sigmoid(o["offroad_logit"][0]).item())
        entropy = float(-np.sum(zp * np.log(zp + 1e-8)))
        confidence = float(zp.max())
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else confidence
        uncertainty = float(np.clip((entropy / np.log(len(ZONE_CLASSES))) * 0.6 + (1.0 - confidence) * 0.4, 0.0, 1.0))
        return {
            "zone_pred": ZONE_CLASSES[zi],
            "risk_group_pred": get_risk_group(ZONE_CLASSES[zi]),
            "offroad_prob": op,
            "offroad_pred": 1 if op >= offroad_threshold else 0,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "uncertainty_score": uncertainty,
            **{f"zone_prob_{z.lower().replace(' ', '_')}": float(zp[i]) for i, z in enumerate(ZONE_CLASSES)},
        }
    except Exception:
        return None


def sim_a():
    eye_closed = float(np.random.beta(2, 5))
    drowsy = float(np.random.beta(2, 5))
    yawn = float(np.random.beta(1.5, 8))
    nod = float(np.random.beta(1.5, 6))
    return {
        "a1_prob_drowsy": drowsy,
        "a1_prob_yawn": yawn,
        "a1_prob_nod": nod,
        "a2_prob_eye_closed": eye_closed,
        "a2_eye_openness_score": float(1.0 - eye_closed),
        "a1_confidence": float(np.random.uniform(0.45, 0.85)),
        "a2_confidence": float(np.random.uniform(0.45, 0.85)),
    }


def sim_b(offroad_threshold=OFFROAD_THRESHOLD):
    pr = np.random.dirichlet([5, 1, 1.5, 0.8, 1.5, 1.5, 0.5, 0.8])
    zi = int(np.argmax(pr))
    zone_name = ZONE_CLASSES[zi]
    op = float(np.clip(1 - pr[0] + np.random.normal(0, 0.05), 0, 1))
    confidence = float(pr.max())
    entropy = float(-np.sum(pr * np.log(pr + 1e-8)))
    sorted_probs = np.sort(pr)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else confidence
    uncertainty = float(np.clip((entropy / np.log(len(ZONE_CLASSES))) * 0.6 + (1.0 - confidence) * 0.4, 0.0, 1.0))
    return {
        "zone_pred": zone_name,
        "risk_group_pred": get_risk_group(zone_name),
        "offroad_prob": op,
        "offroad_pred": 1 if op >= offroad_threshold else 0,
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "uncertainty_score": uncertainty,
        **{f"zone_prob_{z.lower().replace(' ', '_')}": float(pr[i]) for i, z in enumerate(ZONE_CLASSES)},
    }


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
    fps = cfg["fps"]
    df = df.copy()
    df["a2_prob_eye_closed"] = df["a2_prob_eye_closed"].fillna(0)
    df["a1_confidence"] = df["a1_confidence"].fillna(0)
    df["a2_confidence"] = df["a2_confidence"].fillna(0)
    df["ear"] = df["ear"].fillna(np.nan)
    df["preprocess_confidence"] = (df["quality_score"].fillna(0) * 0.6 + df["face_confidence"].fillna(0) * 0.4)
    df["eye_closed_binary"] = ((df["a2_prob_eye_closed"] >= cfg["t_eye"]) | (df["ear"].notna() & (df["ear"] <= cfg["ear_closed"]))).astype(int)

    blink_events, blink_binary = _derive_blinks(df, fps)
    df["blink_binary"] = blink_binary

    # rolling fatigue features
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
        df.loc[s:e - 1, "avg_blink_duration_s"] = evt["dur"]
    df["avg_blink_duration_s"] = df["avg_blink_duration_s"].rolling(win_m, min_periods=1).mean()

    for wn, ws in [("s", cfg["win_s"]), ("m", cfg["win_m"]), ("l", cfg["win_l"] )]:
        w = max(1, int(ws * fps))
        df[f"rd_{wn}"] = df["a1_prob_drowsy"].rolling(w, min_periods=1).mean()
        df[f"rdmx_{wn}"] = df["a1_prob_drowsy"].rolling(w, min_periods=1).max()
        df[f"re_{wn}"] = df["a2_prob_eye_closed"].rolling(w, min_periods=1).mean()
        df[f"ry_{wn}"] = df["a1_prob_yawn"].rolling(w, min_periods=1).mean()
        df[f"rn_{wn}"] = df["a1_prob_nod"].rolling(w, min_periods=1).mean()

    df["fatigue_signal_confidence"] = (
        0.35 * df["preprocess_confidence"]
        + 0.35 * df["a1_confidence"]
        + 0.30 * df["a2_confidence"]
    ).clip(0, 1)
    df["uncertain_frame"] = (
        (df["fatigue_signal_confidence"] < MIN_ALERT_CONF)
        | (df["quality_score"] < MIN_FRAME_QUALITY)
    ).astype(int)
    df["review_flag"] = np.where(df["uncertain_frame"] == 1, "needs_review", "clear")

    normalized_blink = np.clip(df["blink_rate_per_min"] / 20.0, 0.0, 1.0)
    df["fatigue_score_raw"] = (
        cfg["w_drowsy"] * df["a1_prob_drowsy"]
        + cfg["w_eye_closed"] * df["a2_prob_eye_closed"]
        + cfg["w_nod"] * df["a1_prob_nod"]
        + cfg["w_yawn"] * df["a1_prob_yawn"]
        + cfg["w_perclos"] * df["perclos_30s"]
        + cfg["w_blink"] * normalized_blink
    ).clip(0, 1)
    df["frame_risk"] = np.where(df["uncertain_frame"] == 1, df["fatigue_score_raw"] * 0.7, df["fatigue_score_raw"]).clip(0, 1)
    df["risk_level"] = df["frame_risk"].apply(lambda r: "high" if r >= cfg["t_alert"] else ("medium" if r >= cfg["t_caution"] else "low"))
    df.loc[df["uncertain_frame"] == 1, "risk_level"] = df.loc[df["uncertain_frame"] == 1, "risk_level"] + "_uncertain"

    closure_events = []
    for evt in _streak_events(df["eye_closed_binary"].values, fps, cfg["closure_frames"] / fps, "prolonged_eye_closure"):
        mask = (df["timestamp_seconds"] >= evt["start"]) & (df["timestamp_seconds"] <= evt["end"])
        conf = float(df.loc[mask, "fatigue_signal_confidence"].mean()) if mask.any() else 0.0
        sev, sev_score = _severity_bucket(max(df.loc[mask, "frame_risk"].mean(), 0.55) if mask.any() else 0.55, evt["dur"])
        closure_events.append({
            **evt,
            "severity": sev,
            "severity_score": sev_score,
            "confidence": conf,
            "alert_sent": conf >= MIN_ALERT_CONF and sev in ["high", "critical"],
            "alert_type": _alert_type_from_severity(sev),
            "event_confirmation_status": "pending_review" if conf < MIN_ALERT_CONF else "auto_confirmed",
            "risk_group": "fatigue",
        })

    high_binary = (df["frame_risk"] >= cfg["t_alert"]).astype(int)
    high_fatigue_events = []
    for evt in _streak_events(high_binary.values, fps, 2.0, "high_fatigue"):
        mask = (df["timestamp_seconds"] >= evt["start"]) & (df["timestamp_seconds"] <= evt["end"])
        conf = float(df.loc[mask, "fatigue_signal_confidence"].mean()) if mask.any() else 0.0
        sev, sev_score = _severity_bucket(max(df.loc[mask, "frame_risk"].mean(), 0.60) if mask.any() else 0.60, evt["dur"])
        high_fatigue_events.append({
            **evt,
            "severity": sev,
            "severity_score": sev_score,
            "confidence": conf,
            "alert_sent": conf >= MIN_ALERT_CONF and sev in ["high", "critical"],
            "alert_type": _alert_type_from_severity(sev),
            "event_confirmation_status": "pending_review" if conf < MIN_ALERT_CONF else "auto_confirmed",
            "risk_group": "fatigue",
        })

    uncertain_events = _streak_events(df["uncertain_frame"].values, fps, 2.0, "uncertain_segment_review")
    uncertain_event_rows = []
    for evt in uncertain_events:
        uncertain_event_rows.append({
            **evt,
            "severity": "medium",
            "severity_score": 0.50,
            "confidence": 0.20,
            "alert_sent": False,
            "alert_type": "review_queue",
            "event_confirmation_status": "pending_review",
            "risk_group": "review",
        })

    all_events = closure_events + high_fatigue_events + uncertain_event_rows

    total_duration = len(df) / fps if fps > 0 else 0
    blink_count = len(blink_events)
    avg_blink_dur = float(np.mean([e["dur"] for e in blink_events])) if blink_events else 0.0
    high_risk_dur = float((df["frame_risk"] >= cfg["t_alert"]).sum() / fps)
    caution_dur = float(((df["frame_risk"] >= cfg["t_caution"]) & (df["frame_risk"] < cfg["t_alert"]).astype(bool)).sum() / fps)
    uncertain_ratio = float(df["uncertain_frame"].mean())
    poor_quality_session = uncertain_ratio >= POOR_QUALITY_SESSION_RATIO
    avg_conf = float(df["fatigue_signal_confidence"].mean())

    final = "high" if high_risk_dur > 5 or len(closure_events) >= 2 else ("medium" if high_risk_dur > 2 or df["perclos_30s"].mean() > 0.15 or len(closure_events) >= 1 else "low")
    if poor_quality_session:
        final = final + "_review"

    summary = {
        "total_frames": int(len(df)),
        "total_duration": float(total_duration),
        "avg_drowsy": float(df["a1_prob_drowsy"].mean()),
        "max_drowsy": float(df["a1_prob_drowsy"].max()),
        "avg_eye": float(df["a2_prob_eye_closed"].mean()),
        "perclos": float(df["perclos_30s"].mean()),
        "eye_closure_burden": float(df["eye_closed_binary"].mean()),
        "closure_count": int(len(closure_events)),
        "blink_count": int(blink_count),
        "blink_freq_per_min": float(blink_count / total_duration * 60) if total_duration > 0 else 0.0,
        "avg_blink_duration": avg_blink_dur,
        "avg_ear": float(df["ear"].dropna().mean()) if df["ear"].notna().any() else None,
        "yawn_sup": float(df["a1_prob_yawn"].mean()),
        "nod_sup": float(df["a1_prob_nod"].mean()),
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
    td = len(df) / fps if fps > 0 else 0
    if sw > 1:
        for c in ["offroad_prob", "confidence", "entropy", "uncertainty_score"]:
            df[f"{c}_sm"] = df[c].rolling(sw, min_periods=1).mean()
    norm_entropy = df["entropy"] / np.log(len(ZONE_CLASSES))
    df["attn_uncertain"] = ((df["confidence"] < MIN_ALERT_CONF) | (norm_entropy > 0.75) | (df["uncertainty_score"] > 0.65)).astype(int)
    df["review_flag"] = np.where(df["attn_uncertain"] == 1, "needs_review", "clear")
    df["offroad_pred"] = (df["offroad_prob"] >= offroad_threshold).astype(int)
    df.loc[df["attn_uncertain"] == 1, "offroad_pred"] = 0  # suppress alerts when uncertain

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
        conf = float(df.loc[mask, "confidence"].mean()) if mask.any() else 0.0
        sev, sev_score = _severity_bucket(max(df.loc[mask, "offroad_prob"].mean(), 0.55) if mask.any() else 0.55, evt["dur"])
        offroad_events.append({
            **evt,
            "dominant_zone": df.loc[mask, "zone_pred"].mode().iloc[0] if mask.any() else "",
            "risk_group": "distraction",
            "severity": sev,
            "severity_score": sev_score,
            "confidence": conf,
            "alert_sent": conf >= MIN_ALERT_CONF and sev in ["high", "critical"],
            "alert_type": _alert_type_from_severity(sev),
            "event_confirmation_status": "pending_review" if conf < MIN_ALERT_CONF else "auto_confirmed",
        })

    repeated_distraction = []
    if td > 0 and ot >= 3:
        repeated_distraction.append({
            "type": "repeated_distraction",
            "start": 0.0,
            "end": float(df["timestamp_seconds"].iloc[-1]),
            "dur": float(df["timestamp_seconds"].iloc[-1]),
            "dominant_zone": df[df["offroad_pred"] == 1]["zone_pred"].mode().iloc[0] if (df["offroad_pred"] == 1).any() else "",
            "risk_group": "distraction",
            "severity": "high" if ot >= 5 else "medium",
            "severity_score": 0.80 if ot >= 5 else 0.60,
            "confidence": float(df["confidence"].mean()),
            "alert_sent": ot >= 4,
            "alert_type": "dashboard_escalation" if ot >= 4 else "dashboard_warning",
            "event_confirmation_status": "auto_confirmed",
        })

    mirror_events = []
    for evt in _streak_events(mb, fps, 0.0, "mirror_glance"):
        mask = (df["timestamp_seconds"] >= evt["start"]) & (df["timestamp_seconds"] <= evt["end"])
        mirror_events.append({
            **evt,
            "dominant_zone": df.loc[mask, "zone_pred"].mode().iloc[0] if mask.any() else "",
            "risk_group": "monitoring",
            "severity": "low",
            "severity_score": 0.20,
            "confidence": float(df.loc[mask, "confidence"].mean()) if mask.any() else 0.0,
            "alert_sent": False,
            "alert_type": "none",
            "event_confirmation_status": "auto_confirmed",
        })

    uncertain_events = []
    for evt in _streak_events(df["attn_uncertain"].values, fps, 2.0, "uncertain_segment_review"):
        uncertain_events.append({
            **evt,
            "dominant_zone": "",
            "risk_group": "review",
            "severity": "medium",
            "severity_score": 0.50,
            "confidence": 0.20,
            "alert_sent": False,
            "alert_type": "review_queue",
            "event_confirmation_status": "pending_review",
        })

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
        "mc": float(df["confidence"].mean()),
        "me": float(df["entropy"].mean()),
        "uncertain_ratio": float(df["attn_uncertain"].mean()),
        "poor_quality_session": float(df["attn_uncertain"].mean()) >= POOR_QUALITY_SESSION_RATIO,
        "repeated_distraction_events": int(len(repeated_distraction)),
    }
    return df, kpis, events


def build_timeline(df_a, df_b):
    if df_a.empty and df_b.empty:
        return pd.DataFrame()
    if not df_a.empty and not df_b.empty:
        timeline = pd.merge(df_a[["frame_id", "timestamp_seconds", "frame_risk", "eye_closed_binary", "uncertain_frame"]],
                            df_b[["frame_index", "timestamp_seconds", "offroad_pred", "risk_group_pred", "attn_uncertain"]],
                            on="timestamp_seconds", how="outer").sort_values("timestamp_seconds")
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
    return timeline


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
            rows.append({
                "timeline_state": current,
                "start_ts": times[start],
                "end_ts": times[i - 1],
                "duration_seconds": max(0.0, times[i - 1] - times[start] + (1 / fps if fps else 0)),
            })
            start = i
            current = labels[i]
    rows.append({
        "timeline_state": current,
        "start_ts": times[start],
        "end_ts": times[-1],
        "duration_seconds": max(0.0, times[-1] - times[start] + (1 / fps if fps else 0)),
    })
    return rows


def build_driver_scorecard(driver_id, trip_id, sid, a_summary, b_summary):
    fatigue_score = float(np.clip(
        0.45 * a_summary.get("avg_drowsy", 0)
        + 0.25 * a_summary.get("perclos", 0)
        + 0.20 * min(a_summary.get("closure_count", 0) / 3.0, 1.0)
        + 0.10 * min(a_summary.get("hr_dur", 0) / 10.0, 1.0),
        0,
        1,
    ))
    distraction_score = float(np.clip(
        0.45 * b_summary.get("or", 0)
        + 0.25 * min(b_summary.get("oepm", 0) / 6.0, 1.0)
        + 0.20 * (1 - b_summary.get("sfr", 0))
        + 0.10 * min(b_summary.get("repeated_distraction_events", 0) / 2.0, 1.0),
        0,
        1,
    ))
    combined = float(np.clip(0.55 * fatigue_score + 0.45 * distraction_score, 0, 1))
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
        "highrisk_events_per_driving_hour": float(b_summary.get("hepm", 0) * 60 / 60),
        "driver_rating": rating,
        "created_at_client": datetime.utcnow().isoformat(),
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


def extract_frames(uploaded, fps_target, file_ext):
    if not CV2_AVAILABLE:
        return []
    uploaded.seek(0)
    with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
        tmp.write(uploaded.read())
        tp = tmp.name
    cap = cv2.VideoCapture(tp)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(vfps / fps_target))
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append((idx / vfps, frame))
        idx += 1
    cap.release()
    os.unlink(tp)
    return frames


def analyze_video(frames, cfg_a, fps_b, bsw, offroad_threshold, driver_id, trip_id, trip_start_ts, trip_end_ts, source_name, sim_a_mode, sim_b_mode):
    m1, m2, bm = load_a1(), load_a2(), load_b()
    rows_a, rows_b, thumbs = [], [], []
    for i, (ts, frame_bgr) in enumerate(frames):
        meta = detect_face_eyes(frame_bgr)
        if sim_a_mode:
            ra = sim_a()
        else:
            o1 = infer_a1(m1, meta["face"])
            o2 = infer_a2(m2, meta["eye_strip"])
            ra = {**(o1 or {}), **(o2 or {})} if o1 or o2 else sim_a()
        if sim_b_mode:
            rb = sim_b(offroad_threshold)
        else:
            rb = infer_b(bm, frame_bgr, offroad_threshold) or sim_b(offroad_threshold)

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
            "preprocess_method": meta.get("preprocess_method"),
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
            **rb,
        }
        rows_b.append(row_b)

        if i % max(1, len(frames) // 8) == 0:
            thumbs.append((i, ts, frame_bgr.copy(), meta, ra, rb))

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    df_a, a_summary, a_events = temporal_a(df_a, cfg_a)
    df_b, b_summary, b_events = temporal_b(df_b, fps_b, bsw, offroad_threshold)
    return df_a, a_summary, a_events, df_b, b_summary, b_events, thumbs


def home_page():
    st.title(":material/directions_car: Driver Safety Analytics Platform")
    st.markdown("Unified fatigue + attentiveness analytics with MediaPipe-based preprocessing and Snowflake-backed storage.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("A1 Model", "Loaded" if load_a1() else "Not loaded")
        st.metric("A2 Model", "Loaded" if load_a2() else "Not loaded")
    with c2:
        st.metric("Module B Model", "Loaded" if load_b() else "Not loaded")
        st.metric("MediaPipe", "Available" if MP_AVAILABLE else "Fallback mode")
    with c3:
        pkg_status = {
            "PyTorch": TORCH_AVAILABLE,
            "OpenCV": CV2_AVAILABLE,
            "Pillow": PIL_AVAILABLE,
        }
        st.caption("Package status")
        for k, v in pkg_status.items():
            st.write(f"- {'OK' if v else 'MISSING'}: {k}")
        st.caption(f"Snowflake session: {'Connected' if get_session() else 'Not connected'}")

    st.divider()
    with st.container(border=True):
        st.subheader("What’s new in this version")
        st.markdown(
            """
- MediaPipe Face Mesh preprocessing with exact eye region crop and head pose angles
- EAR, blink count, blink duration, and PERCLOS features
- Uncertainty-aware suppression and review tagging
- Driver ID / Trip ID / trip context stored with outputs
- Driver scorecard and combined safety score
- Driver timeline with escalation-ready event count panels
            """
        )


def unified_page():
    st.title(":material/security: Unified Driver Safety Analyzer")
    with st.sidebar:
        st.subheader("Driver & Trip Context")
        driver_id = st.text_input("Driver ID", value="DRV_001")
        trip_id = st.text_input("Trip ID", value="TRIP_001")
        trip_start_ts = st.text_input("Trip Start TS", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        trip_end_ts = st.text_input("Trip End TS", value="")

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
        afps = st.number_input("Sampling FPS", 1, 30, DEFAULT_A["fps"])
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
        st.info(":material/science: Some models are not loaded. The app will use simulation mode where needed.")

    uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])
    if not uploaded:
        st.caption("Upload a file to begin analysis.")
        return
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    is_video = ext in ("mp4", "avi", "mov", "mkv")
    sid = str(uuid.uuid4())[:12]

    if not is_video:
        st.warning("Unified driver scorecard, combined safety score, and driver timeline are most meaningful on video input.")
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
        return

    frames = extract_frames(uploaded, afps, ext)
    if not frames:
        st.error("Could not extract frames.")
        return

    st.caption(f"Extracted {len(frames)} frames at ~{afps} fps")
    prog = st.progress(0, "Processing...")
    # process with manual progress pulses
    step = max(1, len(frames) // 20)
    for i in range(0, len(frames), step):
        prog.progress(min((i + step) / len(frames), 1.0), f"Preparing batch around frame {min(i + step, len(frames))}/{len(frames)}")
    df_a, a_summary, a_events, df_b, b_summary, b_events, thumbs = analyze_video(
        frames, cfg_a, afps, bsw, bort, driver_id, trip_id, trip_start_ts, trip_end_ts, uploaded.name, sim_a_mode, sim_b_mode
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
    }

    # write outputs
    fdf_a = df_a[[
        "session_id", "driver_id", "trip_id", "source_type", "source_name", "frame_id", "timestamp_seconds",
        "a1_prob_drowsy", "a1_prob_yawn", "a1_prob_nod", "a2_prob_eye_closed", "a2_eye_openness_score",
        "frame_risk", "risk_level", "head_pitch", "head_yaw", "head_roll", "ear", "left_ear", "right_ear",
        "quality_score", "face_confidence", "preprocess_method", "fatigue_signal_confidence", "uncertain_frame",
        "review_flag", "eye_closed_binary", "blink_binary", "perclos_30s", "blink_rate_per_min"
    ]].copy()
    write_to_snowflake(fdf_a, "MODULE_A_FRAME_PREDICTIONS")

    fdf_b = df_b[[
        "session_id", "driver_id", "trip_id", "input_type", "source_file_name", "frame_index", "timestamp_seconds",
        "zone_pred", "risk_group_pred", "offroad_prob", "offroad_pred", "confidence", "entropy", "margin",
        "uncertainty_score", "attn_uncertain", "review_flag"
    ] + [f"zone_prob_{z.lower().replace(' ', '_')}" for z in ZONE_CLASSES]].copy()
    write_to_snowflake(fdf_b, "MODULE_B_FRAME_PREDICTIONS")

    a_event_rows = []
    for ev in a_events:
        a_event_rows.append({
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
        })
    if a_event_rows:
        write_to_snowflake(pd.DataFrame(a_event_rows), "MODULE_A_EVENTS")

    b_event_rows = []
    for ev in b_events:
        b_event_rows.append({
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
        })
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
    }
    write_to_snowflake(pd.DataFrame([b_summary_row]), "MODULE_B_SESSION_SUMMARY")

    write_to_snowflake(pd.DataFrame([unified_summary]), "UNIFIED_DRIVER_SESSION_SUMMARY")
    write_to_snowflake(pd.DataFrame([scorecard]), "DRIVER_SCORECARDS")
    if timeline_rows:
        timeline_upload = pd.DataFrame(timeline_rows)
        timeline_upload["session_id"] = sid
        timeline_upload["driver_id"] = driver_id
        timeline_upload["trip_id"] = trip_id
        write_to_snowflake(timeline_upload, "DRIVER_TIMELINE")

    risk_color = "green" if combined_driver_safety_score < 0.35 else ("orange" if combined_driver_safety_score < 0.65 else "red")
    st.markdown(f"### Unified Session Score: :{risk_color}[{combined_driver_safety_score:.2f}] | Driver Rating: **{scorecard['driver_rating']}**")
    st.caption(combined_summary_text)

    t1, t2, t3, t4, t5, t6 = st.tabs(["Overview", "Fatigue", "Attentiveness", "Timeline", "Events", "Scorecard"])
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
    with t2:
        fatigue_trend = df_a[["timestamp_seconds", "frame_risk", "a1_prob_drowsy", "a2_prob_eye_closed", "perclos_30s", "blink_rate_per_min"]].rename(columns={
            "timestamp_seconds": "Time", "frame_risk": "Fatigue Risk", "a1_prob_drowsy": "Drowsy", "a2_prob_eye_closed": "Eye Closed", "perclos_30s": "PERCLOS", "blink_rate_per_min": "Blink/min"
        })
        st.line_chart(fatigue_trend, x="Time")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg EAR", f"{a_summary['avg_ear']:.3f}" if a_summary["avg_ear"] is not None else "N/A")
        c2.metric("Blink Count", a_summary["blink_count"])
        c3.metric("Avg Blink Duration", f"{a_summary['avg_blink_duration']:.2f}s")
        c4.metric("Poor-quality Ratio", f"{a_summary['uncertain_ratio']:.1%}")
    with t3:
        att_trend = df_b[["timestamp_seconds", "offroad_prob", "confidence", "uncertainty_score"]].rename(columns={
            "timestamp_seconds": "Time", "offroad_prob": "Off-road Prob", "confidence": "Confidence", "uncertainty_score": "Uncertainty"
        })
        st.line_chart(att_trend, x="Time")
        zd = df_b["zone_pred"].value_counts().reset_index()
        zd.columns = ["Zone", "Count"]
        st.bar_chart(zd, x="Zone", y="Count")
    with t4:
        if timeline_df.empty:
            st.caption("No timeline available")
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
    with t6:
        st.dataframe(pd.DataFrame([scorecard]), hide_index=True, use_container_width=True)
        st.download_button("Download Scorecard CSV", pd.DataFrame([scorecard]).to_csv(index=False), f"driver_scorecard_{sid}.csv", "text/csv")

    combined_export = pd.merge(df_a, df_b, left_on="timestamp_seconds", right_on="timestamp_seconds", how="outer", suffixes=("_A", "_B"))
    st.download_button("Download Combined Frame CSV", combined_export.to_csv(index=False), f"unified_driver_safety_{sid}.csv", "text/csv")


def analytics_page():
    st.title(":material/analytics: Historical Analytics")
    tab1, tab2, tab3 = st.tabs(["Unified Sessions", "Driver Scorecards", "Timeline States"])
    with tab1:
        df = query_sf(f"SELECT * FROM {DATABASE}.{SCHEMA}.UNIFIED_DRIVER_SESSION_SUMMARY ORDER BY CREATED_AT DESC LIMIT 50")
        if df.empty:
            st.caption("No unified sessions yet.")
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)
            if {"DRIVER_ID", "COMBINED_DRIVER_SAFETY_SCORE"}.issubset(df.columns):
                score_df = df[["DRIVER_ID", "COMBINED_DRIVER_SAFETY_SCORE"]].groupby("DRIVER_ID", as_index=False).mean()
                st.bar_chart(score_df, x="DRIVER_ID", y="COMBINED_DRIVER_SAFETY_SCORE")
    with tab2:
        sc = query_sf(f"SELECT * FROM {DATABASE}.{SCHEMA}.DRIVER_SCORECARDS ORDER BY CREATED_AT DESC LIMIT 50")
        if sc.empty:
            st.caption("No driver scorecards yet.")
        else:
            st.dataframe(sc, hide_index=True, use_container_width=True)
            if {"DRIVER_ID", "DRIVER_RATING"}.issubset(sc.columns):
                rating_counts = sc["DRIVER_RATING"].value_counts().reset_index()
                rating_counts.columns = ["Driver Rating", "Count"]
                st.bar_chart(rating_counts, x="Driver Rating", y="Count")
    with tab3:
        tl = query_sf(f"SELECT * FROM {DATABASE}.{SCHEMA}.DRIVER_TIMELINE ORDER BY CREATED_AT DESC LIMIT 200")
        if tl.empty:
            st.caption("No timeline rows yet.")
        else:
            st.dataframe(tl, hide_index=True, use_container_width=True)
            if {"TIMELINE_STATE", "DURATION_SECONDS"}.issubset(tl.columns):
                state_dur = tl.groupby("TIMELINE_STATE", as_index=False)["DURATION_SECONDS"].sum()
                st.bar_chart(state_dur, x="TIMELINE_STATE", y="DURATION_SECONDS")


pages = [
    st.Page(home_page, title="Home", icon=":material/home:"),
    st.Page(unified_page, title="Unified Safety", icon=":material/security:"),
    st.Page(analytics_page, title="Historical Analytics", icon=":material/analytics:"),
]
pg = st.navigation(pages)
pg.run()
