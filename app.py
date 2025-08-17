# app.py — Multilingual, Visual-Heavy 2-Stage Deepfake Cascade with MediaPipe + Hindi/English NLP
import os
import uuid
import json
import numpy as np
import cv2
import torch
import torchaudio
import librosa
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Optional: dlib for fallback landmarks if MediaPipe not available
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False

try:
    import dlib
    DLIB_OK = True
except Exception:
    DLIB_OK = False

from skimage.metrics import structural_similarity as ssim

from transformers import pipeline
# avoid bitsandbytes/peft chain at import time
# from sentence_transformers import SentenceTransformer, util
from speechbrain.inference import SpeakerRecognition
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# ======================
# Device helpers
# ======================
CUDA = torch.cuda.is_available()
DEVICE_ID = 0 if CUDA else -1
TORCH_DEVICE = "cuda" if CUDA else "cpu"

# ======================
# Demo Mode & Cache
# ======================
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
DEMO_CACHE = "./demo_cache"
os.makedirs(DEMO_CACHE, exist_ok=True)

def file_sha1_like_upload(upload_path: str, chunk=1<<20) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(upload_path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def load_demo_result_by_hash(h: str) -> Optional[dict]:
    p = os.path.join(DEMO_CACHE, f"{h}.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ======================
# Globals: Face landmark helpers (MediaPipe preferred)
# ======================
MP_FACE = None  # lazy init
MP_DRAW = None

def init_mediapipe():
    global MP_FACE, MP_DRAW
    if not MP_OK:
        return False
    if MP_FACE is None:
        mp_face_mesh = mp.solutions.face_mesh
        MP_FACE = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        MP_DRAW = mp.solutions.drawing_utils
    return True

def ensure_landmark_model(path: str = "shape_predictor_68_face_landmarks.dat"):
    if not DLIB_OK:
        return
    if os.path.exists(path):
        return
    try:
        import urllib.request, bz2, shutil
        url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        tmp = path + ".bz2"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, tmp)
        print("Decompressing landmarks...")
        with bz2.BZ2File(tmp) as fr, open(path, "wb") as fw:
            shutil.copyfileobj(fr, fw)
        os.remove(tmp)
    except Exception as e:
        print("Could not auto-download landmarks:", e)

def mp_face_landmarks(frame: np.ndarray):
    """Return 468x2 landmark array in pixel coords, or None."""
    if not init_mediapipe():
        return None
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = MP_FACE.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark], dtype=np.float32)
    return pts

def dlib_face_bbox_and_shape(frame: np.ndarray):
    """Fallback: return (bbox, shape) where shape has .part() API, else (None, None)."""
    if not DLIB_OK:
        return None, None
    ensure_landmark_model()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = dlib.get_frontal_face_detector()
    pred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = det(gray)
    if len(faces) == 0:
        return None, None
    shape = pred(gray, faces[0])
    return faces[0], shape

def get_face_roi(frame: np.ndarray, margin: int = 20) -> Optional[np.ndarray]:
    """Face crop using MediaPipe landmarks, fallback to dlib bbox."""
    if MP_OK and init_mediapipe():
        pts = mp_face_landmarks(frame)
        if pts is not None and len(pts) > 0:
            x1 = max(0, int(np.min(pts[:,0]) - margin))
            y1 = max(0, int(np.min(pts[:,1]) - margin))
            x2 = min(frame.shape[1], int(np.max(pts[:,0]) + margin))
            y2 = min(frame.shape[0], int(np.max(pts[:,1]) + margin))
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]
    # fallback
    if DLIB_OK:
        bbox, shape = dlib_face_bbox_and_shape(frame)
        if bbox is not None:
            x1, y1, x2, y2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
            x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin); y2 = min(frame.shape[0], y2 + margin)
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]
    return None

def mp_mouth_aspect_ratio(frame: np.ndarray) -> Optional[float]:
    """
    Approx MAR with MediaPipe:
      vertical: distance(13,14)
      horizontal: distance(61,291)
    """
    if not MP_OK or not init_mediapipe():
        return None
    pts = mp_face_landmarks(frame)
    if pts is None or pts.shape[0] < 292:
        return None
    def dist(a, b): return float(np.linalg.norm(pts[a] - pts[b]))
    horiz = dist(61, 291)
    vert  = dist(13, 14)
    if horiz <= 1e-6:
        return None
    return float(np.clip(vert / horiz, 0.0, 1.0))

def dlib_mouth_aspect_ratio(frame: np.ndarray) -> Optional[float]:
    """Fallback MAR using dlib 68 points."""
    if not DLIB_OK:
        return None
    ensure_landmark_model()
    bbox, shape = dlib_face_bbox_and_shape(frame)
    if shape is None:
        return None
    top = np.array([shape.part(51).x, shape.part(51).y], dtype=np.float32)
    bot = np.array([shape.part(57).x, shape.part(57).y], dtype=np.float32)
    lc  = np.array([shape.part(48).x, shape.part(48).y], dtype=np.float32)
    rc  = np.array([shape.part(54).x, shape.part(54).y], dtype=np.float32)
    horiz = float(np.linalg.norm(rc - lc))
    vert  = float(np.linalg.norm(bot - top))
    if horiz <= 1e-6:
        return None
    return float(np.clip(vert / horiz, 0.0, 1.0))

def mouth_aspect_ratio(frame: np.ndarray) -> Optional[float]:
    mar = mp_mouth_aspect_ratio(frame)
    if mar is not None:
        return mar
    return dlib_mouth_aspect_ratio(frame)

# ======================
# Models (global) — LAZY with DEMO_MODE guards
# ======================
print(">>> Loading models (auto device), please wait...")

asr = sentiment_pipe = zero_shot = clip_zero_shot = None
paraphrase_model = None
ST_UTIL = None  # holder for sentence_transformers.util if available
recognizer = None
visual_model = None
visual_preprocess = None

if not DEMO_MODE:
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration  # noqa: F401
        ASR_MODEL_NAME = "openai/whisper-small"
        asr = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device=DEVICE_ID)

        SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL, device=DEVICE_ID)

        ZSL_MODEL = "joeddav/xlm-roberta-large-xnli"
        zero_shot = pipeline("zero-shot-classification", model=ZSL_MODEL, device=DEVICE_ID)

        clip_zero_shot = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device=DEVICE_ID)

        # Try sentence_transformers only here
        try:
            from sentence_transformers import SentenceTransformer as _ST, util as _ST_UTIL
            paraphrase_model = _ST("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=TORCH_DEVICE)
            ST_UTIL = _ST_UTIL
        except Exception as e:
            paraphrase_model = None
            ST_UTIL = None
            print("sentence_transformers unavailable; script deviation disabled:", e)

        recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": TORCH_DEVICE}
        )

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        visual_model = efficientnet_b4(weights=weights).to(TORCH_DEVICE).eval()
        visual_preprocess = weights.transforms()
    except Exception as e:
        print("Model load error:", e)

print(">>> All models loaded and ready <<<" if not DEMO_MODE else ">>> DEMO_MODE: Skipping heavy model loads <<<")

# ======================
# Dirs
# ======================
SCRIPT_DIR = "./scripts"
UPLOAD_DIR = "./uploads"
LOG_DIR = "./audit_logs"
for d in [SCRIPT_DIR, UPLOAD_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ======================
# FastAPI
# ======================
controller_app = FastAPI(title="Deepfake Cascade (2-stage, visual-heavy/accuracy-weighted, multilingual)")
controller_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ======================
# Data models
# ======================
class VideoMetadata(BaseModel):
    codec: str
    duration: float
    resolution: Tuple[int, int]
    frame_rate: float
    has_audio: bool
    has_visual: bool
    language: Optional[str] = None
    source: str
    frame_count: Optional[int] = None

class AgentResponse(BaseModel):
    agent: str
    logit: float
    evidence: Dict[str, float]
    confidence_interval: Tuple[float, float]
    processing_time: float

class DetectionResult(BaseModel):
    version: str
    mode: str
    label: str
    deepfake_score: float
    confidence: Tuple[float, float]
    stage_reached: int
    stages_run: List[int]
    top_contributors: List[str]
    agent_evidence: Dict[str, str]
    explanation: str

# ======================
# Math / helpers
# ======================
def radial_power_peakiness(gray: np.ndarray) -> float:
    """Robust radial 'peakiness' using full FFT and fftshift; higher ~ more periodic (GAN-ish)."""
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    F = np.fft.fft2(gray)
    F = np.abs(np.fft.fftshift(F))
    h, w = F.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - w/2.0)**2 + (yy - h/2.0)**2).astype(np.int32)
    r_flat = rr.ravel(); F_flat = F.ravel()
    max_r = int(r_flat.max())
    radial_sum = np.bincount(r_flat, weights=F_flat, minlength=max_r + 1)
    radial_cnt = np.bincount(r_flat, minlength=max_r + 1).astype(np.float32)
    radial = radial_sum / (radial_cnt + 1e-6)
    if radial.size < 16:
        return 0.0
    band = radial[radial.size // 3 :]
    peakiness = float((band.max() + 1e-6) / (np.median(band) + 1e-6))
    return peakiness

def label_from_score(score: float, real_cut: float, fake_cut: float) -> str:
    return "fake" if score >= fake_cut else ("real" if score <= real_cut else "uncertain")

STRONG_FAKE_KEYS = ["prnu", "gan_fft", "depth_parallax", "av_sync", "scene_consistency", "id_drift"]

RELIABILITY = {
    "visual": 0.82, "av_sync": 0.78, "audio": 0.72, "text": 0.68, "metadata": 0.55, "temporal": 0.50,
    "prnu": 0.88, "gan_fft": 0.85, "depth_parallax": 0.80, "scene_consistency": 0.75, "id_drift": 0.72, "compression": 0.60,
    "audio_spoof": 0.78
}

def resolve_uncertain_with_tiebreakers(per: dict, score: float, real_cut: float, fake_cut: float):
    rules = []
    active = {k:v for k,v in per.items() if isinstance(v, dict) and "logit" in v}
    if not active:
        return "uncertain", ["no_active_layers"]
    strong_hits = [k for k in STRONG_FAKE_KEYS if k in active and float(active[k]["logit"]) >= 0.85]
    if len(strong_hits) >= 2:
        rules.append(f"strong_fake_layers={','.join(strong_hits)}")
        return "fake", rules
    low_realish = [k for k,v in active.items() if float(v["logit"]) <= 0.25]
    if len(low_realish) >= max(2, len(active)//2 + 1):
        rules.append(f"majority_low_logits={','.join(low_realish[:5])}")
        return "real", rules
    total_w = sum(RELIABILITY.get(k, 0.0) for k in active) or 1.0
    wmean = sum(RELIABILITY.get(k, 0.0) * float(v["logit"]) for k,v in active.items()) / total_w
    mid = 0.5*(real_cut + fake_cut)
    rules.append(f"reliability_weighted_mean={wmean:.2f} vs mid={mid:.2f}")
    return ("fake" if wmean >= mid else "real"), rules

def _layer_sentence(name: str, v: dict) -> str:
    ev = v.get("evidence", {}) if isinstance(v, dict) else {}
    kv = list(ev.items())[:2]
    kv_txt = ", ".join(f"{k}={float(x):.3f}" for k,x in kv if isinstance(x, (int,float)))
    return f"{name}: logit={float(v.get('logit',0)):.2f}" + (f" ({kv_txt})" if kv_txt else "")

def build_natural_language_explanation(per: dict, label: str, score: float, top_contribs: List[str]) -> str:
    active = {k: v for k, v in per.items() if isinstance(v, dict) and "logit" in v}
    pros = sorted([(k, v) for k, v in active.items() if float(v["logit"]) >= 0.80],
                  key=lambda x: float(x[1]["logit"]), reverse=True)[:3]
    cons = sorted([(k, v) for k, v in active.items() if float(v["logit"]) <= 0.20],
                  key=lambda x: float(x[1]["logit"]))[:2]

    pro_txt = "; ".join(_layer_sentence(k, v) for k, v in pros) or "none strong"
    con_txt = "; ".join(_layer_sentence(k, v) for k, v in cons) or "no strong counter-evidence"

    if label == "fake":
        interp = f"Interpretation: higher logits ⇒ more likely fake; fusion={score:.2f} supports this."
    elif label == "real":
        interp = f"Interpretation: lower logits ⇒ more likely real; fusion={score:.2f} supports this."
    else:  # 'uncertain'
        interp = f"Interpretation: mid-range logits (~0.5) indicate uncertainty; fusion={score:.2f} is inconclusive."

    return (
        f"Decision: {label.upper()} (fusion={score:.2f}). "
        f"Top contributors: {', '.join(top_contribs) if top_contribs else 'n/a'}. "
        f"Strong ‘fake’ indicators: {pro_txt}. "
        f"Counter-evidence: {con_txt}. "
        f"{interp} "
        f"If high-stakes, consider strict mode or human review."
    )

# ======================
# IO
# ======================
def process_video(file_path: str) -> Dict:
    # Audio?
    has_audio = bool(os.popen(
        f'ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "{file_path}"'
    ).read().strip())
    audio_path = None
    if has_audio:
        audio_path = f"{file_path}_audio.wav"
        os.system(f'ffmpeg -y -i "{file_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}"')

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames, count = [], 0
    # ~1 fps up to 8s
    while cap.isOpened() and count < int(fps * 8):
        ret, frame = cap.read()
        if not ret: break
        if int(count % max(1, int(fps))) == 0:
            frames.append(frame)
        count += 1
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    codec = os.popen(
        f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "{file_path}"'
    ).read().strip()
    duration = count / (fps if fps > 0 else 1)

    metadata = VideoMetadata(
        codec=codec or "unknown",
        duration=float(duration),
        resolution=(width, height),
        frame_rate=float(fps),
        has_audio=bool(has_audio),
        has_visual=bool(len(frames) > 0),
        language=None,
        source='file',
        frame_count=total_frames if total_frames > 0 else None
    )
    return {'metadata': metadata, 'keyframes': frames, 'audio_path': audio_path}

# ======================
# Stage 1 agents (upgraded + multilingual)
# ======================
def visual_agent_task(task_id: str, frames: List[np.ndarray]):
    """
    Multi-cue visual prior (frame-only), Stage-1:
      - CLIP zero-shot prior
      - GAN-like FFT peakiness
      - Facial symmetry
      - ImageNet OODness proxy
    """
    t0 = datetime.utcnow().timestamp()
    clip_probs: List[float] = []
    fft_peaks: List[float] = []
    sym_asym:  List[float] = []
    oodness:   List[float] = []

    for f in frames:
        # --- Face ROI (prefer face crop when available)
        crop = get_face_roi(f, margin=20)
        roi  = crop if crop is not None else f

        # --- CLIP prior (AI vs real)
        try:
            if DEMO_MODE or clip_zero_shot is None:
                ai_prob = 0.5
            else:
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                res = clip_zero_shot(
                    pil,
                    candidate_labels=["AI-generated image", "real image"],
                    hypothesis_template="This is {}."
                )
                ai_prob = next((r["score"] for r in res if r["label"] == "AI-generated image"), 0.5)
        except Exception:
            ai_prob = 0.5
        clip_probs.append(float(ai_prob))

        # --- FFT peakiness
        gray_u8 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        peak = radial_power_peakiness(gray_u8.astype(np.float32))
        fft_norm = float(np.clip((peak - 2.0) / 2.0, 0.0, 1.0))
        fft_peaks.append(fft_norm)

        # --- Facial symmetry
        mid = gray_u8.shape[1] // 2
        left = gray_u8[:, :mid]
        right = gray_u8[:, mid:]
        right_m = cv2.flip(right, 1)
        h = min(left.shape[0], right_m.shape[0])
        w = min(left.shape[1], right_m.shape[1])
        if h > 8 and w > 8:
            try:
                s = ssim(left[:h, :w], right_m[:h, :w])
                asym = float(np.clip(1.0 - s, 0.0, 1.0))
            except Exception:
                asym = 0.0
        else:
            asym = 0.0
        sym_asym.append(asym)

        # --- EfficientNet OODness
        try:
            if DEMO_MODE or visual_model is None or visual_preprocess is None:
                ood = 0.5
            else:
                ten = visual_preprocess(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(TORCH_DEVICE)
                with torch.no_grad():
                    logits = visual_model(ten)
                    prob = torch.softmax(logits, dim=1).max(dim=1).values.item()
                ood = float(np.clip(1.0 - prob, 0.0, 1.0))
        except Exception:
            ood = 0.5
        oodness.append(ood)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    clip_m = mean(clip_probs); fft_m = mean(fft_peaks); sym_m = mean(sym_asym); ood_m = mean(oodness)
    logit = float(np.clip(0.40 * clip_m + 0.25 * fft_m + 0.20 * sym_m + 0.15 * ood_m, 0.0, 1.0))
    t1 = datetime.utcnow().timestamp()
    return AgentResponse(
        agent='visual',
        logit=logit,
        evidence={'clip_ai_prob_mean': clip_m, 'fft_peakiness_norm_mean': fft_m, 'symmetry_asym_mean': sym_m, 'imagenet_ood_mean': ood_m},
        confidence_interval=(max(0.0, logit - 0.08), min(1.0, logit + 0.08)),
        processing_time=float(t1 - t0)
    ).dict()

def audio_anti_spoof(y: np.ndarray, sr: int) -> float:
    """Lightweight anti-spoof proxy in [0,1]."""
    try:
        S_flat = librosa.feature.spectral_flatness(y=y)
        flat_m = float(np.median(S_flat))
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_var = float(np.var(cent) / (np.mean(cent)**2 + 1e-6))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        dmfcc = librosa.feature.delta(mfcc)
        dvar = float(np.mean(np.var(dmfcc, axis=1)))
        flat_s = np.clip((flat_m - 0.1) / 0.5, 0, 1)
        cent_s = np.clip(cent_var / 2.0, 0, 1)
        dvar_s = np.clip(dvar / 50.0, 0, 1)
        return float(np.clip(0.5*flat_s + 0.3*cent_s + 0.2*dvar_s, 0, 1))
    except Exception:
        return 0.5

def audio_agent_task(task_id: str, audio_path: str):
    """Audio-only cue."""
    t0 = datetime.utcnow().timestamp()
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception:
        waveform = torch.zeros(1, 16000)
        sr = 16000

    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != 16000:
        try:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
            sr = 16000
        except Exception:
            sr = 16000

    y = waveform.squeeze(0).cpu().numpy()
    try:
        rms = float(np.mean(librosa.feature.rms(y=y, frame_length=512, hop_length=256)))
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    except Exception:
        rms, flat = 0.0, 0.0

    norm_score = 0.5
    if not DEMO_MODE and recognizer is not None:
        try:
            with torch.no_grad():
                emb = recognizer.encode_batch(waveform.to(TORCH_DEVICE))
            norm = float(torch.linalg.norm(emb).item())
            norm_score = float(np.clip(norm / (norm + 1.0), 0.0, 1.0))
        except Exception:
            norm_score = 0.5

    flat_score = float(np.clip((flat - 0.10) / 0.40, 0.0, 1.0))
    spoof = audio_anti_spoof(y, sr)
    logit = float(np.clip(0.6 * norm_score + 0.2 * flat_score + 0.2 * spoof, 0.0, 1.0))
    t1 = datetime.utcnow().timestamp()
    return AgentResponse(
        agent='audio',
        logit=logit,
        evidence={'embedding_norm': float(norm_score), 'rms': rms, 'spectral_flatness': flat, 'anti_spoof': spoof},
        confidence_interval=(max(0.0, logit - 0.10), min(1.0, logit + 0.10)),
        processing_time=float(t1 - t0)
    ).dict()

def load_original_script(task_id: str) -> Optional[str]:
    p = os.path.join(SCRIPT_DIR, f"{task_id}.txt")
    if not os.path.isfile(p): return None
    with open(p, "r", encoding="utf-8") as f: return f.read()

def text_agent_task(task_id: str, audio_path: str, lang: str):
    """Transcript → sentiment + zero-shot NLI + optional script deviation."""
    transcript = ""
    incons = 0.5
    dev = 0.0
    sentiment_score = 0.0

    # ASR
    if not DEMO_MODE and asr is not None:
        gen_kwargs = {}
        if lang in {"hi", "en"}:
            gen_kwargs = {"language": lang}
        try:
            asr_out = asr(audio_path, generate_kwargs=gen_kwargs) if gen_kwargs else asr(audio_path)
            transcript = asr_out.get('text', "")
        except Exception:
            transcript = ""

    # Sentiment
    if not DEMO_MODE and sentiment_pipe is not None and transcript:
        try:
            sent = sentiment_pipe(transcript)[0]
            sentiment_score = float(sent.get('score', 0.0)) * (1 if sent.get('label','').upper().endswith('POSITIVE') else -1)
        except Exception:
            sentiment_score = 0.0

    # Zero-shot consistency
    if not DEMO_MODE and zero_shot is not None and transcript:
        try:
            z = zero_shot(transcript, candidate_labels=['consistent','inconsistent'], hypothesis_template="This statement is {}.")
            incons = float(z['scores'][z['labels'].index('inconsistent')])
        except Exception:
            incons = 0.5

    # Script deviation
    original = load_original_script(task_id)
    if not DEMO_MODE and paraphrase_model is not None and ST_UTIL is not None and original and transcript:
        try:
            emb1, emb2 = paraphrase_model.encode([transcript, original], convert_to_tensor=True, normalize_embeddings=True)
            dev = 1 - float(ST_UTIL.cos_sim(emb1, emb2).item())
        except Exception:
            dev = 0.0

    logit = float(np.clip(0.6 * incons + 0.4 * abs(dev), 0, 1))
    return AgentResponse(agent='text', logit=logit,
                         evidence={'sentiment_score': float(sentiment_score),
                                   'context_inconsistency': float(incons),
                                   'script_deviation': float(dev)},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)),
                         processing_time=0.0).dict()

def metadata_agent_task(task_id: str, metadata: dict):
    codec = (metadata.get('codec') or '').lower()
    fps = float(metadata.get('frame_rate') or 0.0)
    frame_count = metadata.get('frame_count', None)
    duration = float(metadata.get('duration') or 0.0)
    w, h = metadata.get('resolution', (0,0))
    has_audio = bool(metadata.get('has_audio'))
    codec_ok = codec in ['avc1','avc3','h264','hev1','h265','vp09','vp9','av01']
    fps_ok = fps <= 0 or (8 <= fps <= 120)
    res_ok = (w >= 320 and h >= 240)
    timing_ok = True
    if fps > 0 and frame_count:
        timing_ok = abs((frame_count / fps) - duration) < 0.2 * max(1.0, duration)
    issues = int(not codec_ok) + int(not fps_ok) + int(not res_ok) + int(not timing_ok)
    logit = float(min(1.0, 0.25 * issues))
    return AgentResponse(agent='metadata', logit=logit,
                         evidence={'codec_ok': float(codec_ok),'fps_ok': float(fps_ok),
                                   'res_ok': float(res_ok),'timing_ok': float(timing_ok),'has_audio': float(has_audio)},
                         confidence_interval=(0,1), processing_time=0.0).dict()

def temporal_agent_task(task_id: str, frames: list):
    if not frames:
        return AgentResponse(agent='temporal', logit=0.0, evidence={'motion_jitter': 0.0},
                             confidence_interval=(0, 0.05), processing_time=0.0).dict()
    diffs = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.mean(np.abs(curr.astype(float)-prev.astype(float)))))
        prev = curr
    jitter = float(np.clip(np.mean(diffs)/255.0, 0, 1)) if diffs else 0.0
    return AgentResponse(agent='temporal', logit=jitter,
                         evidence={'motion_jitter': jitter},
                         confidence_interval=(max(0, jitter-0.05), min(1, jitter+0.05)),
                         processing_time=0.0).dict()

def av_sync_agent_task(task_id: str, audio_path: str, frames: list):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception:
        y = np.zeros(16000, dtype=np.float32)
        sr = 16000
    try:
        env = librosa.feature.rms(y=y, frame_length=512, hop_length=256).flatten()
    except Exception:
        env = np.zeros(16, dtype=np.float32)

    ratios = []
    for f in frames:
        mar = mouth_aspect_ratio(f)
        if mar is not None:
            ratios.append(float(mar))
    if len(ratios) >= 3 and len(env) >= 3:
        L = min(len(env), len(ratios))
        env_r = cv2.resize(env.reshape(-1,1).astype(np.float32), (1, L), interpolation=cv2.INTER_LINEAR).flatten()
        mar_r = cv2.resize(np.array(ratios, dtype=np.float32).reshape(-1,1), (1, L), interpolation=cv2.INTER_LINEAR).flatten()
        try:
            corr = float(np.corrcoef(env_r, mar_r)[0,1])
        except Exception:
            corr = 0.0
        mismatch = float(np.clip(1.0 - max(0.0, corr), 0, 1))
    else:
        corr, mismatch = 0.0, 1.0
    logit = float(np.clip(mismatch, 0, 1))
    return AgentResponse(agent='av_sync', logit=logit,
                         evidence={'corr': corr, 'mismatch': mismatch},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)),
                         processing_time=0.0).dict()

# ======================
# Stage 2 agents (heavier add-ons)
# ======================
def prnu_agent_task(task_id: str, frames: List[np.ndarray]):
    if len(frames) < 3:
        return AgentResponse(agent='prnu', logit=0.5, evidence={'prnu_xcorr_mean':0.0,'residual_snr':0.0}, confidence_interval=(0,1), processing_time=0.0).dict()
    residuals, snrs = [], []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(g, (3,3), 0); res = (g - blur)
        res = (res - res.mean()) / (res.std() + 1e-6)
        residuals.append(res.ravel())
        snrs.append(float(np.std(res) / (np.std(g)+1e-6)))
    xs = []
    for i in range(1, len(residuals)):
        r = float(np.corrcoef(residuals[i-1], residuals[i])[0,1])
        if np.isfinite(r): xs.append(r)
    prnu_xcorr_mean = float(np.mean(xs)) if xs else 0.0
    residual_snr = float(np.mean(snrs)) if snrs else 0.0
    logit = float(np.clip(0.7*(0.3 - prnu_xcorr_mean)/0.3 + 0.3*(1.0 - min(1.0, residual_snr)), 0, 1))
    return AgentResponse(agent='prnu', logit=logit,
                         evidence={'prnu_xcorr_mean': prnu_xcorr_mean, 'residual_snr': residual_snr},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)), processing_time=0.0).dict()

def compression_agent_task(task_id: str, frames: List[np.ndarray]):
    def blockiness(gray):
        h, w = gray.shape
        v = np.mean(np.abs(gray[:, 8::8].astype(float) - gray[:, 7:-1:8].astype(float))) if w>=9 else 0.0
        h_ = np.mean(np.abs(gray[8::8, :].astype(float) - gray[7:-1:8, :].astype(float))) if h>=9 else 0.0
        return float((v+h_)/2.0)
    all_b, face_b, bg_b = [], [], []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        all_b.append(blockiness(gray))
        face = get_face_roi(f, margin=0)
        if face is not None and face.size > 0:
            fgray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_b.append(blockiness(fgray))
            bg = gray.copy()
            # wipe face region roughly by reprojecting center box (approx)
            h,w = gray.shape
            fh,fw = fgray.shape
            cx, cy = w//2, h//2
            x1 = max(0, cx - fw//2); x2 = min(w, cx + fw//2)
            y1 = max(0, cy - fh//2); y2 = min(h, cy + fh//2)
            bg[y1:y2, x1:x2] = gray.mean()
            bg_b.append(blockiness(bg))
    double_comp = float(np.clip(np.std(all_b)/(np.mean(all_b)+1e-6), 0, 1)) if all_b else 0.0
    qdiff = float(abs(np.mean(face_b or [0]) - np.mean(bg_b or [0])))
    logit = float(np.clip(0.6*double_comp + 0.4*min(1.0, qdiff/5.0), 0, 1))
    return AgentResponse(agent='compression', logit=logit,
                         evidence={'double_comp_prob': double_comp, 'qtable_var_face_bg': qdiff},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)), processing_time=0.0).dict()

def scene_agent_task(task_id: str, frames: List[np.ndarray]):
    if len(frames) < 2:
        return AgentResponse(agent='scene_consistency', logit=0.5, evidence={'inlier_mean':0.0,'cv_mag_mean':0.0,'corr_radial_mean':0.0}, confidence_interval=(0,1), processing_time=0.0).dict()
    inliers, resid_norm, corr_radial, cv_mags = [], [], [], []
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=1200)
        k1,d1 = orb.detectAndCompute(g1,None); k2,d2 = orb.detectAndCompute(g2,None)
        if d1 is None or d2 is None: continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(d1,d2,k=2)
        pts1=[]; pts2=[]
        for m in matches:
            if len(m)<2: continue
            m1,m2 = m
            if m1.distance < 0.75*m2.distance:
                pts1.append(k1[m1.queryIdx].pt); pts2.append(k2[m1.trainIdx].pt)
        if len(pts1) < 8: continue
        H, mask = cv2.findHomography(np.float32(pts1), np.float32(pts2), cv2.RANSAC, 5.0)
        if H is None or mask is None: continue
        inliers.append(float(np.mean(mask.ravel().astype(np.uint8))))
        # residuals
        pts1h = np.hstack([np.float32(pts1), np.ones((len(pts1),1),dtype=np.float32)])
        proj = (H @ pts1h.T).T; proj = proj[:,:2]/proj[:,2:3]
        r = np.linalg.norm(proj - np.float32(pts2), axis=1)
        diag = np.sqrt(g1.shape[0]**2 + g1.shape[1]**2) + 1e-6
        resid_norm.append(float(np.mean(r)/diag))
        # flow stats
        flow = cv2.calcOpticalFlowFarneback(g1,g2,None,0.5,3,15,3,5,1.2,0)
        mag = np.linalg.norm(flow,axis=2)
        cv_mags.append(float(np.std(mag)/(np.mean(mag)+1e-6)))
        # radial corr
        h,w = g1.shape; cx,cy = w/2.0,h/2.0
        pts1a = np.float32(pts1); pts2a = np.float32(pts2)
        motion = np.linalg.norm(pts2a-pts1a,axis=1)
        radii = np.linalg.norm(pts1a - np.array([[cx,cy]],dtype=np.float32),axis=1)
        if len(motion)>=8 and np.std(radii)>1e-6:
            try:
                corr_radial.append(float(np.corrcoef(motion, radii)[0,1]))
            except Exception:
                pass
    if not inliers:
        return AgentResponse(agent='scene_consistency', logit=0.5, evidence={'inlier_mean':0.0,'cv_mag_mean':0.0,'corr_radial_mean':0.0}, confidence_interval=(0,1), processing_time=0.0).dict()
    inlier_mean = float(np.mean(inliers)); resid_mean = float(np.mean(resid_norm))
    cv_mean = float(np.mean(cv_mags)); corr_mean = float(np.mean(corr_radial)) if corr_radial else 0.0
    sus = 0.0
    if inlier_mean < 0.3: sus += 0.45
    if resid_mean  > 0.02: sus += 0.25
    if cv_mean     > 1.0:  sus += 0.20
    if corr_mean   < 0.2:  sus += 0.10
    logit = float(np.clip(sus, 0, 1))
    return AgentResponse(agent='scene_consistency', logit=logit,
                         evidence={'inlier_mean': inlier_mean,'cv_mag_mean': cv_mean,'corr_radial_mean': corr_mean},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)), processing_time=0.0).dict()

def depth_parallax_agent_task(task_id: str, frames: List[np.ndarray]):
    if len(frames) < 2:
        return AgentResponse(agent='depth_parallax', logit=0.5, evidence={'parallax_slope':0.0,'stripe_cv_mean':0.0}, confidence_interval=(0,1), processing_time=0.0).dict()
    corrs, stripe_cv = [], []
    h,w = frames[0].shape[:2]; cx,cy = w/2.0,h/2.0
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1,g2,None,0.5,3,15,3,5,1.2,0)
        mag = np.linalg.norm(flow, axis=2)
        yy,xx = np.indices(mag.shape); r = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        m = mag.ravel(); rf = r.ravel()
        if np.std(m)>1e-6 and np.std(rf)>1e-6:
            try:
                corrs.append(float(np.corrcoef(m, rf)[0,1]))
            except Exception:
                pass
        stripes = np.array_split(mag, 6, axis=1)
        cvs = [float(np.std(s)/(np.mean(s)+1e-6)) for s in stripes if np.mean(s)>1e-6]
        if cvs: stripe_cv.append(float(np.mean(cvs)))
    corr_mean = float(np.mean(corrs)) if corrs else 0.0
    stripe_cv_mean = float(np.mean(stripe_cv)) if stripe_cv else 0.0
    violations = 0.0
    if corr_mean < 0.1: violations += 0.5
    if stripe_cv_mean > 0.9: violations += 0.5
    logit = float(np.clip(violations, 0, 1))
    return AgentResponse(agent='depth_parallax', logit=logit,
                         evidence={'parallax_slope': corr_mean,'stripe_cv_mean': stripe_cv_mean},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)), processing_time=0.0).dict()

def id_drift_agent_task(task_id: str, frames: List[np.ndarray]):
    from skimage.feature import hog
    from sklearn.metrics.pairwise import cosine_similarity
    vecs = []
    for f in frames:
        face = get_face_roi(f, margin=0)
        if face is None:
            continue
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            continue
        patch = cv2.resize(gray,(128,128))
        v = hog(patch, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
        vecs.append(v)
    if len(vecs) < 4:
        return AgentResponse(agent='id_drift', logit=0.5, evidence={'id_cos_std':0.0,'id_jump_rate':0.0}, confidence_interval=(0,1), processing_time=0.0).dict()
    sims = []; jumps=0
    for i in range(1,len(vecs)):
        s = float(cosine_similarity(vecs[i-1].reshape(1,-1), vecs[i].reshape(1,-1))[0,0])
        sims.append(s)
        if s < 0.75: jumps += 1
    id_cos_std = float(np.std(sims)) if sims else 0.0
    id_jump_rate = float(jumps / (len(vecs)-1 + 1e-6))
    logit = float(np.clip(0.6*id_cos_std + 0.4*id_jump_rate, 0, 1))
    return AgentResponse(agent='id_drift', logit=logit,
                         evidence={'id_cos_std': id_cos_std,'id_jump_rate': id_jump_rate},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)), processing_time=0.0).dict()

def gan_fft_agent_task(task_id: str, frames: List[np.ndarray]):
    peaks=[]
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        peaks.append(radial_power_peakiness(g))
    peakiness = float(np.mean(peaks)) if peaks else 2.0
    logit = float(np.clip((peakiness - 2.0)/2.0, 0, 1))
    return AgentResponse(agent='gan_fft', logit=logit,
                         evidence={'peakiness': peakiness},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)), processing_time=0.0).dict()

# ======================
# Weight profiles
# ======================
WEIGHTS_BALANCED = {
    "visual": 0.35, "text": 0.18, "audio": 0.14, "audio_spoof": 0.08, "av_sync": 0.12, "metadata": 0.08, "temporal": 0.05,
    "prnu": 0.12, "compression": 0.08, "scene_consistency": 0.16, "depth_parallax": 0.12, "id_drift": 0.10, "gan_fft": 0.12,
}

WEIGHTS_VISUAL_HEAVY = {
    "visual": 0.55, "av_sync": 0.12, "audio": 0.10, "audio_spoof": 0.06, "text": 0.06, "metadata": 0.06, "temporal": 0.05,
    "prnu": 0.22, "gan_fft": 0.18, "depth_parallax": 0.16, "scene_consistency": 0.14, "id_drift": 0.12, "compression": 0.08,
}

def get_weights(profile: str) -> dict:
    profile = (profile or "visual_heavy").lower()
    base = WEIGHTS_BALANCED if profile == "balanced" else WEIGHTS_VISUAL_HEAVY
    if profile == "accuracy":
        base = {k: WEIGHTS_VISUAL_HEAVY.get(k, 0.0) * RELIABILITY.get(k, 1.0) for k in WEIGHTS_VISUAL_HEAVY}
    s = sum(base.values()) or 1.0
    return {k: v / s for k, v in base.items()}

STOP_HIGH = 0.85
STOP_MID  = 0.70

def fuse(agent_results: Dict[str, dict], weights: dict) -> Tuple[float, List[str]]:
    active = {k:v for k,v in agent_results.items() if isinstance(v, dict) and "logit" in v}
    if not active: return 0.0, []
    total = sum(weights.get(k, 0.0) for k in active) or 1.0
    norm = {k: weights.get(k, 0.0)/total for k in active}
    score = float(sum(norm[k]*float(active[k]["logit"]) for k in active))
    contribs = sorted(active.keys(), key=lambda k: norm[k]*float(active[k]["logit"]), reverse=True)[:3]
    return float(np.clip(score, 0, 1)), contribs

def explanations_map(agent_results: Dict[str, dict]) -> Dict[str, str]:
    exp={}
    for k,v in agent_results.items():
        if not isinstance(v, dict): continue
        parts = [f"logit={v.get('logit',0):.2f}"]
        ev=v.get("evidence",{})
        parts += [f"{ek}={float(ev[ek]):.3f}" for ek in list(ev.keys())[:4]]
        exp[k] = "; ".join(parts)
    return exp

# ======================
# Helpers: gray-zone policy + threshold guard
# ======================
def apply_gray_policy(label: str, rules: List[str], per: dict, score: float, real_cut: float, fake_cut: float, gray_policy: str):
    if label != "uncertain":
        return label, rules
    gp = (gray_policy or "tiebreak").lower()
    if gp == "favor_real":
        rules.append("gray_policy=favor_real")
        return "real", rules
    if gp == "favor_fake":
        rules.append("gray_policy=favor_fake")
        return "fake", rules
    if gp == "uncertain":
        rules.append("gray_policy=uncertain")
        return "uncertain", rules
    # default: tiebreak
    lb, r2 = resolve_uncertain_with_tiebreakers(per, score, real_cut, fake_cut)
    return lb, (rules + r2)

def threshold_guard(label: str, rules: List[str], score: float, real_cut: float, fake_cut: float):
    # Never allow "fake" if fusion score is below fake_cut.
    if label == "fake" and score < fake_cut:
        rules.append("threshold_guard=fake_cut")
        return "real", rules
    return label, rules

# ======================
# Stage runners
# ======================
def run_stage_1(task_id, frames, audio_path, metadata, lang: str) -> Dict[str, dict]:
    res = {
        "visual":   visual_agent_task(task_id, frames),
        "temporal": temporal_agent_task(task_id, frames),
        "metadata": metadata_agent_task(task_id, metadata.dict())
    }
    if metadata.has_audio and audio_path:
        res["audio"]   = audio_agent_task(task_id, audio_path)
        res["text"]    = text_agent_task(task_id, audio_path, lang)
        res["av_sync"] = av_sync_agent_task(task_id, audio_path, frames)
    return res

def run_stage_2(task_id, frames) -> Dict[str, dict]:
    return {
        "prnu":              prnu_agent_task(task_id, frames),
        "compression":       compression_agent_task(task_id, frames),
        "scene_consistency": scene_agent_task(task_id, frames),
        "depth_parallax":    depth_parallax_agent_task(task_id, frames),
        "id_drift":          id_drift_agent_task(task_id, frames),
        "gan_fft":           gan_fft_agent_task(task_id, frames)
    }

# ======================
# Async task store
# ======================
TASKS: Dict[str, dict] = {}

# ======================
# API
# ======================
@controller_app.get("/warmup")
async def warmup():
    if DEMO_MODE:
        return {"ok": True, "msg": "Demo mode: nothing to warm up."}
    try:
        if visual_model is not None:
            _ = visual_model(torch.randn(1,3,380,380, device=TORCH_DEVICE))
        if recognizer is not None:
            _ = recognizer.encode_batch(torch.randn(1,16000*2, device=TORCH_DEVICE))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@controller_app.post("/detect")
async def detect_deepfake(
    file: UploadFile = File(...),
    script: UploadFile | None = File(None),
    mode: str = Query("balanced", description="fast|balanced|strict"),
    fake_cut: float = Query(0.70),
    real_cut: float = Query(0.30),
    force_decision: bool = Query(False, description="Resolve 'uncertain' using tie-breakers"),
    weight_profile: str = Query("visual_heavy", description="balanced|visual_heavy|accuracy"),
    lang: str = Query("auto", description="hi|en|auto (for ASR/NLP routing)"),
    gray_policy: str = Query("tiebreak", description="How to resolve gray zone: tiebreak|favor_real|favor_fake|uncertain")
):
    mode = (mode or "balanced").lower()
    if mode not in {"fast","balanced","strict"}:
        mode = "balanced"
    lang = lang.lower()
    if lang not in {"hi","en","auto"}:
        lang = "auto"

    task_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(video_path, 'wb') as f: f.write(await file.read())
    if script:
        with open(os.path.join(SCRIPT_DIR, f"{task_id}.txt"), 'wb') as sf:
            sf.write(await script.read())

    # DEMO short-circuit
    if DEMO_MODE:
        h = file_sha1_like_upload(video_path)
        cached = load_demo_result_by_hash(h)
        if cached:
            return JSONResponse({"task_id": task_id, "status": "completed", "result": cached})
        default_p = os.path.join(DEMO_CACHE, "default.json")
        if os.path.isfile(default_p):
            with open(default_p, "r", encoding="utf-8") as f:
                return JSONResponse({"task_id": task_id, "status": "completed", "result": json.load(f)})
        # fallback canned
        return JSONResponse({"task_id": task_id, "status": "completed",
                             "result": {"version":"cascade_v1_multilingual","mode":"demo/visual_heavy",
                                        "label":"fake","deepfake_score":0.76,"confidence":[0.70,0.82],
                                        "stage_reached":1,"stages_run":[1],
                                        "top_contributors":["visual","gan_fft","av_sync"],
                                        "agent_evidence":{},"explanation":"Demo fallback result",
                                        "explanation_nl":"Demo mode: returning canned response.",
                                        "decision_rules":[]}})

    data = process_video(video_path)
    frames = data['keyframes']; audio_path = data['audio_path']; metadata = data['metadata']

    stages_run = []; per = {}
    weights = get_weights(weight_profile)

    # ---- Stage 1
    s1 = run_stage_1(task_id, frames, audio_path, metadata, lang); per.update(s1); stages_run.append(1)
    score1, contrib1 = fuse(per, weights)
    label1 = label_from_score(score1, real_cut, fake_cut)
    rules: List[str] = []

    # Apply gray-policy if in gray zone
    if label1 == "uncertain":
        # only resolve if strict/force OR gray_policy explicitly asks to
        if (mode == "strict" or force_decision or gray_policy != "uncertain"):
            label1, rules = apply_gray_policy(label1, rules, per, score1, real_cut, fake_cut, gray_policy)

    # Threshold guard (never fake below fake_cut)
    label1, rules = threshold_guard(label1, rules, score1, real_cut, fake_cut)

    # Early stop
    if mode == "fast" or score1 >= STOP_HIGH or (score1 >= STOP_MID and mode != "strict"):
        result = DetectionResult(
            version="cascade_v1_multilingual",
            mode=f"{mode}/{weight_profile}",
            label=label1,
            deepfake_score=score1,
            confidence=(max(0, score1-0.06), min(1, score1+0.06)),
            stage_reached=1,
            stages_run=stages_run,
            top_contributors=contrib1,
            agent_evidence=explanations_map(per),
            explanation=f"Stage1 fusion={score1:.2f} → {label1}. Early-stop."
        )
        explanation_nl = build_natural_language_explanation(per, label1, score1, contrib1)
        return JSONResponse({
          "task_id": task_id,
          "status": "completed",
          "result": {**result.dict(), "explanation_nl": explanation_nl, "decision_rules": rules}
        })

    # ---- Stage 2
    s2 = run_stage_2(task_id, frames); per.update(s2); stages_run.append(2)
    score2, contrib2 = fuse(per, weights)
    label2 = label_from_score(score2, real_cut, fake_cut)
    rules = []

    if label2 == "uncertain":
        if (mode == "strict" or force_decision or gray_policy != "uncertain"):
            label2, rules = apply_gray_policy(label2, rules, per, score2, real_cut, fake_cut, gray_policy)

    # Threshold guard again
    label2, rules = threshold_guard(label2, rules, score2, real_cut, fake_cut)

    result = DetectionResult(
        version="cascade_v1_multilingual",
        mode=f"{mode}/{weight_profile}",
        label=label2,
        deepfake_score=score2,
        confidence=(max(0, score2-0.06), min(1, score2+0.06)),
        stage_reached=2,
        stages_run=stages_run,
        top_contributors=contrib2,
        agent_evidence=explanations_map(per),
        explanation=f"Stage2 fusion={score2:.2f} → {label2}. Added heavier visual checks."
    )
    explanation_nl = build_natural_language_explanation(per, label2, score2, contrib2)
    return JSONResponse({
      "task_id": task_id,
      "status": "completed",
      "result": {**result.dict(), "explanation_nl": explanation_nl, "decision_rules": rules}
    })

# ============ Async variant for live demo with progress ============
def _heavy_job(task_id: str, video_path: str, params: dict):
    try:
        data = process_video(video_path)
        frames = data['keyframes']; audio_path = data['audio_path']; metadata = data['metadata']
        weights = get_weights(params.get("weight_profile","visual_heavy"))
        real_cut = params.get("real_cut", 0.30)
        fake_cut = params.get("fake_cut", 0.70)
        mode = params.get("mode","balanced")
        gray_policy = params.get("gray_policy","tiebreak")
        lang = params.get("lang","auto")

        per = {}
        s1 = run_stage_1(task_id, frames, audio_path, metadata, lang); per.update(s1)
        score1, contrib1 = fuse(per, weights)
        label1 = label_from_score(score1, real_cut, fake_cut)
        rules: List[str] = []

        if label1 == "uncertain":
            if (mode == "strict" or params.get("force_decision") or gray_policy != "uncertain"):
                label1, rules = apply_gray_policy(label1, rules, per, score1, real_cut, fake_cut, gray_policy)
        label1, rules = threshold_guard(label1, rules, score1, real_cut, fake_cut)

        if score1 >= STOP_HIGH or (score1 >= STOP_MID and mode != "strict"):
            res = {
                "version":"cascade_v1_multilingual","mode":f"{mode}/{params.get('weight_profile')}",
                "label":label1,"deepfake_score":score1,"confidence":[max(0,score1-0.06),min(1,score1+0.06)],
                "stage_reached":1,"stages_run":[1],"top_contributors":contrib1,
                "agent_evidence":explanations_map(per),
                "explanation":f"Stage1 fusion={score1:.2f} → {label1}. Early-stop.",
                "decision_rules": rules
            }
            res["explanation_nl"] = build_natural_language_explanation(per, label1, score1, contrib1)
            TASKS[task_id] = {"status":"completed","result":res}
            return

        s2 = run_stage_2(task_id, frames); per.update(s2)
        score2, contrib2 = fuse(per, weights)
        label2 = label_from_score(score2, real_cut, fake_cut)
        rules = []
        if label2 == "uncertain":
            if (mode == "strict" or params.get("force_decision") or gray_policy != "uncertain"):
                label2, rules = apply_gray_policy(label2, rules, per, score2, real_cut, fake_cut, gray_policy)
        label2, rules = threshold_guard(label2, rules, score2, real_cut, fake_cut)

        res = {
            "version":"cascade_v1_multilingual","mode":f"{mode}/{params.get('weight_profile')}",
            "label":label2,"deepfake_score":score2,"confidence":[max(0,score2-0.06),min(1,score2+0.06)],
            "stage_reached":2,"stages_run":[1,2],"top_contributors":contrib2,
            "agent_evidence":explanations_map(per),
            "explanation":f"Stage2 fusion={score2:.2f} → {label2}. Added heavier visual checks.",
            "decision_rules": rules
        }
        res["explanation_nl"] = build_natural_language_explanation(per, label2, score2, contrib2)
        TASKS[task_id] = {"status":"completed","result":res}
    except Exception as e:
        TASKS[task_id] = {"status":"error","error":str(e)}

@controller_app.post("/detect_async")
async def detect_async(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Query("balanced"),
    fake_cut: float = Query(0.70),
    real_cut: float = Query(0.30),
    weight_profile: str = Query("visual_heavy"),
    lang: str = Query("auto"),
    force_decision: bool = Query(False),
    gray_policy: str = Query("tiebreak", description="tiebreak|favor_real|favor_fake|uncertain")
):
    task_id = uuid.uuid4().hex
    video_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(video_path, 'wb') as f: f.write(await file.read())

    if DEMO_MODE:
        h = file_sha1_like_upload(video_path)
        cached = load_demo_result_by_hash(h)
        if cached:
            TASKS[task_id] = {"status":"completed","result":cached}
            return {"task_id": task_id}

    TASKS[task_id] = {"status":"running","progress":"queued"}
    bg.add_task(_heavy_job, task_id, video_path, {
        "mode":mode, "fake_cut":fake_cut, "real_cut":real_cut,
        "weight_profile":weight_profile, "lang":lang,
        "force_decision": force_decision, "gray_policy": gray_policy
    })
    return {"task_id": task_id}

@controller_app.get("/task/{task_id}")
async def task_status(task_id: str):
    return TASKS.get(task_id, {"status":"unknown"})

# ======================
# Main
# ======================
if __name__ == "__main__":
    import uvicorn
    print("MediaPipe available:", MP_OK, "| dlib available:", DLIB_OK, "| DEMO_MODE:", DEMO_MODE)
    uvicorn.run(controller_app, host="0.0.0.0", port=8000)
