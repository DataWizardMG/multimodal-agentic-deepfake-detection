import os
import uuid
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import dlib
from datetime import datetime
from typing import Dict, Optional, Tuple
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from speechbrain.inference import SpeakerRecognition
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# ======================
# Load Models Globally
# ======================
print(">>> Loading models, please wait...")

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0)
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
zero_shot = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=0)
paraphrase_model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device='cuda')
recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb",
    run_opts={"device": "cuda"}
)
weights = EfficientNet_B4_Weights.IMAGENET1K_V1
visual_model = efficientnet_b4(weights=weights).to("cuda").eval()
visual_preprocess = weights.transforms()

print(">>> All models loaded and ready <<<")

# ======================
# Directories & Config
# ======================
MODEL_CACHE_DIR = "./model_cache"
VECTOR_DB_PATH = "./vector_db/evidence_db.faiss"
SCRIPT_DIR = "./scripts"
UPLOAD_DIR = "./uploads"
LOG_DIR = "./audit_logs"

for d in [MODEL_CACHE_DIR, os.path.dirname(VECTOR_DB_PATH), SCRIPT_DIR, UPLOAD_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ======================
# FastAPI App & CORS
# ======================
controller_app = FastAPI()
controller_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Pydantic Models
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

class AgentResponse(BaseModel):
    agent: str
    logit: float
    evidence: Dict[str, float]
    confidence_interval: Tuple[float, float]
    processing_time: float

class DetectionResult(BaseModel):
    label: str
    deepfake_score: float
    confidence: Tuple[float, float]
    agent_evidence: Dict[str, str]
    explanation: str

# ======================
# Helpers
# ======================
def load_original_script(task_id: str) -> Optional[str]:
    path = os.path.join(SCRIPT_DIR, f"{task_id}.txt")
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ======================
# Video Processing
# ======================
def process_video(file_path: str) -> Dict:
    # Detect audio stream
    has_audio = bool(os.popen(
        f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name "
        f"-of default=noprint_wrappers=1:nokey=1 \"{file_path}\""
    ).read().strip())
    audio_path = None
    if has_audio:
        audio_path = f"{file_path}_audio.wav"
        os.system(f"ffmpeg -y -i \"{file_path}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{audio_path}\"")

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    frames, count = [], 0
    while cap.isOpened() and count < int(fps * 8):
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(fps) == 0:
            frames.append(frame)
        count += 1
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    codec = os.popen(
        f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name "
        f"-of default=noprint_wrappers=1:nokey=1 \"{file_path}\""
    ).read().strip()
    duration = count / fps

    metadata = VideoMetadata(
        codec=codec,
        duration=duration,
        resolution=(width, height),
        frame_rate=fps,
        has_audio=has_audio,
        has_visual=True,
        language='en' if has_audio else None,
        source='file'
    )
    return {'metadata': metadata, 'keyframes': frames, 'audio_path': audio_path}

# ======================
# Agent Tasks
# ======================
def visual_agent_task(task_id: str, frames: list):
    imgs = []
    for f in frames:
        img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        imgs.append(visual_preprocess(pil_img).unsqueeze(0))
    imgs = torch.cat(imgs, dim=0).to("cuda")
    with torch.no_grad():
        logits = visual_model(imgs)
        probs = torch.softmax(logits, dim=1).max(dim=1).values
    score = float((1 - probs).mean().item())
    return AgentResponse(agent='visual', logit=score,
                         evidence={'avg_anomaly': score},
                         confidence_interval=(max(0, score-0.05), min(1, score+0.05)),
                         processing_time=0.0).dict()

def audio_agent_task(task_id: str, audio_path: str):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    emb = recognizer.encode_batch(waveform.to("cuda"))
    norm = torch.norm(emb).item()
    logit = float(norm / (norm + 1))
    return AgentResponse(agent='audio', logit=logit,
                         evidence={'embedding_norm': norm},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)),
                         processing_time=0.0).dict()

def text_agent_task(task_id: str, audio_path: str):
    transcript = asr(audio_path)['text']
    sent = sentiment_pipe(transcript)[0]
    sentiment_score = sent['score'] * (1 if sent['label']=='POSITIVE' else -1)
    zero = zero_shot(transcript, candidate_labels=['consistent','inconsistent'])
    incons = float(zero['scores'][zero['labels'].index('inconsistent')])
    original = load_original_script(task_id)
    dev = 0.0
    if original:
        emb1, emb2 = paraphrase_model.encode([transcript, original], convert_to_tensor=True)
        dev = 1 - util.cos_sim(emb1, emb2).item()
    logit = 0.6 * incons + 0.4 * abs(dev)
    return AgentResponse(agent='text', logit=logit,
                         evidence={'sentiment_score': sentiment_score,
                                   'context_inconsistency': incons,
                                   'script_deviation': dev},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)),
                         processing_time=0.0).dict()

def metadata_agent_task(task_id: str, metadata: dict):
    codec = metadata.get('codec','')
    anomaly = 1.0 if codec not in ['avc1','vp09'] else 0.0
    return AgentResponse(agent='metadata', logit=anomaly,
                         evidence={'codec_anomaly': anomaly},
                         confidence_interval=(0,1),
                         processing_time=0.0).dict()

def temporal_agent_task(task_id: str, frames: list):
    diffs = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.mean(np.abs(curr.astype(float)-prev.astype(float)))))
        prev = curr
    jitter = float(np.mean(diffs)/255)
    return AgentResponse(agent='temporal', logit=jitter,
                         evidence={'motion_jitter': jitter},
                         confidence_interval=(0,1),
                         processing_time=0.0).dict()

def av_sync_agent_task(task_id: str, audio_path: str, frames: list):
    # audio energy
    y, sr = librosa.load(audio_path, sr=16000)
    energy = float((y**2).mean())
    # mouth-open ratio
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    ratios = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:
            shape = predictor(gray, faces[0])
            top = shape.part(51).y
            bot = shape.part(57).y
            ratios.append((bot - top) / float(f.shape[0]))
    sync_score = 1 - abs(energy * np.mean(ratios)) if ratios else 0.0
    logit = float(sync_score / (sync_score + 1))
    return AgentResponse(agent='av_sync', logit=logit,
                         evidence={'sync_score': sync_score},
                         confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)),
                         processing_time=0.0).dict()

# ======================
# Fusion & Audit
# ======================
def fusion_core(task_id: str, agent_results: Dict[str, dict]) -> DetectionResult:
    weights = {
        "visual":   0.30,
        "text":     0.25,
        "audio":    0.20,
        "av_sync":  0.10,
        "metadata": 0.10,
        "temporal": 0.05
    }
    active = {k: v for k,v in agent_results.items() if k in weights}
    total = sum(weights[k] for k in active)
    norm = {k: weights[k]/total for k in active}
    score = sum(norm[k] * active[k]['logit'] for k in active)
    label = "fake" if score > 0.7 else "real"
    explanations = {k: f"score={v['logit']:.2f}" for k,v in agent_results.items()}
    return DetectionResult(
        label=label,
        deepfake_score=score,
        confidence=(max(0, score-0.05), min(1, score+0.05)),
        agent_evidence=explanations,
        explanation=f"Weighted score {score:.2f} ({label})"
    )

def log_audit(task_id: str, agent_results: Dict[str, dict], final: DetectionResult):
    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'agents': agent_results,
        'final': final.dict()
    }
    with open(os.path.join(LOG_DIR, f"{task_id}.json"), 'w') as f:
        json.dump(entry, f, indent=2)

# ======================
# API Endpoints
# ======================
@controller_app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...), script: UploadFile | None = File(None)):
    task_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(video_path, 'wb') as f:
        f.write(await file.read())

    if script:
        script_path = os.path.join(SCRIPT_DIR, f"{task_id}.txt")
        with open(script_path, 'wb') as sf:
            sf.write(await script.read())

    data = process_video(video_path)
    frames = data['keyframes']
    audio_path = data['audio_path']
    metadata = data['metadata']

    results = {
        'visual':   visual_agent_task(task_id, frames),
        'temporal': temporal_agent_task(task_id, frames),
        'metadata': metadata_agent_task(task_id, metadata.dict())
    }

    if metadata.has_audio and audio_path:
        results['audio']   = audio_agent_task(task_id, audio_path)
        results['text']    = text_agent_task(task_id, audio_path)
        results['av_sync'] = av_sync_agent_task(task_id, audio_path, frames)

    final = fusion_core(task_id, results)
    log_audit(task_id, results, final)
    return JSONResponse({"task_id": task_id, "status": "completed", "result": final.dict()})

@controller_app.get("/status/{task_id}")
def get_status(task_id: str):
    log_file = os.path.join(LOG_DIR, f"{task_id}.json")
    if os.path.exists(log_file):
        with open(log_file) as f:
            data = json.load(f)
        return {"status": "completed", "result": data['final']}
    return {"status": "not_found"}

# ======================
# Main
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(controller_app, host="0.0.0.0", port=8000)
