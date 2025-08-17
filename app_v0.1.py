import os
import uuid
import json
import numpy as np
import cv2
import librosa
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from speechbrain.pretrained import SpeakerRecognition
import torchaudio

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, parse_obj_as

from celery import Celery, group
from celery.result import AsyncResult

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import faiss
from torch.quantization import quantize_dynamic

# ======================
# Pipeline Initializations
# ======================
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=0
)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0
)
zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)
paraphrase_model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device='cuda')

recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb",
    run_opts={"device": "cuda"} 
)

def load_efficientnet():
    """Load EfficientNet-B4 model and its default preprocessing pipeline from torchvision."""
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    model = efficientnet_b4(weights=weights)
    model.eval()
    # Use the built-in transforms from the weights
    preprocess = weights.transforms()
    return model, preprocess

visual_model, visual_preprocess = load_efficientnet()
visual_model = quantize_dynamic(visual_model, {torch.nn.Linear}, dtype=torch.qint8)
visual_model = visual_model.to("cuda")   # Move to GPU


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
# Celery Setup
# ======================
celery_app = Celery(
    'deepfake_detection',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# ======================
# FastAPI App & CORS
# ======================
controller_app = FastAPI()
controller_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
# Helper: Load Script
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
    cap = cv2.VideoCapture(file_path)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('utf-8')
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    resolution = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(frame_rate) == 0:
            keyframes.append(frame)
        count += 1
    cap.release()

    audio_path = f"{file_path}_audio.wav"
    os.system(f"ffmpeg -i \"{file_path}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{audio_path}\"")
    has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
    language = "en" if has_audio else None

    metadata = VideoMetadata(
        codec=codec,
        duration=duration,
        resolution=resolution,
        frame_rate=frame_rate,
        has_audio=has_audio,
        has_visual=True,
        language=language,
        source='file'
    )
    return {'metadata': metadata, 'keyframes': keyframes, 'audio_path': audio_path if has_audio else None}

# ======================
# Agent Implementations
# ======================
@celery_app.task
def visual_agent_task(task_id: str, video_path: str, metadata: dict):
    data = process_video(video_path)
    frames = data['keyframes'][:8]
    scores = []
    for frame in frames:
        img = visual_preprocess(frame).unsqueeze(0)
        with torch.no_grad():
            logits = visual_model(img)
            prob = F.softmax(logits, dim=1)[0].max().item()
        scores.append(1 - prob)
    avg_score = float(np.mean(scores))
    return AgentResponse(
        agent='visual',
        logit=avg_score,
        evidence={'avg_anomaly': avg_score},
        confidence_interval=(max(0, avg_score-0.05), min(1, avg_score+0.05)),
        processing_time=0.0
    ).dict()

@celery_app.task
def audio_agent_task(task_id: str, audio_path: str):
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    
    # Get embeddings
    embeddings = recognizer.encode_batch(waveform)
    
    # Simple anomaly score (distance from average human embedding)
    mean_embedding = torch.zeros_like(embeddings)
    score = torch.norm(embeddings - mean_embedding).item()
    
    # Normalize score to [0,1]
    logit = float(score / (score + 1))
    
    return AgentResponse(
        agent='audio',
        logit=logit,
        evidence={'embedding_norm': score},
        confidence_interval=(max(0, logit - 0.1), min(1, logit + 0.1)),
        processing_time=0.0
    ).dict()

@celery_app.task
def text_agent_task(task_id: str, video_path: str, metadata: dict):
    data = process_video(video_path)
    transcript = asr(data['audio_path'])['text']
    sent = sentiment_pipe(transcript)[0]
    sentiment_score = sent['score'] * (1 if sent['label']=='POSITIVE' else -1)
    zero = zero_shot(transcript, candidate_labels=['consistent','inconsistent'])
    incons = float(zero['scores'][zero['labels'].index('inconsistent')])
    original = load_original_script(task_id)
    if original:
        emb1, emb2 = paraphrase_model.encode([transcript, original], convert_to_tensor=True)
        dev = 1 - util.cos_sim(emb1, emb2).item()
    else:
        dev = 0.0
    logit = 0.6 * incons + 0.4 * abs(dev)
    return AgentResponse(
        agent='text',
        logit=logit,
        evidence={
            'sentiment_score': sentiment_score,
            'context_inconsistency': incons,
            'script_deviation': dev
        },
        confidence_interval=(max(0, logit-0.1), min(1, logit+0.1)),
        processing_time=0.0
    ).dict()

@celery_app.task
def metadata_agent_task(task_id: str, video_path: str, metadata: dict):
    meta = metadata
    codec = meta.get('codec', '')
    allowed = ['avc1','vp09']
    codec_anomaly = 1.0 if codec not in allowed else 0.0
    return AgentResponse(
        agent='metadata',
        logit=codec_anomaly,
        evidence={'codec_anomaly': codec_anomaly},
        confidence_interval=(0,1),
        processing_time=0.0
    ).dict()

@celery_app.task
def temporal_agent_task(task_id: str, video_path: str, metadata: dict):
    frames = process_video(video_path)['keyframes'][:5]
    diffs = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.mean(np.abs(curr.astype(float)-prev.astype(float)))))
        prev = curr
    jitter = float(np.mean(diffs) / 255)
    return AgentResponse(
        agent='temporal',
        logit=jitter,
        evidence={'motion_jitter': jitter},
        confidence_interval=(0,1),
        processing_time=0.0
    ).dict()

# ======================
# Fusion & Audit
# ======================
def fusion_core(task_id: str, agent_results: Dict[str, dict]) -> DetectionResult:
    logits = np.array([v['logit'] for v in agent_results.values()])
    score = float(np.mean(logits))
    label = 'fake' if score > 0.5 else 'real'
    explanations = {k: f"score={v['logit']:.2f}" for k,v in agent_results.items()}
    return DetectionResult(
        label=label,
        deepfake_score=score,
        confidence=(max(0, score-0.05), min(1, score+0.05)),
        agent_evidence=explanations,
        explanation=f"Final score {score:.2f} ({label})"
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
# Celery Task: Dispatch Agents
# ======================
@celery_app.task(bind=True)
def dispatch_agents(self, task_id: str, file_path: str, metadata: dict):
    """Select and run all agent tasks in parallel, then fuse results."""
    from pydantic import parse_obj_as
    md = parse_obj_as(VideoMetadata, metadata)

    names = []
    sigs = []
    if md.has_visual:
        names.append('visual')
        sigs.append(visual_agent_task.s(task_id, file_path, metadata))
    if md.has_audio:
        names.append('audio')
        sigs.append(audio_agent_task.s(task_id, file_path, metadata))
        if md.language in ['en', 'hi']:
            names.append('text')
            sigs.append(text_agent_task.s(task_id, file_path, metadata))

    # Always run metadata and temporal agents
    names.extend(['metadata', 'temporal'])
    sigs.append(metadata_agent_task.s(task_id, file_path, metadata))
    sigs.append(temporal_agent_task.s(task_id, file_path, metadata))

    # Execute tasks in parallel
    job = group(sigs)
    results_list = job().get()

    # Map results back to agent names
    results = {name: output for name, output in zip(names, results_list)}

    # Fusion and audit
    final = fusion_core(task_id, results)
    log_audit(task_id, results, final)
    return final

# ======================
# FastAPI Endpoints
# ======================
@controller_app.post("/detect")
async def detect_deepfake(
    file: UploadFile = File(...),
    script: UploadFile | None = File(None)
):
    """Upload video (and optional script), enqueue deepfake detection task"""
    task_id = str(uuid.uuid4())
    # Save video
    video_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(video_path, 'wb') as f:
        f.write(await file.read())
    # Save script
    if script:
        script_path = os.path.join(SCRIPT_DIR, f"{task_id}.txt")
        with open(script_path, 'wb') as sf:
            sf.write(await script.read())
    # Process metadata and enqueue
    metadata = process_video(video_path)['metadata']
    dispatch_agents.apply_async(
        args=[task_id, video_path, metadata.dict()],
        task_id=task_id
    )
    return JSONResponse({"task_id": task_id, "status": "processing"})

@controller_app.get("/status/{task_id}")
def get_status(task_id: str):
    """Poll task status and return result when complete"""
    res = AsyncResult(task_id, app=celery_app)
    if res.state in ("PENDING", "STARTED"):
        return {"status": "processing"}
    if res.state == "SUCCESS":
        return {"status": "completed", "result": res.result}
    # Error states
    return {"status": res.state.lower(), "error": str(res.result)}

# ======================
# Main: Spawn Agents + Run Controller
# ======================
def start_agent(name, port):
    import uvicorn
    agent_app = globals()[f"{name}_agent_app"]
    uvicorn.run(agent_app, host='0.0.0.0', port=port)

if __name__ == "__main__":
    import multiprocessing, uvicorn
    multiprocessing.freeze_support()
    # Start each agent in its own process
    for name, port in [('visual',8001),('audio',8002),('text',8003),('metadata',8004),('temporal',8005)]:
        multiprocessing.Process(target=start_agent, args=(name, port)).start()
    # Start controller app
    uvicorn.run(controller_app, host='0.0.0.0', port=8000, reload=True)
