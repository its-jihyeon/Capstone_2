from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import traceback

# --- 설정 ---
MODEL_DIR = r"C:\HEAT_project\HEAT_final_model"
DEFAULT_THRESHOLD = 0.50
MAX_LENGTH = 256

# --- 모델 로드 ---
print("🔹 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_labels = int(getattr(model.config, "num_labels", 2))
id2label = getattr(model.config, "id2label", {}) or {}
label2id = getattr(model.config, "label2id", {}) or {}

def _find_malicious_idx(label2id_dict):
    if isinstance(label2id_dict, dict):
        for k, v in label2id_dict.items():
            kl = str(k).lower()
            if any(x in kl for x in ["malicious", "phishing", "phish", "attack", "spam", "fraud"]):
                return int(v)
    return 1  # fallback

MAL_IDX = _find_malicious_idx(label2id)
print(f"🧭 num_labels={num_labels}, MAL_IDX={MAL_IDX}")
print("✅ 모델 로드 완료")

# --- App/CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return "캡스톤 웹 테스트"

# --- 스키마 ---
class URLRequest(BaseModel):
    url: str

class URLResponse(BaseModel):
    url: str
    is_malicious: bool
    confidence_score: float  

# --- 캐시 ---
def _normalize(u: str) -> str:
    return (u or "").strip().lower()

class _LRU:
    def __init__(self, cap: int = 1024):
        self.cap = cap
        self.od = OrderedDict()
    def get(self, k):
        v = self.od.get(k)
        if v is not None:
            self.od.move_to_end(k)
        return v
    def set(self, k, v):
        self.od[k] = v
        self.od.move_to_end(k)
        if len(self.od) > self.cap:
            self.od.popitem(last=False)

_cache = _LRU(cap=1024)

# --- 추론 ---
@torch.no_grad()
def _infer_once(raw_url: str):
    inputs = tokenizer(raw_url, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits

    if logits.shape[-1] == 2:
        probs = torch.softmax(logits, dim=-1)[0]
        prob_mal = float(probs[min(MAL_IDX, probs.shape[0]-1)].item())
        is_mal = prob_mal >= DEFAULT_THRESHOLD
        return is_mal, round(prob_mal, 4)

    if logits.shape[-1] == 1:
        prob_mal = float(torch.sigmoid(logits.view(-1)[0]).item())
        is_mal = prob_mal >= DEFAULT_THRESHOLD
        return is_mal, round(prob_mal, 4)

    probs = torch.softmax(logits, dim=-1)[0]
    prob_mal = float(probs[min(MAL_IDX, probs.shape[0]-1)].item())
    is_mal = prob_mal >= DEFAULT_THRESHOLD
    return is_mal, round(prob_mal, 4)

async def _predict_model(raw_url: str):
    return await asyncio.to_thread(_infer_once, raw_url)

# --- 엔드포인트 ---
@app.post("/analyze", response_model=URLResponse)
async def analyze_url(request: URLRequest):
    raw_input = request.url
    key = _normalize(raw_input)

    cached = _cache.get(key)
    if cached is None:
        try:
            is_mal, prob_mal = await asyncio.wait_for(_predict_model(raw_input), timeout=20.0)
            _cache.set(key, (is_mal, prob_mal))
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="모델 추론 타임아웃")
        except Exception as e:
            print("❌ Inference error:\n", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"모델 오류: {e}")
    else:
        is_mal, prob_mal = cached

    return {"url": raw_input, "is_malicious": is_mal, "confidence_score": prob_mal}

# 디버그(선택): 악성 확률만 확인용
@app.post("/analyze_debug")
async def analyze_debug(request: URLRequest):
    is_mal, prob_mal = await _predict_model(request.url)
    return {
        "url": request.url,
        "is_malicious": is_mal,
        "malicious_probability": prob_mal,
        "num_labels": num_labels,
        "label2id": label2id,
        "MAL_IDX": MAL_IDX,
    }
