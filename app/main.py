# kobert_model/app/main.py
# -*- coding: utf-8 -*-

import os
from .kobert_eval import router as kobert_router
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv(override=True)

# 스키마, 서비스
from .schemas import (
    EmbedRequest, EmbedResponse,
    SimilarityRequest, SimilarityResponse,
    ClassifyRequest, ClassifyResponse,
    EvaluateRequest, EvaluateResponse,
)
from .services import (
    ModelService, MultiModelService, mongo_dialog_to_evaluate_payload
)

# FastAPI 앱 생성
app = FastAPI(title="KoBERT FastAPI", version="0.3.0")
app.include_router(kobert_router)

# 모델 서비스 초기화
svc = ModelService()
multi = MultiModelService()

# CORS 설정
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/health")
def health():
    return {"status": "ok", "multi_loaded": True}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    try:
        return {"embeddings": svc.embed(req.texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    try:
        return {"score": svc.similarity(req.a, req.b)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    try:
        labels, scores = svc.classify(req.texts, return_probs=req.return_probs)
        return {"labels": labels, "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# /evaluate : 일반 history 입력
@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    try:
        res = multi.evaluate_dialog([t.model_dump() for t in req.history])
        multi.save_to_mongo(res, userId=req.userId, meta=req.meta)
        return {
            "goal_score": res["goal_score"],
            "coherence_score": res["coherence_score"],
            "goal_probs": res["goal_probs"],
            "coherence_probs": res["coh_probs"],
            "goal_label": res["goal_label"],
            "coherence_label": res["coherence_label"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# /evaluate_mongo : Mongo 문서 평가
@app.post("/evaluate_mongo", response_model=EvaluateResponse)
def evaluate_mongo(doc: dict):
    try:
        hist, meta = mongo_dialog_to_evaluate_payload(doc)
        res = multi.evaluate_dialog(hist)
        user_id = meta.get("userId") or doc.get("userId")
        multi.save_to_mongo(res, userId=user_id, meta=meta)
        return {
            "goal_score": res["goal_score"],
            "coherence_score": res["coherence_score"],
            "goal_probs": res["goal_probs"],
            "coherence_probs": res["coh_probs"],
            "goal_label": res["goal_label"],
            "coherence_label": res["coherence_label"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
