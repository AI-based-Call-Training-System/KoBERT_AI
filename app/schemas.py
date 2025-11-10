# kobert_model/app/schemas.py
# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class SimilarityRequest(BaseModel):
    a: str
    b: str

class SimilarityResponse(BaseModel):
    score: float

class ClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    return_probs: bool = True

class ClassifyResponse(BaseModel):
    labels: List[str]
    scores: List[List[float]]

# 평가용 (goal, coherence)
class HistoryTurn(BaseModel):
    role: str
    content: str

class EvaluateRequest(BaseModel):
    history: List[HistoryTurn]
    userId: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class EvaluateResponse(BaseModel):
    goal_score: float
    coherence_score: float
    goal_probs: List[float]
    coherence_probs: List[float]
    goal_label: str
    coherence_label: str
