# kobert_model/app/services.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

# 공용 ENV 유틸
def _env_stripped(key: str) -> Optional[str]:
    v = os.getenv(key)
    if v is None:
        return None
    return v.split("#", 1)[0].strip() or None

def _env_int(key: str) -> Optional[int]:
    v = _env_stripped(key)
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return None

def _env_json(key: str):
    v = _env_stripped(key)
    if not v:
        return None
    try:
        return json.loads(v)
    except Exception:
        return None

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """간단 getter (빈 문자열/공백이면 default 반환)"""
    v = os.getenv(key)
    return v.strip() if v and v.strip() != "" else default


# 기존: 단일 모델 서비스(임베딩/단일 분류)
class ModelService:
    def __init__(self):
        # 순환 방지: 이곳에서 지연 import
        from .model import KoBERTEmbedding, KoBERTClassifier  # noqa: WPS433

        name = _env_stripped("MODEL_NAME_OR_PATH") or "skt/kobert-base-v1"
        ckpt = _env_stripped("FINETUNED_CKPT") or ""
        self.labels = _env_json("LABELS_JSON")
        num_labels = _env_int("NUM_LABELS")

        if ckpt:
            # 분류 모드
            self.clf = KoBERTClassifier(ckpt_or_name=ckpt, num_labels=num_labels, labels=self.labels)
            self.emb = None
            self.mode = "classification"
        else:
            # 임베딩 모드
            self.emb = KoBERTEmbedding(model_name_or_path=name)
            self.clf = None
            self.mode = "embedding"

        print(f"[ModelService] mode={self.mode} model={name} ckpt={ckpt or '-'}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.emb is None:
            raise RuntimeError("Server is in classification mode; embedding disabled.")
        return self.emb.embed(texts)

    def classify(self, texts: List[str], return_probs: bool = True):
        if self.clf is None:
            raise RuntimeError("Server is in embedding mode; classification disabled.")
        scores = self.clf.predict(texts, return_probs=return_probs)
        return (self.labels or []), scores

    def similarity(self, a: str, b: str) -> float:
        """코사인 유사도"""
        vecs = self.embed([a, b])
        v1 = np.asarray(vecs[0], dtype=np.float32)
        v2 = np.asarray(vecs[1], dtype=np.float32)
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / denom)


# MongoDB 저장소 연결 (베스트-에포트: 연결 실패 시 비활성)
class ScoreRepository:
    def __init__(self):
        uri = _env("MONGODB_URI", "mongodb://localhost:27017")
        dbname = _env("DB_NAME", "Call-Training-DB")
        coll_name = _env("SCORE_COLLECTION", "Scores")

        self.db = None
        self.coll = None
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=1500)
            client.admin.command("ping")
            self.db = client[dbname]
            self.coll = self.db[coll_name]
            self.coll.create_index([("createdAt", ASCENDING)])
            self.coll.create_index([("userId", ASCENDING)])
        except Exception as e:
            print(f"[Mongo] disabled (no connection): {e}")

    def insert(self, doc: Dict[str, Any]) -> str:
        if self.coll is None:
            # 저장 비활성 환경에서는 조용히 스킵
            return ""
        try:
            res = self.coll.insert_one(doc)
            return str(res.inserted_id)
        except PyMongoError as e:
            print(f"[Mongo] insert skipped: {e}")
            return ""


# 대화 텍스트 직렬화
def _linearize(history: List[Dict]) -> str:
    parts = []
    for t in history or []:
        role = str(t.get("role", "")).upper().strip()
        content = str(t.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return " [SEP] ".join(parts) if parts else "[EMPTY]"


# 멀티모델(Goal / Coherence) 서비스
class MultiModelService:
    def __init__(self):
        from .model import KoBERTClassifier  # 지연 import

        goal_ckpt = _env("GOAL_CKPT_DIR", "./data/ckpt_goal")
        coh_ckpt  = _env("COH_CKPT_DIR", "./data/ckpt_coherence")
        if not goal_ckpt or not coh_ckpt:
            raise RuntimeError("GOAL_CKPT_DIR / COH_CKPT_DIR 경로가 설정되지 않았습니다.")

        # 반환용 표기(고정)
        self.goal_labels = ["failure", "success"]
        self.coh_labels  = ["incoherent", "coherent"]

        # per-task max_len (없으면 MAX_LEN 또는 256)
        def _pick_len(specific_key: str, fallback_key: str = "MAX_LEN", default: int = 256) -> int:
            v = _env_int(specific_key)
            if v is not None:
                return v
            v2 = _env_int(fallback_key)
            return v2 if v2 is not None else default

        goal_max_len = _pick_len("GOAL_MAX_LEN")
        coh_max_len  = _pick_len("COH_MAX_LEN")

        # 모델 로드 (+ task별 max_len 반영)
        self.goal = KoBERTClassifier(
            ckpt_or_name=goal_ckpt, num_labels=2, labels=self.goal_labels, max_len=goal_max_len
        )
        self.coh  = KoBERTClassifier(
            ckpt_or_name=coh_ckpt,  num_labels=2, labels=self.coh_labels,  max_len=coh_max_len
        )

        # 체크포인트의 id2label에서 "긍정" 인덱스 동기화
        def _positive_index(id2label_dict, positive_names: set[str], fallback: int = 1) -> int:
            if not id2label_dict:
                return fallback
            try:
                norm = {int(k): str(v).strip().lower() for k, v in id2label_dict.items()}
                for idx, name in norm.items():
                    if name in positive_names:
                        return idx
            except Exception:
                pass
            return fallback

        self.goal_pos_idx = _positive_index(getattr(self.goal.model.config, "id2label", None), {"success"})
        self.coh_pos_idx  = _positive_index(getattr(self.coh.model.config,  "id2label", None), {"coherent"})

        print("[DBG] GOAL id2label:", getattr(self.goal.model.config, "id2label", None))
        print("[DBG] COH  id2label:", getattr(self.coh.model.config,  "id2label", None))
        print("[DBG] POS IDX  goal:", self.goal_pos_idx, "coh:", self.coh_pos_idx)

        # 리포지토리(베스트-에포트)
        self.repo = ScoreRepository()

    def evaluate_dialog(self, history: List[Dict]) -> Dict[str, Any]:
        text = _linearize(history)

        g_probs = self.goal.predict([text], return_probs=True)[0]  # 예: [p0, p1] (순서는 ckpt따라 다름)
        c_probs = self.coh.predict([text],  return_probs=True)[0]

        g_p_pos = float(g_probs[self.goal_pos_idx])
        c_p_pos = float(c_probs[self.coh_pos_idx])

        return {
            "text": text,
            "goal_probs": [float(x) for x in g_probs],
            "coh_probs":  [float(x) for x in c_probs],
            "goal_score": round(g_p_pos * 100.0, 2),
            "coherence_score": round(c_p_pos * 100.0, 2),
            "goal_label": self.goal_labels[int(g_p_pos >= 0.5)],
            "coherence_label": self.coh_labels[int(c_p_pos >= 0.5)],
        }

    def save_to_mongo(
        self,
        eval_result: Dict[str, Any],
        userId: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Score 컬렉션에 저장. userId가 있으면 루트에 포함.
        meta에는 sessionId/title/tags 등 임의 키를 자유롭게 보관.
        Mongo 미연결 환경에서도 예외 없이 조용히 스킵.
        """
        now = datetime.now(timezone.utc)

        # 원본문 텍스트의 해시(중복 확인 등 활용 가능)
        input_hash = hashlib.sha256(eval_result["text"].encode("utf-8")).hexdigest()[:16]

        doc: Dict[str, Any] = {
            "userId": userId,
            "goal_score": float(eval_result["goal_score"]),
            "coherence_score": float(eval_result["coherence_score"]),
            "goal_probs": [float(x) for x in eval_result["goal_probs"]],
            "coherence_probs": [float(x) for x in eval_result["coh_probs"]],
            "goal_label": eval_result["goal_label"],
            "coherence_label": eval_result["coherence_label"],
            "text": eval_result["text"],
            "input_hash": input_hash,
            "createdAt": now,          # ISODate로 저장됨
            "meta": meta or {},
        }

        # 자주 쓰는 메타를 루트로 복사(선택)
        if meta:
            for k in ("sessionId", "title", "tags"):
                if k in meta:
                    doc[k] = meta[k]

        return self.repo.insert(doc)


# Mongo 스타일 문서 → 평가 입력 어댑터
_ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "gemini": "assistant",
    "system": "assistant",
}

def _coerce_role(role: str) -> str:
    if not role:
        return "assistant"
    return _ROLE_MAP.get(role.lower().strip(), "assistant")

def _get_ts_iso(m: Dict) -> str:
    try:
        return (m.get("timestamp") or {}).get("$date") or ""
    except Exception:
        return ""

def mongo_dialog_to_evaluate_payload(doc: Dict) -> Tuple[List[Dict], Dict]:
    history = doc.get("history") or []

    # 정렬: seq → timestamp.$date → 원래 순서
    if any("seq" in m for m in history):
        history = sorted(history, key=lambda x: x.get("seq", 0))
    elif any((m.get("timestamp") or {}).get("$date") for m in history):
        history = sorted(history, key=_get_ts_iso)

    eval_history: List[Dict[str, str]] = []
    for m in history:
        role = _coerce_role(m.get("role", ""))
        content = (m.get("content") or "").strip()
        if content:
            eval_history.append({"role": role, "content": content})

    meta: Dict[str, Any] = {
        "source": "mongo_document",
        "userId": doc.get("userId"),
        "sessionId": doc.get("sessionId"),
        "title": doc.get("title"),
        "tags": doc.get("tags") or [],
        "archived": bool(doc.get("archived")),
        "bookmark": bool(doc.get("bookmark")),
        "label_success": bool(doc.get("success")),
        "messageCount": doc.get("messageCount"),
        "lastMessageAt": (doc.get("lastMessageAt") or {}).get("$date"),
        "raw_id": (doc.get("_id") or {}).get("$oid"),
    }
    return eval_history, meta
