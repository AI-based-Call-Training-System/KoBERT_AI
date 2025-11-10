# kobert_model/app/kobert_eval.py
# -*- coding: utf-8 -*-
r"""
KoBERT 평가 라우터 (Goal / Coherence)
- prefix="/kobert", tags=["kobert"]
- router.get("/evaluate-preprocess/{pid}")
- http://localhost:8000/kobert/evaluate-preprocess/S-TEST-MID-001 (예시)
"""  

import os
import sys
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from fastapi import APIRouter, HTTPException

# =========================
# 환경 변수 & 상수 (unified)
# =========================

def _as_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def _as_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

def _as_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "")
    if not v:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off")

# 체크포인트 디렉터리 (unified)
GOAL_CKPT = os.getenv("GOAL_CKPT_DIR", "./data/ckpt_goal")
COH_CKPT  = os.getenv("COH_CKPT_DIR",  "./data/ckpt_coherence")

# 입력 길이
MAX_LEN_DEFAULT = _as_int("MAX_LEN", 256)

# τ* (coherence/goal) : ckpt JSON이 우선, 단 "FORCE=true"면 .env 값을 강제
COH_THRESHOLD_ENV        = _as_float("COH_THRESHOLD", 0.55)
GOAL_THRESHOLD_ENV       = _as_float("GOAL_THRESHOLD", 0.70)
COH_THRESHOLD_FORCE_ENV  = _as_bool("COH_THRESHOLD_FORCE", False)
GOAL_THRESHOLD_FORCE_ENV = _as_bool("GOAL_THRESHOLD_FORCE", False)

# goal 하이브리드 파라미터
GOAL_TAIL_WEIGHT   = _as_float("GOAL_TAIL_WEIGHT", 0.6)
GOAL_CANCEL_PENALTY = _as_float("GOAL_CANCEL_PENALTY", 0.20)

# coherence 집계/정규화 및 라벨
COH_AGG            = (os.getenv("COH_AGG", "mean") or "mean").lower().strip()  # "min" | "mean" | "p10"
POS_LABEL_GOAL     = os.getenv("POS_LABEL_GOAL", "success")
POS_LABEL_COH      = os.getenv("POS_LABEL_COHERENCE", "coherent")
COH_EVAL_NORMALIZE = _as_bool("COH_EVAL_NORMALIZE", True)

# 디바이스
_device_env = (os.getenv("DEVICE", "") or "").strip().lower()
if _device_env in ("cuda", "cpu"):
    DEVICE = torch.device("cuda" if _device_env == "cuda" and torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MongoDB (unified)
MONGODB_URI          = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME              = os.getenv("DB_NAME", "Call-Training-DB")
PREPROCESS_COLLECTION = os.getenv("PREPROCESS_COLLECTION", "Preprocesses")
SCORE_COLLECTION      = os.getenv("SCORE_COLLECTION", "Scores")

# 디버그
EMIT_DEBUG_TAIL = _as_bool("EMIT_DEBUG_TAIL", True)

# =========================
# 공용 유틸
# =========================

def softmax_np(logits: torch.Tensor) -> np.ndarray:
    arr = logits.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[0]
    e = np.exp(arr - np.max(arr))
    return e / e.sum()

def probs_with_labels(probs: np.ndarray, id2label: Dict[int, str]) -> Dict[str, float]:
    return { id2label[i]: float(p) for i, p in enumerate(probs) }

def pick_positive_score(bylabel: Dict[str, float], positive_label: str) -> float:
    if positive_label in bylabel:
        return bylabel[positive_label]
    for k in ("success", "coherent", "True", "positive", "ok", "pass"):
        if k in bylabel:
            return bylabel[k]
    return max(bylabel.values())

def argmax_label(bylabel: Dict[str, float]) -> str:
    return max(bylabel.items(), key=lambda x: x[1])[0]

def _eff_max_len(cfg: AutoConfig, max_len_env: int) -> int:
    return min(max_len_env, int(getattr(cfg, "max_position_embeddings", 512)))

def _upper_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("gemini", "assistant", "bot", "agent", "ai"):
        return "GEMINI"
    if r in ("user", "human", "customer"):
        return "USER"
    return (role or "USER").upper()

def _history_parts(doc: dict) -> List[str]:
    parts: List[str] = []
    for t in doc.get("history", []) or []:
        content = (t.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{_upper_role(t.get('role',''))}: {content}")
    return parts

def _apply_head_tail(parts: List[str], head: int, tail: int) -> List[str]:
    n = len(parts)
    if n <= head + tail:
        return parts
    return parts[:head] + parts[-tail:]

def linearize_history(doc: dict) -> str:
    view = (doc.get("view") or {})
    trunc = (view.get("truncate") or {})
    head = int(trunc.get("head_turns", 2))
    tail = int(trunc.get("tail_turns", 8))
    sep  = (view.get("linearize_sep") or "[SEP]")

    parts = _history_parts(doc)
    used = parts if len(parts) <= 10 else _apply_head_tail(parts, head, tail)
    return f" {sep} ".join(used) if used else "[EMPTY]"

def get_candidate_texts_from_doc(doc: dict) -> Tuple[List[str], int]:
    """(coherence용) windows > linearized > history"""
    view = (doc.get("view") or {})
    doc_max_len = int(view.get("max_len_tokens", MAX_LEN_DEFAULT))

    wins = []
    for w in doc.get("windows", []) or []:
        txt = (w.get("text") or "").strip()
        if txt:
            wins.append(txt)
    if wins:
        return wins, doc_max_len

    lin = ((doc.get("linearized") or {}).get("text") or "").strip()
    if not lin:
        lin = linearize_history(doc)
    return [lin], doc_max_len

# =========================
# (coherence 전용) 평가 입력 정규화
# =========================

_RE_ROLE_PREFIX = re.compile(r"\b(?:USER|GEMINI|ASSISTANT|BOT|AGENT|AI)\s*:\s*", re.I)

def normalize_for_coh_eval(text: str, sep_replacement: str = "\n") -> str:
    """리터럴 [SEP] 제거 + ROLE 프리픽스 제거 + 공백 정리 (평가 입력 전용)"""
    t = text.replace(" [SEP] ", sep_replacement).replace("[SEP]", sep_replacement)
    t = _RE_ROLE_PREFIX.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# =========================
# 인코딩/재슬라이스
# =========================

class EncWin:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 full_tokens: int, eff_len: int, truncated: bool,
                 debug_tail_text: str = ""):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.full_tokens = full_tokens
        self.eff_len = eff_len
        self.truncated = truncated
        self.debug_tail_text = debug_tail_text

def _tokenize_with_offsets(tok: AutoTokenizer, text: str) -> Tuple[List[int], Optional[List[Tuple[int,int]]]]:
    try:
        enc = tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return enc["input_ids"], enc.get("offset_mapping", None)
    except Exception:
        return tok.encode(text, add_special_tokens=False), None

def _wrap_with_special(tok: AutoTokenizer, body_ids: List[int], eff: int) -> List[int]:
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    if cls_id is None or sep_id is None:
        return body_ids[:eff]
    body_max = max(0, eff - 2)
    body = body_ids[:body_max]
    return [cls_id] + body + [sep_id]

def _encode_single_text(tok: AutoTokenizer, cfg: AutoConfig, text: str, max_len_env: int) -> EncWin:
    eff = _eff_max_len(cfg, max_len_env)
    body, offsets = _tokenize_with_offsets(tok, text)
    full_tokens = len(body) + 2

    def _make_tail_for_whole():
        if offsets and len(offsets) > 0:
            end_char = offsets[-1][1]
            start_char = max(0, end_char - 120)
            return text[start_char:end_char]
        else:
            return text[-120:]

    if full_tokens <= eff:
        ids = _wrap_with_special(tok, body, eff)
        input_ids = torch.tensor([ids], dtype=torch.long)
        attn = torch.ones_like(input_ids)
        tail_text = _make_tail_for_whole() if EMIT_DEBUG_TAIL else ""
        return EncWin(input_ids, attn, full_tokens, eff, truncated=False, debug_tail_text=tail_text)

    # 초과면 head+tail 병합
    k_head = max(2, int(eff * 0.25)) - 1
    k_tail = max(1, eff - 2 - k_head)
    head_part = body[:k_head]
    tail_part = body[-k_tail:]
    merged = head_part + tail_part
    ids = _wrap_with_special(tok, merged, eff)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attn = torch.ones_like(input_ids)

    if EMIT_DEBUG_TAIL:
        if offsets and len(offsets) > 0 and len(body) > 0:
            end_char = offsets[-1][1]
            start_char = max(0, end_char - 120)
            tail_text = text[start_char:end_char]
        else:
            tail_text = text[-120:]
    else:
        tail_text = ""

    return EncWin(input_ids, attn, len(body) + 2, eff, truncated=True, debug_tail_text=tail_text)

def _reslice_overlong_as_ids(
    texts: List[str],
    tok: AutoTokenizer,
    cfg: AutoConfig,
    max_len_env: int,
    stride_ratio: float = 0.2
) -> List[EncWin]:
    eff = _eff_max_len(cfg, max_len_env)
    out: List[EncWin] = []
    for t in texts:
        out.append(_encode_single_text(tok, cfg, t, max_len_env))
    return out

# =========================
# 모델 로딩 + τ*(임계값) 로딩
# =========================
_goal_loaded = _coh_loaded = False
goal_cfg = goal_tok = goal_model = None
coh_cfg  = coh_tok  = coh_model  = None

_coh_threshold: float  = COH_THRESHOLD_ENV
_goal_threshold: float = GOAL_THRESHOLD_ENV

def _try_load_coh_threshold_from_ckpt(ckpt_dir: str) -> Optional[float]:
    path = os.path.join(ckpt_dir, "coh_threshold.json")
    try:
        if os.path.exists(path):
            obj = json.loads(open(path, encoding="utf-8").read())
            return float(obj.get("best_threshold"))
    except Exception as e:
        print(f"[WARN] failed to load coh_threshold.json: {e}", file=sys.stderr)
    return None

def _try_load_goal_threshold_from_ckpt(ckpt_dir: str) -> Optional[float]:
    path = os.path.join(ckpt_dir, "goal_threshold.json")
    try:
        if os.path.exists(path):
            obj = json.loads(open(path, encoding="utf-8").read())
            return float(obj.get("best_threshold"))
    except Exception as e:
        print(f"[WARN] failed to load goal_threshold.json: {e}", file=sys.stderr)
    return None

def _load_goal():
    global _goal_loaded, goal_cfg, goal_tok, goal_model, _goal_threshold
    if _goal_loaded:
        return
    goal_cfg = AutoConfig.from_pretrained(GOAL_CKPT)
    goal_tok = AutoTokenizer.from_pretrained(GOAL_CKPT, use_fast=True)
    goal_model = AutoModelForSequenceClassification.from_pretrained(GOAL_CKPT, config=goal_cfg)
    goal_model.to(DEVICE).eval()
    t = _try_load_goal_threshold_from_ckpt(GOAL_CKPT)
    if t is not None:
        _goal_threshold = t
        print(f"[LOAD] Goal τ* from ckpt: {_goal_threshold:.4f}")
    else:
        print(f"[LOAD] Goal τ* from env:  {_goal_threshold:.4f}")
    if GOAL_THRESHOLD_FORCE_ENV:
        _goal_threshold = GOAL_THRESHOLD_ENV
        print(f"[FORCE] Goal τ* forced by env: {_goal_threshold:.4f}")
    _goal_loaded = True
    print(f"[LOAD] Goal model @ {GOAL_CKPT}")

def _load_coh():
    global _coh_loaded, coh_cfg, coh_tok, coh_model, _coh_threshold
    if _coh_loaded:
        return
    coh_cfg = AutoConfig.from_pretrained(COH_CKPT)
    coh_tok = AutoTokenizer.from_pretrained(COH_CKPT, use_fast=True)
    coh_model = AutoModelForSequenceClassification.from_pretrained(COH_CKPT, config=coh_cfg)
    coh_model.to(DEVICE).eval()
    t = _try_load_coh_threshold_from_ckpt(COH_CKPT)
    if t is not None:
        _coh_threshold = t
        print(f"[LOAD] Coherence τ* from ckpt: {_coh_threshold:.4f}")
    else:
        print(f"[LOAD] Coherence τ* from env:  {_coh_threshold:.4f}")
    if COH_THRESHOLD_FORCE_ENV:
        _coh_threshold = COH_THRESHOLD_ENV
        print(f"[FORCE] Coherence τ* forced by env: {_coh_threshold:.4f}")
    _coh_loaded = True
    print(f"[LOAD] Coherence model @ {COH_CKPT}")

# =========================
# 하이브리드 goal 유틸
# =========================

_CANCEL_RE = re.compile(r"(취소|못\s*하겠|진행\s*안\s*하겠|접수\s*되지\s*않|주문\s*안\s*하겠)", re.I)

def _last_two_turns(linearized_text: str, sep: str = "[SEP]") -> str:
    turns = [t.strip() for t in linearized_text.split(sep) if t.strip()]
    if not turns:
        return linearized_text
    # 마지막 GEMINI부터 다음 USER까지 2턴
    last_ass = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].upper().startswith("GEMINI:"):
            last_ass = i
            break
    if last_ass < 0:
        return " ".join(turns[-2:]) if len(turns) >= 2 else turns[-1]
    tail = turns[last_ass:last_ass + 2]
    if not tail:
        tail = turns[-2:] if len(turns) >= 2 else turns[-1:]
    return " ".join(tail)

def _has_cancel_trigger(text: str) -> bool:
    return bool(_CANCEL_RE.search(text))

# =========================
# ID 기반 단일/배치 평가
# =========================

def _evaluate_encwin(enc: EncWin, mdl, cfg, pos_label: str, model_kind: str, debug: bool = False) -> Dict[str, Any]:
    with torch.no_grad():
        out = mdl(input_ids=enc.input_ids.to(DEVICE), attention_mask=enc.attention_mask.to(DEVICE))
    probs = softmax_np(out.logits)
    bylbl = probs_with_labels(probs, cfg.id2label)
    label = argmax_label(bylbl)
    pos_score = pick_positive_score(bylbl, pos_label)

    resp = {
        f"{model_kind}_score": float(pos_score),
        f"{model_kind}_probs": bylbl,
        f"{model_kind}_label": label
    }
    if debug and EMIT_DEBUG_TAIL:
        resp[f"_debug_{model_kind}"] = {
            "used_tokens": int(enc.input_ids.shape[-1]),
            "full_tokens": int(enc.full_tokens),
            "eff_max_len": int(enc.eff_len),
            "truncated": bool(enc.truncated),
            "tail": enc.debug_tail_text,
        }
    return resp

def _aggregate_goal(per_win: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_win:
        return {"strategy": "last-assistant-approx", "agg_score": 0.0, "last_idx": -1, "best_idx": -1, "best_score": 0.0}
    cand = per_win[-2:] if len(per_win) >= 2 else per_win
    best_local = max(cand, key=lambda x: x["goal_score"])
    best_global = max(per_win, key=lambda x: x["goal_score"])
    return {
        "strategy": "last-assistant-approx",
        "agg_score": float(best_local["goal_score"]),
        "last_idx": int(best_local["idx"]),
        "best_idx": int(best_global["idx"]),
        "best_score": float(best_global["goal_score"]),
    }

def _aggregate_coh(per_win: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = np.array([w["coherence_score"] for w in per_win], dtype=float)
    if scores.size == 0:
        return {"strategy": COH_AGG, "agg_score": 0.0}
    if COH_AGG == "min":
        val = float(scores.min()); idx = int(scores.argmin())
        return {"strategy": "min", "agg_score": val, "min_idx": idx}
    elif COH_AGG == "p10":
        val = float(np.percentile(scores, 10.0))
        return {"strategy": "p10", "agg_score": val}
    else:
        val = float(scores.mean())
        return {"strategy": "mean", "agg_score": val}

# =========================
# 윈도우 집계 평가
# =========================

def evaluate_windows(texts: List[str], model_kind: str, max_len_env: int) -> Dict[str, Any]:
    _load_goal(); _load_coh()
    if model_kind == "goal":
        tok, cfg, mdl, pos = goal_tok, goal_cfg, goal_model, POS_LABEL_GOAL
        eval_texts = texts  # goal은 ROLE/[SEP] 유지
    else:
        tok, cfg, mdl, pos = coh_tok, coh_cfg, coh_model, POS_LABEL_COH
        eval_texts = [normalize_for_coh_eval(t) for t in texts] if COH_EVAL_NORMALIZE else texts

    encwins: List[EncWin] = _reslice_overlong_as_ids(eval_texts, tok, cfg, max_len_env)

    per_win: List[Dict[str, Any]] = []
    for i, ew in enumerate(encwins):
        out = _evaluate_encwin(ew, mdl, cfg, pos, model_kind, debug=True)
        per_win.append({"idx": i, **out})

    agg = _aggregate_goal(per_win) if model_kind == "goal" else _aggregate_coh(per_win)
    return {"per_window": per_win, "aggregate": agg}

# goal 하이브리드 1샷 함수

def evaluate_goal_hybrid(linearized_text: str, max_len_env: int) -> Dict[str, Any]:
    """linearized 전체 + 마지막 1~2턴을 각각 평가 후 가중 블렌딩 및 패널티 적용."""
    # full / tail 텍스트 구성
    full_text = (linearized_text or "").strip()
    tail_text = _last_two_turns(full_text)

    # 각각 평가
    g_full = evaluate_windows([full_text], "goal", max_len_env=max_len_env)
    g_tail = evaluate_windows([tail_text], "goal", max_len_env=max_len_env)

    s_full = float(g_full["aggregate"]["agg_score"])
    s_tail = float(g_tail["aggregate"]["agg_score"])

    # 가중 블렌딩
    score = GOAL_TAIL_WEIGHT * s_tail + (1.0 - GOAL_TAIL_WEIGHT) * s_full

    # 취소/미접수 트리거 패널티
    penalty_applied = False
    if _has_cancel_trigger(tail_text):
        score -= GOAL_CANCEL_PENALTY
        penalty_applied = True

    # 클립
    score = max(0.0, min(1.0, score))

    # 라벨
    label = "success" if score >= _goal_threshold else "failure"

    # per_window/aggregate 구성(디버깅)
    per_window = [
        {"idx": 0, "goal_score": s_full, "goal_label": "success" if s_full >= _goal_threshold else "failure",
         "source": "full-linearized"},
        {"idx": 1, "goal_score": s_tail, "goal_label": "success" if s_tail >= _goal_threshold else "failure",
         "source": "last-two-turns"},
    ]
    aggregate = {
        "strategy": "hybrid(full+last2)",
        "agg_score": score,
        "full_score": s_full,
        "tail_score": s_tail,
        "w_tail": GOAL_TAIL_WEIGHT,
        "penalty_applied": penalty_applied,
    }
    return {"label": label, "score": score, "per_window": per_window, "aggregate": aggregate}

# =========================
# FastAPI 라우터
# =========================

router = APIRouter(prefix="/kobert", tags=["kobert"])

def _load_from_mongo(pid: str) -> Optional[dict]:
    try:
        from pymongo import MongoClient
        from bson import ObjectId
    except Exception:
        return None
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    col = db[PREPROCESS_COLLECTION]
    doc = col.find_one({"preprocessId": pid}) or col.find_one({"_id": pid})
    if not doc:
        try:
            oid = ObjectId(pid)
            doc = col.find_one({"_id": oid})
        except Exception:
            pass
    client.close()
    return doc

@router.get("/evaluate-preprocess/{pid}")
def kobert_eval_preprocess(pid: str):
    """
    Preprocess 문서 평가:
      - goal: linearized 전체 + 말미 2턴 하이브리드
      - coherence: windows[].text (없으면 linearized)
    """
    doc = _load_from_mongo(pid)
    if not doc:
        raise HTTPException(status_code=404, detail="Preprocess not found")

    # goal 입력: linearized 우선 (없으면 history 직렬화)
    goal_text = ((doc.get("linearized") or {}).get("text") or "").strip()
    if not goal_text:
        goal_text = linearize_history(doc)

    # coherence 입력: windows 우선
    coh_inputs: List[str] = []
    for w in doc.get("windows", []) or []:
        t = (w.get("text") or "").strip()
        if t:
            coh_inputs.append(t)
    if not coh_inputs:
        coh_inputs = [goal_text]

    doc_max_len = int((doc.get("view") or {}).get("max_len_tokens", MAX_LEN_DEFAULT))

    # goal 하이브리드 평가 
    g = evaluate_goal_hybrid(goal_text, max_len_env=doc_max_len)
    g_score = g["score"]
    g_label = g["label"]

    # coherence 평가
    c = evaluate_windows(coh_inputs, "coherence", max_len_env=doc_max_len)
    c_score = c["aggregate"]["agg_score"]
    c_label = "coherent" if c_score >= _coh_threshold else "incoherent"

    result = {
        "preprocessId": pid,
        "input_count": {"goal": 2, "coherence": len(coh_inputs)},  # goal은 full+tail 2회
        "goal": {
            "label": g_label,
            "score": float(g_score),
            "aggregate": g["aggregate"],
            "per_window": g["per_window"],
        },
        "coherence": {
            "label": c_label,
            "score": float(c_score),
            "aggregate": c["aggregate"],
            "per_window": c["per_window"],
        },
        "meta": {
            "max_len_tokens_from_doc": doc_max_len,
            "goal_threshold": _goal_threshold,
            "coh_threshold": _coh_threshold,
            "coh_agg": COH_AGG,
            "coh_eval_normalize": COH_EVAL_NORMALIZE,
            "goal_tail_weight": GOAL_TAIL_WEIGHT,
            "goal_cancel_penalty": GOAL_CANCEL_PENALTY,
        }
    }

    # Mongo 저장
    try:
        from pymongo import MongoClient
        from datetime import datetime
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        scores = db[SCORE_COLLECTION]
        scores.update_one(
            {"preprocessId": pid},
            {"$set": {
                **result,
                "createdAt": datetime.utcnow().isoformat(),
                "source": "kobert_eval_api",
            }},
            upsert=True
        )
        client.close()
    except Exception as e:
        print(f"[WARN] Mongo 저장 실패: {e}", file=sys.stderr)

    return result

# (옵션) 본문 입력 평가
@router.post("/evaluate-preprocess-body")
def kobert_eval_preprocess_body(doc: dict):
    # goal 입력: linearized 우선
    goal_text = ((doc.get("linearized") or {}).get("text") or "").strip()
    if not goal_text:
        goal_text = linearize_history(doc)

    # coherence 입력: windows 우선
    coh_inputs: List[str] = []
    for w in doc.get("windows", []) or []:
        t = (w.get("text") or "").strip()
        if t:
            coh_inputs.append(t)
    if not coh_inputs:
        coh_inputs = [goal_text]

    view = (doc.get("view") or {})
    doc_max_len = int(view.get("max_len_tokens", MAX_LEN_DEFAULT))

    # goal 하이브리드
    g = evaluate_goal_hybrid(goal_text, max_len_env=doc_max_len)
    c = evaluate_windows(coh_inputs, "coherence", max_len_env=doc_max_len)

    return {
        "input_count": {"goal": 2, "coherence": len(coh_inputs)},
        "goal": {
            "label": g["label"],
            "score": float(g["score"]),
            "aggregate": g["aggregate"],
            "per_window": g["per_window"],
        },
        "coherence": {
            "label": "coherent" if c["aggregate"]["agg_score"] >= _coh_threshold else "incoherent",
            "score": float(c["aggregate"]["agg_score"]),
            "aggregate": c["aggregate"],
            "per_window": c["per_window"],
        },
        "meta": {
            "max_len_tokens_from_doc": doc_max_len,
            "goal_threshold": _goal_threshold,
            "coh_threshold": _coh_threshold,
            "coh_agg": COH_AGG,
            "coh_eval_normalize": COH_EVAL_NORMALIZE,
            "emit_debug_tail": EMIT_DEBUG_TAIL,
            "goal_tail_weight": GOAL_TAIL_WEIGHT,
            "goal_cancel_penalty": GOAL_CANCEL_PENALTY,
        }
    }
