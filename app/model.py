# kobert_model/app/model.py
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _device() -> torch.device:
    dev = os.getenv("DEVICE", "cpu")
    if dev.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(dev)
        logger.info(f"Using device: {device} - {torch.cuda.get_device_name(0)}")
        return device
    logger.info("Using CPU device")
    return torch.device("cpu")


def _max_len() -> int:
    try:
        return int(os.getenv("MAX_LEN", "256"))
    except Exception:
        return 256


class KoBERTEmbedding:
    def __init__(self, model_name_or_path: str):
        self.device = _device()
        self.max_len = _max_len()

        logger.info(f"Loading tokenizer: {model_name_or_path}")
        # Windows/KoBERT: fast 비활성화 + trust_remote_code 차단
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=False,
        )

        # 패딩 토큰이 없는 경우 추가
        if self.tokenizer.pad_token is None:
            # eos가 없으면 unk를 사용
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            logger.info(f"Set pad_token to: {self.tokenizer.pad_token}")

        logger.info(f"Loading model: {model_name_or_path}")
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
        )
        self.model.to(self.device).eval()

        # 안전 설정: pad_token_id 동기화
        if (
            getattr(self.model.config, "pad_token_id", None) is None
            or self.model.config.pad_token_id != self.tokenizer.pad_token_id
        ):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # vocab 불일치 시 임베딩 테이블 리사이즈
        model_vocab_size = int(self.model.config.vocab_size)
        tokenizer_vocab_size = int(len(self.tokenizer))
        logger.info(
            f"Model vocab size: {model_vocab_size}, Tokenizer vocab size: {tokenizer_vocab_size}"
        )
        if model_vocab_size != tokenizer_vocab_size:
            logger.warning(
                f"Vocab size mismatch! Model: {model_vocab_size}, Tokenizer: {tokenizer_vocab_size} -> resizing"
            )
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            self.model.config.vocab_size = tokenizer_vocab_size

        # 타입토큰 사이즈(세그먼트) 기록
        self.type_vocab_size = int(getattr(self.model.config, "type_vocab_size", 2))

    @torch.inference_mode()
    def embed(self, texts):
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("All texts must be non-empty strings")

        # 빈 문자열/공백 처리
        processed_texts = [(t.strip() or "[UNK]") for t in texts]

        logger.info(f"Embedding {len(processed_texts)} texts, max_length: {self.max_len}")

        try:
            enc = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_attention_mask=True,
            )

            # 하드 클램프: input_ids / token_type_ids / attention_mask 
            vocab_size = int(self.model.config.vocab_size)
            enc["input_ids"] = enc["input_ids"].clamp(min=0, max=vocab_size - 1)

            if "token_type_ids" in enc:
                enc["token_type_ids"] = enc["token_type_ids"].clamp(
                    min=0, max=self.type_vocab_size - 1
                )

            if "attention_mask" in enc:
                # 0/1 보정 (곱셈 시 안정성)
                enc["attention_mask"] = (enc["attention_mask"] > 0).to(enc["input_ids"].dtype)

            logger.info(
                f"Input shape: {enc['input_ids'].shape}, "
                f"Max token ID: {int(enc['input_ids'].max())}, "
                f"Type IDs max: {int(enc.get('token_type_ids', torch.zeros(1)).max()) if 'token_type_ids' in enc else -1}"
            )

            # GPU/CPU 디바이스로 이동
            enc = {k: v.to(self.device) for k, v in enc.items()}

            # 모델 추론
            out = self.model(**enc)  # last_hidden_state: [B, L, H]
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)  # [B, L, 1]

            # Mean pooling
            summed = (last * mask).sum(dim=1)           # [B, H]
            counts = mask.sum(dim=1).clamp(min=1)       # [B, 1]
            mean_pooled = (summed / counts).contiguous()

            result = mean_pooled.cpu().tolist()
            logger.info(
                f"Successfully generated embeddings: {len(result)} x {len(result[0]) if result else 0}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in embedding: {str(e)}")
            # CUDA 에러인 경우 CPU로 fallback 시도
            if "CUDA" in str(e) and self.device.type == "cuda":
                logger.warning("CUDA error detected, falling back to CPU")
                return self._embed_cpu_fallback(processed_texts)
            raise

    def _embed_cpu_fallback(self, texts):
        """CUDA 에러시 CPU로 fallback"""
        original_device = self.device
        self.model = self.model.cpu()
        self.device = torch.device("cpu")

        try:
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_attention_mask=True,
            )

            # CPU 경로도 동일한 가드
            vocab_size = int(self.model.config.vocab_size)
            enc["input_ids"] = enc["input_ids"].clamp(min=0, max=vocab_size - 1)
            if "token_type_ids" in enc:
                enc["token_type_ids"] = enc["token_type_ids"].clamp(
                    min=0, max=self.type_vocab_size - 1
                )
            if "attention_mask" in enc:
                enc["attention_mask"] = (enc["attention_mask"] > 0).to(enc["input_ids"].dtype)

            out = self.model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (last * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = (summed / counts).contiguous()
            return mean_pooled.tolist()
        finally:
            # 원래 device로 복구 시도 (다음 요청을 위해)
            try:
                self.model = self.model.to(original_device)
                self.device = original_device
            except Exception:
                logger.warning("Could not restore original device, staying on CPU")


class KoBERTClassifier:
    def __init__(self, ckpt_or_name: str, num_labels: int | None = None, labels=None, max_len: int | None = None):
        self.device = _device()
        self.max_len = max_len if max_len is not None else _max_len()
        self.labels = labels

        logger.info(f"Loading classifier tokenizer: {ckpt_or_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_or_name,
            use_fast=False,
            trust_remote_code=False,
        )

        # pad_token이 없을 경우 안전하게 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        logger.info(f"Loading classifier model: {ckpt_or_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_or_name,
            num_labels=num_labels if num_labels else None,
            trust_remote_code=False,
        )

        # 디바이스로 이동 후 평가 모드로 전환
        self.model.to(self.device).eval()

        # pad_token_id 동기화
        if (
            getattr(self.model.config, "pad_token_id", None) is None
            or self.model.config.pad_token_id != self.tokenizer.pad_token_id
        ):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # vocab mismatch 시 리사이즈
        model_vocab_size = int(self.model.config.vocab_size)
        tokenizer_vocab_size = int(len(self.tokenizer))
        if model_vocab_size != tokenizer_vocab_size:
            logger.warning(
                f"[CLS] Vocab mismatch -> resizing ({model_vocab_size} -> {tokenizer_vocab_size})"
            )
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            self.model.config.vocab_size = tokenizer_vocab_size

        self.type_vocab_size = int(getattr(self.model.config, "type_vocab_size", 2))

    @torch.inference_mode()
    def predict(self, texts, return_probs: bool = True):
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("All texts must be non-empty strings")

        processed_texts = [(t.strip() or "[UNK]") for t in texts]

        try:
            enc = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_attention_mask=True,
            )

            # 하드 클램프
            vocab_size = int(self.model.config.vocab_size)
            enc["input_ids"] = enc["input_ids"].clamp(min=0, max=vocab_size - 1)
            if "token_type_ids" in enc:
                enc["token_type_ids"] = enc["token_type_ids"].clamp(
                    min=0, max=self.type_vocab_size - 1
                )
            if "attention_mask" in enc:
                enc["attention_mask"] = (enc["attention_mask"] > 0).to(enc["input_ids"].dtype)

            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits

            if return_probs:
                probs = F.softmax(logits, dim=-1).cpu().tolist()
                return probs
            return logits.cpu().tolist()

        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            if "CUDA" in str(e) and self.device.type == "cuda":
                return self._predict_cpu_fallback(processed_texts, return_probs)
            raise

    def _predict_cpu_fallback(self, texts, return_probs):
        """CUDA 에러시 CPU로 fallback"""
        original_device = self.device
        self.model = self.model.cpu()
        self.device = torch.device("cpu")

        try:
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_attention_mask=True,
            )

            vocab_size = int(self.model.config.vocab_size)
            enc["input_ids"] = enc["input_ids"].clamp(min=0, max=vocab_size - 1)
            if "token_type_ids" in enc:
                enc["token_type_ids"] = enc["token_type_ids"].clamp(
                    min=0, max=self.type_vocab_size - 1
                )
            if "attention_mask" in enc:
                enc["attention_mask"] = (enc["attention_mask"] > 0).to(enc["input_ids"].dtype)

            logits = self.model(**enc).logits

            if return_probs:
                probs = F.softmax(logits, dim=-1).tolist()
                return probs
            return logits.tolist()
        finally:
            try:
                self.model = self.model.to(original_device)
                self.device = original_device
            except Exception:
                logger.warning("Could not restore original device, staying on CPU")
