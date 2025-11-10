# kobert_model/app/train_coherence.py
# -*- coding: utf-8 -*-

r"""
KoBERT coherence 분류기 학습 스크립트 (전화체 정규화)
- 데이터 로딩: linearized_clean.text 우선, 없으면 linearized.text
- 검증셋 F1 최대 임계값 τ* 저장: coh_threshold.json
- 학습 로그 CSV + 곡선 PNG(종합/손실/정확도) + 혼동행렬 + (신규) PR/ROC 저장
- trainer.best_model_checkpoint 에서 루트(out_dir)로 최종 모델 export

PowerShell 한 줄 실행 예:
python .\app\train_coherence.py --datasets _dir .\datasets\order_coherence --out .\data\ckpt_coherence --model skt/kobert-base-v1 --epochs 8 --lr 2e-5 --max_len 256 --weight_decay 0.10 --warmup_ratio 0.10 --es_patience 2 --dropout 0.15 --label_smooth 0.10 --bs 8 --grad_acc 3 --seed 42
"""

from __future__ import annotations
import os, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows: 맑은 고딕

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
)

# =============================================================================
# Utils
# =============================================================================

def read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Item:
    text: str
    y: int
    id: str


class TextClsDataset(Dataset):
    def __init__(self, items: List[Item], tok: AutoTokenizer, max_len: int):
        self.items = items
        self.tok = tok
        self.max_len = max_len
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        enc = self.tok(
            it.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(it.y, dtype=torch.long),
        }


def load_items_from_order_coh(root: Path) -> List[Item]:
    """datasets/order_coherence/{success,failure}/*.json 로드
    - success ⇒ coherent(1)
    - failure ⇒ incoherent(0)
    - 텍스트는 linearized_clean.text 우선, 없으면 linearized.text
    """
    items: List[Item] = []
    for split, label in (("success", 1), ("failure", 0)):
        d = root / split
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            obj = read_json(p)
            if not obj:
                continue
            txt = ((obj.get("linearized_clean") or {}).get("text") or "").strip()
            if not txt:
                txt = ((obj.get("linearized") or {}).get("text") or "").strip()
            if not txt:
                continue
            sid = str(obj.get("_id") or p.stem)
            items.append(Item(text=txt, y=label, id=sid))
    return items

# =============================================================================
# Plot helpers
# =============================================================================

def _plot_training_curves(logs: pd.DataFrame, outdir: Path):
    if logs is None or logs.empty:
        return
    # 1) 종합 곡선
    plt.figure(figsize=(8,5))
    if "loss" in logs.columns: plt.plot(logs["epoch"], logs["loss"], marker="o", label="train_loss")
    if "eval_loss" in logs.columns: plt.plot(logs["epoch"], logs["eval_loss"], marker="o", label="val_loss")
    if "eval_accuracy" in logs.columns: plt.plot(logs["epoch"], logs["eval_accuracy"], marker="o", label="val_acc")
    if "eval_f1" in logs.columns: plt.plot(logs["epoch"], logs["eval_f1"], marker="o", label="val_f1")
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.title("Training Progress (Loss/Acc/F1)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=200); plt.close()

    # 2) Loss만
    if "loss" in logs.columns or "eval_loss" in logs.columns:
        plt.figure(figsize=(8,5))
        if "loss" in logs.columns: plt.plot(logs["epoch"], logs["loss"], marker="o", label="train_loss")
        if "eval_loss" in logs.columns: plt.plot(logs["epoch"], logs["eval_loss"], marker="o", label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "loss_curves.png", dpi=200); plt.close()

    # 3) Accuracy만
    if "eval_accuracy" in logs.columns:
        plt.figure(figsize=(8,5))
        plt.plot(logs["epoch"], logs["eval_accuracy"], marker="o", label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves (Val)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "accuracy_curves.png", dpi=200); plt.close()


def _plot_pr_roc(y_true: np.ndarray, prob_pos: np.ndarray, out_dir: Path, title_prefix: str):
    # PR
    p, r, thr = precision_recall_curve(y_true, prob_pos)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0)
    best_idx = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    tau = float(thr[best_idx]) if len(thr) else 0.5
    plt.figure(figsize=(6,5))
    plt.plot(r, p)
    if len(p) and len(r):
        plt.scatter(r[best_idx], p[best_idx], marker="o")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title_prefix} PR Curve (τ*={tau:.3f})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=200); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{title_prefix} ROC Curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200); plt.close()

# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", type=str, default="skt/kobert-base-v1")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=0.10)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--es_patience", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--label_smooth", type=float, default=0.10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_acc", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out); ensure_dir(out_dir)
    set_seed(args.seed)

    # Load data
    root = Path(args.datasets_dir)
    items = load_items_from_order_coh(root)
    if len(items) < 10:
        raise RuntimeError(f"Too few items: {len(items)} in {root}")

    X = [it.text for it in items]
    y = np.array([it.y for it in items], dtype=np.int64)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp
    )

    print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Model / Tokenizer
    cfg = AutoConfig.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "incoherent", 1: "coherent"},
        label2id={"incoherent": 0, "coherent": 1},
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=min(0.3, args.dropout + 0.05),
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=cfg)

    # Datasets
    def _mk_items(Xs, ys):
        return [Item(text=t, y=int(l), id=str(i)) for i, (t, l) in enumerate(zip(Xs, ys))]

    ds_train = TextClsDataset(_mk_items(X_train, y_train), tok, args.max_len)
    ds_val   = TextClsDataset(_mk_items(X_val, y_val), tok, args.max_len)
    ds_test  = TextClsDataset(_mk_items(X_test, y_test), tok, args.max_len)

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = probs.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    # Training args
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=max(8, args.bs),
        gradient_accumulation_steps=args.grad_acc,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        eval_strategy="epoch",  # 신버전 호환
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=args.label_smooth,
        report_to=[],  # wandb/off
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.es_patience)],
    )

    trainer.train()

    # Logs & Curves
    logs = pd.DataFrame(trainer.state.log_history)
    logs.to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8")
    _plot_training_curves(logs, out_dir)

    # τ* from validation
    val_pred = trainer.predict(ds_val)
    val_probs = torch.softmax(torch.tensor(val_pred.predictions), dim=-1)[:, 1].numpy()  # coherent 확률
    val_labels = val_pred.label_ids

    prec, rec, thr = precision_recall_curve(val_labels, val_probs)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    tau = float(thr[best_idx]) if len(thr) > 0 else 0.5

    with (out_dir / "coh_threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"best_threshold": tau}, f, ensure_ascii=False, indent=2)
    print(f"[BEST THRESHOLD][COH] τ* = {tau:.4f}")

    # PR/ROC (validation)
    _plot_pr_roc(val_labels, val_probs, out_dir, title_prefix="Validation")

    # Final test @ τ* 
    test_pred = trainer.predict(ds_test)
    test_probs = torch.softmax(torch.tensor(test_pred.predictions), dim=-1)[:, 1].numpy()
    test_labels = test_pred.label_ids
    test_preds = (test_probs >= tau).astype(int)

    report = classification_report(test_labels, test_preds, target_names=["incoherent", "coherent"], digits=4)
    print("\n[TEST REPORT @ τ*]\n" + report)

    cm = confusion_matrix(test_labels, test_preds, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["incoherent","coherent"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test @ τ*)"); plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200); plt.close()

    # Export best checkpoint to ROOT
    # trainer.state.best_model_checkpoint가 있으면 그걸 루트로 export (safetensors + bin)
    best_dir = Path(trainer.state.best_model_checkpoint or out_dir)
    best_cfg = AutoConfig.from_pretrained(best_dir)
    best_tok = tok  # 동일 토크나이저 사용
    best_mdl = AutoModelForSequenceClassification.from_pretrained(best_dir, config=best_cfg)

    # 루트(out_dir)에 최종본 저장 (둘 다 저장: safetensors + bin)
    best_mdl.save_pretrained(out_dir)                            # model.safetensors + config.json
    best_mdl.save_pretrained(out_dir, safe_serialization=False)  # pytorch_model.bin
    best_tok.save_pretrained(out_dir)

    # label map (명시적)
    (out_dir / "label_map.json").write_text(
        json.dumps({"0": "incoherent", "1": "coherent"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[DONE] Model exported to: {out_dir}")
    print(f"[INFO] Files at root: config.json, model.safetensors, pytorch_model.bin, tokenizer.*, *curves.png, *threshold.json")


if __name__ == "__main__":
    main()
