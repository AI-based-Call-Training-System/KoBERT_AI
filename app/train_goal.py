# kobert_model/app/train_goal.py
# -*- coding: utf-8 -*-

r"""
Goal 분류기 학습 스크립트
- 대화는 literal "[SEP]" + ROLE 프리픽스( USER:/GEMINI: )를 유지하여 직렬화
- 검증셋에서 F1 최대화 임계값 τ*을 산출하여 goal_threshold.json으로 저장
- 학습 로그 CSV + 곡선 PNG(Loss/Acc/F1) + 혼동행렬 + PR/ROC 곡선 저장

PowerShell 한 줄 실행 예:
python .\app\train_goal.py --datasets_dir .\datasets\order --out .\data\ckpt_goal --model skt/kobert-base-v1 --epochs 8 --lr 2e-5 --max_len 256 --weight_decay 0.10 --warmup_ratio 0.10 --es_patience 2 --dropout 0.10 --label_smooth 0.05 --bs 8 --grad_acc 3 --seed 42
"""

from __future__ import annotations
import os, json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve, auc, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 기본 폰트

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
)

# =============================================================================
# 직렬화: HISTORY → "USER: xxx [SEP] GEMINI: yyy ..."
# =============================================================================

def build_dialog_text(obj: Dict) -> str:
    hist = obj.get("history", []) or []
    parts = []
    for t in hist:
        role = str(t.get("role", "")).upper() or "USER"
        cont = str(t.get("content", "")).strip()
        if cont:
            parts.append(f"{role}: {cont}")
    txt = " [SEP] ".join(parts)
    return txt if txt else "[EMPTY]"

# =============================================================================
# Dataset
# =============================================================================

@dataclass
class Item:
    text: str
    y: int   # 0=failure, 1=success
    id: str

class GoalDS(Dataset):
    def __init__(self, items: List[Item], tok: AutoTokenizer, max_len: int):
        self.items, self.tok, self.max_len = items, tok, max_len
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
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
            "labels": torch.tensor(int(it.y), dtype=torch.long),
        }

# =============================================================================
# IO helpers
# =============================================================================

def read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_items_order_dataset(root: Path) -> List[Item]:
    """datasets/order/{success,failure}/*.json → Item 리스트"""
    items: List[Item] = []
    for split, label in (("success", 1), ("failure", 0)):
        d = root / split
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            obj = read_json(p)
            if not obj:
                continue
            text = build_dialog_text(obj)
            if not text:
                continue
            sid = str(obj.get("_id") or p.stem)
            items.append(Item(text=text, y=label, id=sid))
    return items

# =============================================================================
# Plot helpers
# =============================================================================

def _plot_curves(logs: pd.DataFrame, out_dir: Path):
    if logs is None or logs.empty:
        return
    # 통합 곡선
    plt.figure(figsize=(8,5))
    if "loss" in logs.columns: plt.plot(logs["epoch"], logs["loss"], marker="o", label="train_loss")
    if "eval_loss" in logs.columns: plt.plot(logs["epoch"], logs["eval_loss"], marker="o", label="val_loss")
    if "eval_acc" in logs.columns: plt.plot(logs["epoch"], logs["eval_acc"], marker="o", label="val_acc")
    if "eval_f1_macro" in logs.columns: plt.plot(logs["epoch"], logs["eval_f1_macro"], marker="o", label="val_f1")
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.title("Training Progress (Loss/Acc/F1)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=200); plt.close()

    # Loss only
    if "loss" in logs.columns or "eval_loss" in logs.columns:
        plt.figure(figsize=(8,5))
        if "loss" in logs.columns: plt.plot(logs["epoch"], logs["loss"], marker="o", label="train_loss")
        if "eval_loss" in logs.columns: plt.plot(logs["epoch"], logs["eval_loss"], marker="o", label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "loss_curves.png", dpi=200); plt.close()

    # Acc only
    if "eval_acc" in logs.columns:
        plt.figure(figsize=(8,5))
        plt.plot(logs["epoch"], logs["eval_acc"], marker="o", label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves (Val)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "accuracy_curves.png", dpi=200); plt.close()


def _plot_pr_roc(y_true: np.ndarray, prob_pos: np.ndarray, out_dir: Path, title_prefix: str):
    # PR
    p, r, thr = precision_recall_curve(y_true, prob_pos)
    f1 = np.where((p + r) > 0, 2*p*r/(p+r), 0)
    best_idx = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    tau = float(thr[best_idx]) if len(thr) else 0.5
    plt.figure(figsize=(6,5)); plt.plot(r, p); plt.scatter(r[best_idx], p[best_idx], marker="o")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{title_prefix} PR Curve (τ*={tau:.3f})"); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_dir / "pr_curve.png", dpi=200); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5)); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{title_prefix} ROC Curve"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_dir / "roc_curve.png", dpi=200); plt.close()

# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", type=str, required=True, help="datasets/order 디렉토리")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", type=str, default="skt/kobert-base-v1")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--weight_decay", type=float, default=0.10)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--es_patience", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_acc", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out); ensure_dir(out_dir)
    set_seed(args.seed)

    # Load data (order dataset)
    items = load_items_order_dataset(Path(args.datasets_dir))
    assert len(items) > 10, f"Too few items in {args.datasets_dir}"

    X = [it.text for it in items]
    y = np.array([it.y for it in items], dtype=np.int64)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp)

    # Tokenizer/Model
    id2label = {0: "failure", 1: "success"}
    label2id = {v:k for k,v in id2label.items()}

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cfg = AutoConfig.from_pretrained(
        args.model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=min(0.3, args.dropout + 0.05),
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=cfg)

    ds_tr = GoalDS([Item(t,int(l),str(i)) for i,(t,l) in enumerate(zip(X_tr,y_tr))], tok, args.max_len)
    ds_va = GoalDS([Item(t,int(l),str(i)) for i,(t,l) in enumerate(zip(X_va,y_va))], tok, args.max_len)
    ds_te = GoalDS([Item(t,int(l),str(i)) for i,(t,l) in enumerate(zip(X_te,y_te))], tok, args.max_len)

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = probs.argmax(-1)
        acc  = accuracy_score(labels, preds)
        f1   = f1_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        return {"eval_acc": acc, "eval_f1": f1, "eval_prec": prec, "eval_recall": rec}

    # Training args
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=max(8,args.bs),
        gradient_accumulation_steps=args.grad_acc,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=args.label_smooth,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.es_patience)],
    )

    trainer.train()

    # Logs & Curves
    logs = pd.DataFrame(trainer.state.log_history)
    logs.to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8")
    _plot_curves(logs, out_dir)

    # τ* from validation (success=positive)
    val_pred = trainer.predict(ds_va)
    val_probs = torch.softmax(torch.tensor(val_pred.predictions), dim=-1)[:, 1].numpy()
    val_labels = val_pred.label_ids

    prec, rec, thr = precision_recall_curve(val_labels, val_probs)
    f1s = np.where((prec + rec) > 0, 2*prec*rec/(prec+rec), 0)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    tau = float(thr[best_idx]) if len(thr) > 0 else 0.5

    with (out_dir / "goal_threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"best_threshold": tau}, f, ensure_ascii=False, indent=2)
    print(f"[BEST THRESHOLD][GOAL] τ* = {tau:.4f}")

    # PR/ROC curves (validation)
    _plot_pr_roc(val_labels, val_probs, out_dir, title_prefix="Validation")

    # Final test @ τ*
    test_pred = trainer.predict(ds_te)
    test_probs = torch.softmax(torch.tensor(test_pred.predictions), dim=-1)[:, 1].numpy()
    test_labels = test_pred.label_ids
    test_preds = (test_probs >= tau).astype(int)

    report = classification_report(test_labels, test_preds, target_names=["failure","success"], digits=4)
    print("\n[TEST REPORT @ τ*]\n" + report)

    cm = confusion_matrix(test_labels, test_preds, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["failure","success"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test @ τ*)"); plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200); plt.close()

    # Save model/tokenizer/labels
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "labels.json").write_text(json.dumps({"0":"failure","1":"success"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved to: {out_dir}")


if __name__ == "__main__":
    main()
