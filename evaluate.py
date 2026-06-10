"""
evaluate.py
───────────
Standalone evaluation on test set.
Generates: metrics_report.txt, confusion_matrix.png, per_class_metrics.png

Usage:
    python evaluate.py --data_dir ./memotion_dataset_7k --model ./results/best_model.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

from data.memotion_dataset import MemotionDataset, IDX2LABEL
from models.multimodal_model import build_model

TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MAX_LEN   = 64
CLASS_NAMES = [IDX2LABEL[i] for i in range(3)]


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    texts  = [b["text"] for b in batch]
    enc = TOKENIZER(texts, padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
    return {
        "image":          images,
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label":          labels,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"]

        logits = model(images, ids, mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 14}
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Confusion matrix → {save_path}")


def plot_per_class_metrics(labels, preds, save_path):
    prec = precision_score(labels, preds, average=None, zero_division=0)
    rec  = recall_score(labels, preds,    average=None, zero_division=0)
    f1   = f1_score(labels, preds,        average=None, zero_division=0)

    x     = np.arange(len(CLASS_NAMES))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, prec, width, label="Precision", color=colors[0], alpha=0.85)
    b2 = ax.bar(x,          rec,  width, label="Recall",    color=colors[1], alpha=0.85)
    b3 = ax.bar(x + width, f1,   width, label="F1 Score",  color=colors[2], alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics — Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Per-class metrics → {save_path}")


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] Device: {device}")

    # Load data
    test_ds = MemotionDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=collate_fn)

    # Load model
    model = build_model(device)
    ckpt  = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Eval] Loaded checkpoint from epoch {ckpt.get('epoch','?')} "
          f"(val_acc={ckpt.get('val_acc', '?')})")

    # Run
    labels, preds, probs = evaluate(model, test_loader, device)

    # Metrics
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec  = recall_score(labels, preds,    average="weighted", zero_division=0)
    f1   = f1_score(labels, preds,        average="weighted", zero_division=0)
    cm   = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)

    print("\n" + "="*55)
    print("  EVALUATION RESULTS")
    print("="*55)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    # Save report
    report_path = os.path.join(args.results_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("MULTIMODAL SENTIMENT ANALYSIS — EVALUATION REPORT\n")
        f.write("MemoTion-7k Dataset\n")
        f.write("="*55 + "\n\n")
        f.write(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"[Report] Saved → {report_path}")

    # Plots
    plot_confusion_matrix(cm, os.path.join(args.results_dir, "confusion_matrix.png"))
    plot_per_class_metrics(labels, preds, os.path.join(args.results_dir, "per_class_metrics.png"))
    print("\n[Done] All evaluation artifacts saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, default="./memotion_dataset_7k")
    parser.add_argument("--model",       type=str, default="./results/best_model.pt")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()
    main(args)
