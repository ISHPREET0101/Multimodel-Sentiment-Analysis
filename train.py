"""
train.py
────────
Full training pipeline for Multimodal Sentiment Analysis on MemoTion-7k.

Usage:
    python train.py --data_dir ./memotion_dataset_7k --epochs 10 --batch_size 32

Outputs (saved to ./results/):
    best_model.pt          ← best checkpoint (by val accuracy)
    training_curves.png    ← loss + accuracy plots
    metrics_report.txt     ← final test accuracy, precision, recall, F1
    confusion_matrix.png   ← test confusion matrix
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import DistilBertTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data.memotion_dataset import get_dataloaders, IDX2LABEL
from models.multimodal_model import build_model

# ── Tokenizer (shared) ─────────────────────────────────────────────────────────
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MAX_LEN   = 64


def collate_fn(batch):
    """Custom collate: tokenize text strings → tensors."""
    images  = torch.stack([b["image"] for b in batch])
    labels  = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    texts   = [b["text"] for b in batch]

    enc = TOKENIZER(
        texts,
        padding        = True,
        truncation     = True,
        max_length     = MAX_LEN,
        return_tensors = "pt",
    )
    return {
        "image":          images,
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label":          labels,
    }


# ── Training / evaluation helpers ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if train else "eval"):
            images  = batch["image"].to(device)
            ids     = batch["input_ids"].to(device)
            mask    = batch["attention_mask"].to(device)
            labels  = batch["label"].to(device)

            logits  = model(images, ids, mask)
            loss    = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    n    = len(all_labels)
    acc  = accuracy_score(all_labels, all_preds)
    loss = total_loss / n
    return loss, acc, all_preds, all_labels


def compute_metrics(all_labels, all_preds, class_names):
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds,    average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds,        average="weighted", zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": cm, "report": report}


def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.results_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, info = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    # Patch collate
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_loader.dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_loader.dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=collate_fn)

    class_names = [info["classes"][i] for i in range(3)]

    # ── Model ──────────────────────────────────────────────────────────────────
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──────────────────────────────────────────────────────────
    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_path    = os.path.join(args.results_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc, vl_preds, vl_labels = run_epoch(model, val_loader, criterion, None, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"| tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"| vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
              f"| {elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     vl_acc,
                "args":        vars(args),
            }, best_path)
            print(f"  ✓ Best model saved (val_acc={vl_acc:.4f})")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_curves(history, os.path.join(args.results_dir, "training_curves.png"))

    # ── Test evaluation ────────────────────────────────────────────────────────
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n[Eval] Loaded best model from epoch {ckpt['epoch']}")

    _, test_acc, test_preds, test_labels = run_epoch(
        model, test_loader, criterion, None, device, train=False
    )
    metrics = compute_metrics(test_labels, test_preds, class_names)

    print("\n" + "="*55)
    print("TEST METRICS")
    print("="*55)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])

    # Save metrics
    report_path = os.path.join(args.results_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("TEST METRICS\n" + "="*50 + "\n")
        f.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision : {metrics['precision']:.4f}\n")
        f.write(f"Recall    : {metrics['recall']:.4f}\n")
        f.write(f"F1 Score  : {metrics['f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics["report"])

    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        os.path.join(args.results_dir, "confusion_matrix.png")
    )

    print(f"\n[Done] Results saved to {args.results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal Sentiment Model")
    parser.add_argument("--data_dir",    type=str,   default="./memotion_dataset_7k")
    parser.add_argument("--results_dir", type=str,   default="./results")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int,   default=4)
    args = parser.parse_args()
    main(args)
