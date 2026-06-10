"""
data/memotion_dataset.py
────────────────────────
MemoTion-7k Dataset Loader

Expected folder layout (place inside project root):
  memotion_dataset_7k/
      images/          ← 6992 meme images
      labels.csv       ← has columns: image_name, text_ocr, overall_sentiment
      labels.xlsx      ← same (fallback)

labels.csv sentiment values  →  mapped to 3 classes:
  'very positive' / 'positive'  →  0  (Positive)
  'neutral'                     →  1  (Neutral)
  'negative' / 'very negative'  →  2  (Negative)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Label mapping ──────────────────────────────────────────────────────────────
SENTIMENT_MAP = {
    "very positive": 0,
    "positive":      0,
    "neutral":       1,
    "negative":      2,
    "very negative": 2,
}
IDX2LABEL = {0: "Positive", 1: "Neutral", 2: "Negative"}
LABEL2IDX = {v: k for k, v in IDX2LABEL.items()}

# ── Image transforms ───────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


class MemotionDataset(Dataset):
    """
    Parameters
    ----------
    data_dir   : path to memotion_dataset_7k/
    split      : 'train' | 'val' | 'test'
    val_ratio  : fraction of data for val
    test_ratio : fraction of data for test
    transform  : torchvision transform (None → auto based on split)
    max_text_len: max tokens for BERT tokenizer (set externally via collate)
    """

    def __init__(
        self,
        data_dir   : str,
        split      : str  = "train",
        val_ratio  : float = 0.10,
        test_ratio : float = 0.10,
        transform         = None,
        seed       : int  = 42,
    ):
        self.data_dir  = data_dir
        self.img_dir   = os.path.join(data_dir, "images")
        self.split     = split
        self.transform = transform or (TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM)

        # ── Load labels ────────────────────────────────────────────────────────
        csv_path  = os.path.join(data_dir, "labels.csv")
        xlsx_path = os.path.join(data_dir, "labels.xlsx")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        elif os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
        else:
            raise FileNotFoundError(
                f"labels.csv / labels.xlsx not found in {data_dir}"
            )

        # ── Normalise column names ─────────────────────────────────────────────
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Detect image name column
        img_col = next(
            (c for c in df.columns if "image" in c or "file" in c or "name" in c),
            df.columns[0]
        )
        # Detect text column
        text_col = next(
            (c for c in df.columns if "text" in c or "ocr" in c or "caption" in c),
            None
        )
        # Detect label column
        label_col = next(
            (c for c in df.columns if "sentiment" in c or "label" in c or "overall" in c),
            None
        )

        if label_col is None:
            raise ValueError(f"Cannot find sentiment column. Columns: {df.columns.tolist()}")

        # ── Clean ──────────────────────────────────────────────────────────────
        df = df.dropna(subset=[img_col, label_col]).reset_index(drop=True)
        df["_label_str"] = df[label_col].astype(str).str.strip().str.lower()
        df = df[df["_label_str"].isin(SENTIMENT_MAP)].reset_index(drop=True)
        df["_label_idx"] = df["_label_str"].map(SENTIMENT_MAP)

        if text_col:
            df["_text"] = df[text_col].fillna("").astype(str)
        else:
            df["_text"] = ""

        df["_img_name"] = df[img_col].astype(str)

        # ── Train / val / test split ───────────────────────────────────────────
        rng   = np.random.default_rng(seed)
        idx   = rng.permutation(len(df))
        n     = len(df)
        n_test= int(n * test_ratio)
        n_val = int(n * val_ratio)

        if split == "test":
            chosen = idx[:n_test]
        elif split == "val":
            chosen = idx[n_test : n_test + n_val]
        else:
            chosen = idx[n_test + n_val:]

        self.df = df.iloc[chosen].reset_index(drop=True)

        print(f"[Dataset] {split:5s} → {len(self.df):5d} samples  "
              f"| Pos:{(self.df['_label_idx']==0).sum()} "
              f"Neu:{(self.df['_label_idx']==1).sum()} "
              f"Neg:{(self.df['_label_idx']==2).sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_name  = row["_img_name"]
        text      = row["_text"]
        label     = int(row["_label_idx"])

        # ── Load image ─────────────────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        image = self.transform(image)

        return {
            "image":     image,     # [3, 224, 224] tensor
            "text":      text,      # raw string
            "label":     label,     # int 0/1/2
            "img_name":  img_name,
        }


def get_dataloaders(
    data_dir    : str,
    batch_size  : int  = 32,
    num_workers : int  = 4,
    val_ratio   : float = 0.10,
    test_ratio  : float = 0.10,
):
    train_ds = MemotionDataset(data_dir, "train", val_ratio, test_ratio)
    val_ds   = MemotionDataset(data_dir, "val",   val_ratio, test_ratio)
    test_ds  = MemotionDataset(data_dir, "test",  val_ratio, test_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, {
        "n_train": len(train_ds),
        "n_val":   len(val_ds),
        "n_test":  len(test_ds),
        "classes": IDX2LABEL,
    }
