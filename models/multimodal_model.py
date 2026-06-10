"""
models/multimodal_model.py
──────────────────────────
Architecture
============

  ┌─────────────────────┐     ┌──────────────────────────┐
  │  Image (224×224×3)  │     │   Text (meme OCR string)  │
  └────────┬────────────┘     └────────────┬─────────────┘
           │                               │
    ResNet-50 backbone              DistilBERT
    (pretrained ImageNet)        (distilbert-base-uncased)
           │                               │
     pool → [2048]                  [CLS] → [768]
           │                               │
    Linear(2048→256)              Linear(768→256)
      + ReLU + Dropout               + ReLU + Dropout
           │                               │
           └──────────── Concat ───────────┘
                              │
                           [512]
                              │
                   Linear(512→128) + ReLU
                   Linear(128→3)
                   Softmax → [Positive, Neutral, Negative]

Training: Cross-Entropy loss, AdamW optimizer, cosine LR schedule
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel, DistilBertTokenizer

# ── Projection dimension ───────────────────────────────────────────────────────
PROJ_DIM = 256


class VisualEncoder(nn.Module):
    """ResNet-50 image encoder → projected embedding."""

    def __init__(self, proj_dim: int = PROJ_DIM, dropout: float = 0.3):
        super().__init__()
        base        = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # → [B, 2048, 1, 1]
        self.proj     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        feat = self.backbone(x)   # [B, 2048, 1, 1]
        return self.proj(feat)    # [B, proj_dim]


class TextEncoder(nn.Module):
    """DistilBERT text encoder → projected embedding."""

    def __init__(self, proj_dim: int = PROJ_DIM, dropout: float = 0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.proj = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_ids, attention_mask):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls  = out.last_hidden_state[:, 0, :]   # [CLS] token → [B, 768]
        return self.proj(cls)                   # [B, proj_dim]


class FusionClassifier(nn.Module):
    """Concat + MLP → 3-class sentiment."""

    def __init__(self, proj_dim: int = PROJ_DIM, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, visual_emb, text_emb):
        x = torch.cat([visual_emb, text_emb], dim=-1)   # [B, proj_dim*2]
        return self.mlp(x)                               # [B, 3]  raw logits


class MultimodalSentimentModel(nn.Module):
    """Full model: image + text → sentiment logits."""

    def __init__(self, proj_dim: int = PROJ_DIM, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.visual_enc  = VisualEncoder(proj_dim, dropout)
        self.text_enc    = TextEncoder(proj_dim, dropout)
        self.fusion      = FusionClassifier(proj_dim, num_classes, dropout)

    def forward(self, image, input_ids, attention_mask):
        v = self.visual_enc(image)
        t = self.text_enc(input_ids, attention_mask)
        return self.fusion(v, t)   # [B, 3]

    def get_embeddings(self, image, input_ids, attention_mask):
        """Return intermediate embeddings for analysis."""
        v = self.visual_enc(image)
        t = self.text_enc(input_ids, attention_mask)
        return v, t


def build_model(device: str = "cpu") -> MultimodalSentimentModel:
    model = MultimodalSentimentModel()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params/1e6:.2f}M")
    return model
