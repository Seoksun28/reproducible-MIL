import os
import math
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from nystrom_attention import NystromAttention


# ===============================
# Utils
# ===============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_pid(filename: str) -> str:
    """
    파일명에서 환자 ID 추출.
    예: 'P001_CHI_0001.npz' -> 'P001'
        '1_1.npz' -> '1'
        'c2_3.npz' -> 'c2'
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    return parts[0]


def load_items_from_dirs(case_dir: str, control_dir: str):
    """
    case_dir: case 슬라이드 npz들이 있는 폴더
    control_dir: control 슬라이드 npz들이 있는 폴더

    return: items = [(path, label), ...]
            case=1, control=0
    """
    items = []

    def _collect(dir_path, label):
        if dir_path is None:
            return
        if not os.path.isdir(dir_path):
            raise ValueError(f"Not a directory: {dir_path}")
        # 정렬해서 순서 고정 (재현성)
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".npz"):
                continue
            path = os.path.join(dir_path, fname)
            items.append((path, label))

    _collect(case_dir, 1)
    _collect(control_dir, 0)

    return items


# ===============================
# Dataset & Collate
# ===============================
class NPZSlideDataset(Dataset):
    def __init__(self, items, feature_key="features"):
        """
        items: [(path, label), ...]
        """
        self.items = items
        self.feature_key = feature_key

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        npz = np.load(path)
        feats = npz[self.feature_key]
        # sanity check
        if feats.ndim != 2:
            raise ValueError(f"features must be 2D [N, D], got {feats.shape} in {path}")
        feats = torch.from_numpy(feats).float()  # [M, D]
        label = torch.tensor(label).long()
        return feats, label, path


def mil_collate(batch):
    """
    MIL 특성상 slide 단위로 하나씩 들어오므로 batch 크기는 1
    """
    feats, label, path = batch[0]
    return feats.unsqueeze(0), label.unsqueeze(0), [path]


# ===============================
# TransMIL Model
# ===============================
class TransLayer(nn.Module):
    def __init__(
        self,
        dim=512,
        heads=8,
        num_landmarks=None,
        pinv_iterations=6,
        attn_dropout=0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if num_landmarks is None:
            num_landmarks = dim // 2

        if dim % heads != 0:
            raise ValueError(f"embed_dim ({dim}) must be divisible by heads ({heads}).")

        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            residual=True,
            dropout=attn_dropout,
        )

    def forward(self, x):
        # pre-norm + residual
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = (
            self.proj(cnn_feat)
            + cnn_feat
            + self.proj1(cnn_feat)
            + self.proj2(cnn_feat)
        )
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(
        self,
        n_classes=2,
        input_dim=1536,   # 패치 feature dim
        embed_dim=512,
        heads=8,
        num_landmarks=None,
        attn_dropout=0.1,
        pinv_iterations=6,
    ):
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by heads ({heads}).")

        self.n_classes = n_classes
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self._fc1 = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_layer = PPEG(embed_dim)

        self.layer1 = TransLayer(
            dim=embed_dim,
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            attn_dropout=attn_dropout,
        )
        self.layer2 = TransLayer(
            dim=embed_dim,
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            attn_dropout=attn_dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self._fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, data, return_features: bool = False):
        """
        data: [B, N, input_dim]
        return_features:
          - True 일 때
            'feat': cls token embedding (pred 직전 벡터), [B, embed_dim]
            'attn': patch-level attention score, [B, N]
        """
        h = self._fc1(data)  # [B, N, embed_dim]
        B, N, C = h.shape

        # square grid padding
        grid = int(np.ceil(np.sqrt(N)))
        add_len = grid * grid - N
        if add_len > 0:
            h = torch.cat([h, h[:, :add_len]], dim=1)  # [B, grid*grid, C]

        # CLS token 붙이기
        cls = self.cls_token.expand(B, 1, C).to(data.device)
        h = torch.cat([cls, h], dim=1)  # [B, 1+grid*grid, C]

        # Transformer layer 1
        h = self.layer1(h)
        # PPEG
        h = self.pos_layer(h, grid, grid)
        # Transformer layer 2
        h = self.layer2(h)

        # norm
        h = self.norm(h)  # [B, 1+grid*grid, C]

        # CLS feature (pred 직전)
        cls_feat = h[:, 0]  # [B, C]

        # patch-level attention surrogate: CLS vs patch similarity
        patch_tokens = h[:, 1: 1 + N]           # [B, N, C] (원래 patch 수만)
        cls_expanded = cls_feat.unsqueeze(1)    # [B, 1, C]
        attn_scores = (patch_tokens * cls_expanded).sum(dim=-1) / math.sqrt(C)  # [B, N]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, N]

        # classifier
        logits = self._fc2(cls_feat)          # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)     # [B, n_classes]
        Y_hat = torch.argmax(Y_prob, dim=1)   # [B]

        out = {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat}
        if return_features:
            out["feat"] = cls_feat      # [B, C]
            out["attn"] = attn_weights  # [B, N]
        return out


# ===============================
# Train
# ===============================
def train_one_epoch(model, loa
