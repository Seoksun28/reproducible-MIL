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
from tqdm import tqdm


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

    _collect(case_dir, 1)   # case = 1
    _collect(control_dir, 0)  # control = 0

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
        if feats.ndim != 2:
            raise ValueError(
                f"features must be 2D [N, D], got {feats.shape} in {path}"
            )
        feats = torch.from_numpy(feats).float()  # [N, D]
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
        patch_tokens = h[:, 1 : 1 + N]           # [B, N, C] (원래 patch 수만)
        cls_expanded = cls_feat.unsqueeze(1)     # [B, 1, C]
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
# Train (with tqdm)
# ===============================
def train_one_epoch(model, loader, optimizer, device, fold_idx=None, epoch=None, total_epochs=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    desc = "Train"
    if fold_idx is not None and epoch is not None and total_epochs is not None:
        desc = f"Fold {fold_idx+1} | Epoch {epoch}/{total_epochs}"

    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)

    for feats, labels, _ in pbar:
        feats = feats.to(device)   # [1, N, D]
        labels = labels.to(device) # [1]

        optimizer.zero_grad()
        out = model(data=feats, return_features=False)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(loader))


# ===============================
# LOOCV
# ===============================
def run_loocv(args):
    # ===== device 선택 (GPU 인덱스 반영) =====
    if args.device == "cuda":
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            if args.gpu < 0 or args.gpu >= n_gpu:
                print(f"[WARN] GPU index {args.gpu} is out of range (0~{n_gpu-1}). CPU로 fallback 합니다.")
                device_str = "cpu"
            else:
                device_str = f"cuda:{args.gpu}"
        else:
            print("[WARN] CUDA not available. CPU로 fallback 합니다.")
            device_str = "cpu"
    else:
        device_str = "cpu"

    device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed)

    # 데이터 로드 (case, control 디렉토리)
    items = load_items_from_dirs(args.case_dir, args.control_dir)
    print(f"#slides total = {len(items)}")

    # 환자 grouping
    pid_to_indices = defaultdict(list)
    for idx, (path, label) in enumerate(items):
        fname = os.path.basename(path)
        pid = extract_pid(fname)
        pid_to_indices[pid].append(idx)

    patients = sorted(pid_to_indices.keys())
    print(f"#patients = {len(patients)}")

    # result 폴더 구조
    os.makedirs(args.result_dir, exist_ok=True)
    attn_dir = os.path.join(args.result_dir, "attention")
    vec_dir = os.path.join(args.result_dir, "vector")
    os.makedirs(attn_dir, exist_ok=True)
    os.makedirs(vec_dir, exist_ok=True)

    # result.csv
    import csv
    csv_path = os.path.join(args.result_dir, "result.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["slide", "label", "pred", "prob_pos"])

    all_labels = []
    all_probs = []
    all_preds = []

    # ================== patient-wise LOOCV ==================
    for fold_idx, test_pid in enumerate(patients):
        print(f"\n========== Fold {fold_idx+1}/{len(patients)} | Test PID = {test_pid} ==========")

        test_indices = pid_to_indices[test_pid]
        train_indices = [
            i for pid, idxs in pid_to_indices.items() if pid != test_pid for i in idxs
        ]

        train_items = [items[i] for i in train_indices]
        test_items = [items[i] for i in test_indices]

        print(f"Train slides: {len(train_items)}, Test slides: {len(test_items)}")

        train_loader = DataLoader(
            NPZSlideDataset(train_items, feature_key=args.feature_key),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=mil_collate,
            pin_memory=True,
        )

        test_loader = DataLoader(
            NPZSlideDataset(test_items, feature_key=args.feature_key),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=mil_collate,
            pin_memory=True,
        )

        # ----- 모델 생성 -----
        model = TransMIL(
            n_classes=args.n_classes,
            input_dim=args.input_dim,
            embed_dim=args.embed_dim,
            heads=args.heads,
            num_landmarks=args.num_landmarks,
            attn_dropout=args.attn_dropout,
            pinv_iterations=args.pinv_iterations,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # ----- 고정 epoch 학습 (epoch-level tqdm) -----
        epoch_iter = tqdm(
            range(1, args.epochs + 1),
            desc=f"Fold {fold_idx+1} epochs",
            ncols=100
        )

        for epoch in epoch_iter:
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                fold_idx=fold_idx,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            epoch_iter.set_postfix({"loss": f"{loss:.4f}"})

        # ----- Test + 결과 저장 -----
        model.eval()
        with torch.no_grad():
            for feats, labels, paths in tqdm(
                test_loader,
                desc=f"Eval Fold {fold_idx+1}",
                ncols=100,
                leave=False
            ):
                feats = feats.to(device)
                labels = labels.to(device)

                out = model(data=feats, return_features=True)
                probs = out["Y_prob"][:, 1]   # positive class prob
                preds = out["Y_hat"]
                attn = out["attn"]            # [1, N]
                feat_vec = out["feat"]        # [1, embed_dim]

                slide_path = paths[0]
                slide_name = os.path.splitext(os.path.basename(slide_path))[0]

                label_int = int(labels.item())
                pred_int = int(preds.item())
                prob_pos = float(probs.item())

                # result.csv 기록
                writer.writerow([slide_name, label_int, pred_int, prob_pos])

                # global metric 용
                all_labels.append(label_int)
                all_probs.append(prob_pos)
                all_preds.append(pred_int)

                # attention 저장 (patch별 score)
                attn_np = attn.squeeze(0).cpu().numpy()    # [N]
                np.save(os.path.join(attn_dir, f"{slide_name}.npy"), attn_np)

                # pred 직전 layer feature 저장 (cls token)
                feat_np = feat_vec.squeeze(0).cpu().numpy()  # [embed_dim]
                np.save(os.path.join(vec_dir, f"{slide_name}.npy"), feat_np)

    csv_f.close()

    # ================== 전체 LOOCV 성능 ==================
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    overall_acc = accuracy_score(all_labels, all_preds)
    try:
        overall_auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        overall_auroc = float("nan")

    txt_path = os.path.join(args.result_dir, "result.txt")
    with open(txt_path, "w") as f:
        f.write(f"Overall ACC: {overall_acc:.4f}\n")
        f.write(f"Overall AUROC: {overall_auroc:.4f}\n")

    print("\n===== LOOCV DONE =====")
    print(f"Overall ACC: {overall_acc:.4f}, AUROC: {overall_auroc:.4f}")
    print(f"Saved result.csv, result.txt under {args.result_dir}")


# ===============================
# Main
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="Patient-wise LOOCV with TransMIL")

    # data
    parser.add_argument("--case_dir", type=str, required=True,
                        help="case npz 폴더 (label=1)")
    parser.add_argument("--control_dir", type=str, required=True,
                        help="control npz 폴더 (label=0)")
    parser.add_argument("--feature_key", type=str, default="features")
    parser.add_argument("--result_dir", type=str, default="result_transmil_loocv")

    # model
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--input_dim", type=int, default=1536)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num_landmarks", type=int, default=None)
    parser.add_argument("--attn_dropout", type=float, default=0.25)
    parser.add_argument("--pinv_iterations", type=int, default=6)

    # train
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)  # MIL 특성상 1 유지 추천
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # etc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (cuda일 때만 사용)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loocv(args)
