import os
import re
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from src.builder import create_model
from itertools import chain
from collections import defaultdict, Counter
from tqdm import tqdm


# ===============================
# Dataset
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
        feats = torch.from_numpy(npz[self.feature_key]).float()
        label = torch.tensor(label).long()
        return feats, label, path


def mil_collate(batch):
    """
    MIL 특성상 slide 단위로 하나씩 들어오므로 batch 크기는 1
    """
    feats, label, path = batch[0]
    return feats.unsqueeze(0), label.unsqueeze(0), [path]


# ===============================
# Utility
# ===============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_pid(filename: str) -> str:
    """
    파일명에서 환자 ID를 추출하는 함수.
    실제 파일명 규칙에 맞게 필요 시 수정.
    예: 'P001_CHI_0001.npz' -> 'P001'
        '1_1.npz' -> '1'
        'c2_3.npz' -> 'c2'
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    return parts[0]


def get_patientwise_split(case_dir, control_dir, n_folds=3, split_seed=42):
    """
    환자 단위 stratified KFold split
    return: folds = [(train_items, test_items), ...]
        train_items/test_items: [(path, label), ...]
    """
    case_files = sorted([
        os.path.join(case_dir, f)
        for f in os.listdir(case_dir)
        if f.endswith(".npz")
    ])
    control_files = sorted([
        os.path.join(control_dir, f)
        for f in os.listdir(control_dir)
        if f.endswith(".npz")
    ])

    # 환자 단위 그룹화
    case_groups = defaultdict(list)    # pid -> [(path, 1), ...]
    for f in case_files:
        pid = extract_pid(os.path.basename(f))
        case_groups[pid].append((f, 1))

    control_groups = defaultdict(list) # pid -> [(path, 0), ...]
    for f in control_files:
        pid = extract_pid(os.path.basename(f))
        control_groups[pid].append((f, 0))

    # 환자 단위 리스트: [(pid, label, [(path, label), ...]), ...]
    patient_entries = []
    for pid, items in case_groups.items():
        patient_entries.append((pid, 1, items))
    for pid, items in control_groups.items():
        patient_entries.append((pid, 0, items))

    pids = [e[0] for e in patient_entries]
    y = [e[1] for e in patient_entries]

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=split_seed,
    )

    folds = []
    for train_idx, test_idx in skf.split(pids, y):
        train_items = []
        test_items = []
        for i in train_idx:
            train_items.extend(patient_entries[i][2])  # [(path, label), ...]
        for i in test_idx:
            test_items.extend(patient_entries[i][2])
        folds.append((train_items, test_items))

    return folds


def format_split_info(folds):
    """
    split 결과를 문자열로 정리해서 result.txt에 넣기 위한 함수.
    """
    lines = []
    lines.append("=== Split Summary (patient-wise stratified KFold) ===")
    for i, (train_items, test_items) in enumerate(folds, start=1):
        train_labels = [lbl for _, lbl in train_items]
        test_labels = [lbl for _, lbl in test_items]
        lines.append(f"[Fold {i}]")
        lines.append(
            f"  train: {len(train_items)} slides, label dist: {dict(Counter(train_labels))}"
        )
        lines.append(
            f"  test : {len(test_items)} slides, label dist: {dict(Counter(test_labels))}"
        )
    lines.append("====================================================")
    lines.append("")  # 빈 줄
    return "\n".join(lines)


# ===============================
# Training & Eval
# ===============================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for feats, labels, _ in loader:
        feats = feats.to(device)
        labels = labels.to(device)

        results_dict, _ = model(
            feats,
            loss_fn=nn.CrossEntropyLoss(),
            label=labels,
            return_attention=False, # CLAM에선 True
            return_slide_feats=False,
        )
        loss = results_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for feats, labels, _ in loader:
        feats = feats.to(device) # CLAM에선 labels = labels.to(device) 추가하기
        results_dict, _ = model(
            feats,
            loss_fn=None,
            label=None, # CLAM에선 여기서 labels를 넣어줘야됨
            return_attention=False, # CLAM에선 True
            return_slide_feats=False,
        )

        logits = results_dict["logits"]
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().item()
        pred = torch.argmax(logits, dim=1).cpu().item()
        label = labels.cpu().item()

        all_probs.append(prob)
        all_preds.append(pred)
        all_labels.append(label)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")
    try:
        aupr = average_precision_score(all_labels, all_probs)
    except ValueError:
        aupr = float("nan")

    return {"ACC": acc, "F1": f1, "AUROC": auroc, "AUPR": aupr}


# ===============================
# Main
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--control_dir", required=True)
    ap.add_argument("--feature_key", default="features")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--result_dir", default="./results")
    ap.add_argument("--split_seed", type=int, default=42,
                    help="환자 단위 KFold split을 위한 seed (모든 학습 seed에서 동일 split 사용)")
    args = ap.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"result_{args.model_name.replace('.', '_')}.txt")

    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # 1) 환자 단위 stratified KFold split은 split_seed로 한 번만 생성
    folds = get_patientwise_split(
        args.case_dir,
        args.control_dir,
        n_folds=args.folds,
        split_seed=args.split_seed,
    )

    # 2) 결과 파일 초기화 + split 정보 기록
    with open(result_path, "w") as f:
        f.write(f"MIL {args.folds}-Fold × 5-Seed Results (Model={args.model_name}, GPU={args.gpu})\n\n")
        f.write(format_split_info(folds))

    seed_list = [1, 2, 3, 4, 5]
    all_seed_metrics = defaultdict(list)

    for seed in seed_list:
        # 모델 초기값, 학습 순서 등을 위한 seed
        set_seed(seed)
        fold_metrics = []
        fold_losses = []

        print(f"\n=== Seed {seed} ({args.model_name}) ===")

        for fold_idx, (train_items, test_items) in enumerate(folds, start=1):
            train_ds = NPZSlideDataset(train_items, feature_key=args.feature_key)
            test_ds = NPZSlideDataset(test_items, feature_key=args.feature_key)

            # 입력 차원 추출
            sample_feats, _, _ = train_ds[0]
            in_dim = sample_feats.shape[1]

            model = create_model(args.model_name, in_dim=in_dim, num_classes=2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=mil_collate)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=mil_collate)

            epoch_losses = []
            for epoch in tqdm(range(1, args.epochs + 1),
                              desc=f"Seed {seed} | Fold {fold_idx}",
                              ncols=100):
                loss = train_one_epoch(model, train_loader, optimizer, device)
                epoch_losses.append(loss)

            mean_loss = np.mean(epoch_losses)
            fold_losses.append(mean_loss)
            print(f"[Seed {seed}][Fold {fold_idx}] mean_train_loss={mean_loss:.4f}")

            metrics = evaluate(model, test_loader, device)
            fold_metrics.append(metrics)
            print(f"[Seed {seed}][Fold {fold_idx}] {metrics}")

        # fold별 metric의 평균/표준편차 (해당 seed 기준)
        means = {k: np.nanmean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        stds = {k: np.nanstd([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        avg_loss = np.mean(fold_losses)

        # seed별 결과 기록
        with open(result_path, "a") as f:
            f.write(f"=== Seed {seed} ===\n")
            for i, m in enumerate(fold_metrics, 1):
                f.write(f"Fold{i}: ACC={m['ACC']:.4f}, F1={m['F1']:.4f}, "
                        f"AUROC={m['AUROC']:.4f}, AUPR={m['AUPR']:.4f}\n")
            f.write(f"Mean : {means}\n")
            f.write(f"Std  : {stds}\n")
            f.write(f"Fold Avg Train Loss: {avg_loss:.4f}\n\n")

        # 전체 seed 통계용으로 seed별 mean만 모음
        for k in means:
            all_seed_metrics[k].append(means[k])

    # 전체 seed 요약
    with open(result_path, "a") as f:
        f.write("=== Overall Seed Summary ===\n")
        for k, vals in all_seed_metrics.items():
            f.write(f"{k}: mean={np.nanmean(vals):.4f}, std={np.nanstd(vals):.4f}\n")

    print(f"\n✅ Finished all seeds for model [{args.model_name}]. Results saved to {result_path}")


if __name__ == "__main__":
    main()
