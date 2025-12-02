import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import umap  # pip install umap-learn
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn 이 설치되어 있지 않습니다. UMAP은 건너뜁니다.")
    print("       설치: pip install umap-learn")


# ============================
#  Figure 스타일 (논문용)
# ============================
def set_paper_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


# ============================
#  파일명 → label 추론
#  case   : "숫자_숫자" (예: 1_1, 2_3 ...)
#  control: "c숫자_숫자" (예: c1_1, c2_3 ...)
# ============================
def infer_label_from_name(stem: str) -> int:
    """
    stem: 확장자 제거한 파일명 (예: '1_1', 'c2_3')
    return: case=1, control=0
    """
    if stem.lower().startswith("c"):
        return 0  # control
    else:
        return 1  # case


def load_features(vec_dir: str):
    """
    vec_dir 안의 .npy들을 모두 읽어서:
      X: (N, D) feature
      y: (N,) label (0=control, 1=case)
      names: (N,) 슬라이드 이름
    을 반환
    """
    X_list = []
    y_list = []
    name_list = []

    for fname in sorted(os.listdir(vec_dir)):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(vec_dir, fname)
        stem = os.path.splitext(fname)[0]

        feat = np.load(path)
        # (D,) 또는 (1, D)만 허용
        if feat.ndim == 1:
            feat = feat[None, :]  # (D,) -> (1, D)
        elif feat.ndim != 2 or feat.shape[0] != 1:
            raise ValueError(f"{path} shape 이상함: {feat.shape}, (D,) 또는 (1, D) 여야 함")

        feat = feat.squeeze(0)  # (1, D) -> (D,)
        label = infer_label_from_name(stem)

        X_list.append(feat)
        y_list.append(label)
        name_list.append(stem)

    X = np.stack(X_list, axis=0)  # (N, D)
    y = np.array(y_list, dtype=int)
    names = np.array(name_list)

    print(f"[INFO] Loaded {X.shape[0]} slides, feature dim = {X.shape[1]}")
    print(f"[INFO] #case = {(y == 1).sum()}, #control = {(y == 0).sum()}")
    return X, y, names


def plot_2d_scatter(Z, y, names, title, out_png, out_pdf):
    """
    Z: (N, 2) 2D 좌표
    y: (N,) label (0 or 1)
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # 한 컬럼용 크기 (저널 스타일)

    # 색/마커 설정 (필요하면 여기서 바꿔도 됨)
    class_config = [
        (0, "Control", "o"),
        (1, "Case", "^"),
    ]

    for label, cls_name, marker in class_config:
        mask = (y == label)
        if mask.sum() == 0:
            continue
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=36,
            alpha=0.85,
            label=f"{cls_name} (n={mask.sum()})",
            marker=marker,
            linewidths=0.5,
            edgecolors="black",
        )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    # 제목은 논문 Figure에서는 캡션으로 가는 경우가 많아서 옵션으로
    ax.set_title(title, pad=8)

    # 격자는 깔끔하게만
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    # legend를 안 겹치게
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[INFO] Saved: {out_png}")
    print(f"[INFO] Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Slide-level feature (CLS token) UMAP / t-SNE visualization (paper-ready)"
    )
    parser.add_argument(
        "--vec_dir",
        type=str,
        required=True,
        help="슬라이드별 feature(.npy)들이 저장된 폴더 (예: result_transmil_loocv/vector)",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="slide_feat",
        help="저장할 이미지 파일 prefix (기본: slide_feat)",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=20.0,
        help="t-SNE perplexity (기본: 20.0, N보다 작게)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1) feature 로드
    X, y, names = load_features(args.vec_dir)

    # 2) UMAP
    if HAS_UMAP:
        print("[INFO] Running UMAP...")
        reducer = umap.UMAP(
            n_components=2,
            random_state=args.seed,
        )
        Z_umap = reducer.fit_transform(X)  # (N, 2)

        out_umap_png = f"{args.out_prefix}_umap.png"
        out_umap_pdf = f"{args.out_prefix}_umap.pdf"
        plot_2d_scatter(Z_umap, y, names,
                        title="UMAP of slide-level features",
                        out_png=out_umap_png,
                        out_pdf=out_umap_pdf)
    else:
        print("[WARN] UMAP은 건너뜁니다 (umap-learn 미설치).")

    # 3) t-SNE
    print("[INFO] Running t-SNE...")
    n_samples = X.shape[0]
    perp = min(args.tsne_perplexity, max(5.0, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        random_state=args.seed,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
    )
    Z_tsne = tsne.fit_transform(X)

    out_tsne_png = f"{args.out_prefix}_tsne.png"
    out_tsne_pdf = f"{args.out_prefix}_tsne.pdf"
    plot_2d_scatter(Z_tsne, y, names,
                    title="t-SNE of slide-level features",
                    out_png=out_tsne_png,
                    out_pdf=out_tsne_pdf)


if __name__ == "__main__":
    main()
