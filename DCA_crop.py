import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ============================
#  Medical Style (NEJM/JAMA)
# ============================
def set_medical_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.5,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "grid.color": "#bfbfbf",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================
#  Load CSV
# ============================
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if not {"label", "prob_pos"}.issubset(df.columns):
        raise ValueError("CSV must contain 'label' and 'prob_pos'.")
    y_true = df["label"].values
    y_prob = df["prob_pos"].values
    return y_true, y_prob


# ============================
#  ROC Curve
# ============================
def plot_roc_curve(y_true, y_prob, out_path):
    set_medical_style()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#003f5c", label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#7a7a7a", label="No Skill")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[ROC] saved → {out_path}")


# ============================
#  DCA (Decision Curve Analysis)
# ============================
def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """
    y_true : (N,)
    y_prob : (N,)
    thresholds : array-like of probability thresholds

    반환: thresholds, nb_model, nb_all, nb_none
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)

    # clinically relevant range만 기본값으로 사용 (0.01 ~ 0.30)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.30, 30)

    n_pos = (y_true == 1).sum()
    n_neg = N - n_pos

    nb_model = []
    nb_all = []
    nb_none = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()

        # Model net benefit
        nb_m = (TP / N) - (FP / N) * (pt / (1 - pt))
        # Treat-all net benefit
        nb_a = (n_pos / N) - (n_neg / N) * (pt / (1 - pt))
        # Treat-none net benefit (항상 0)
        nb_n = 0.0

        nb_model.append(nb_m)
        nb_all.append(nb_a)
        nb_none.append(nb_n)

    return np.array(thresholds), np.array(nb_model), np.array(nb_all), np.array(nb_none)


def plot_dca_curve(y_true, y_prob, out_png, out_csv=None):
    """
    DCA를 임상 저널 스타일로 플로팅:
    - x축: threshold 0 ~ 0.30
    - y축: net benefit를 작은 범위로 자동 확대
    - curves: Model (solid), Treat All (dashed), Treat None (0 baseline)
    """
    set_medical_style()

    thr, nb_m, nb_a, nb_n = decision_curve_analysis(y_true, y_prob)

    plt.figure(figsize=(6.5, 6))

    # Model & Treat All
    plt.plot(thr, nb_m, label="Model", color="#003f5c")
    plt.plot(thr, nb_a, label="Treat All", linestyle="--", color="#7a7a7a")

    # Treat None = 0 baseline
    plt.axhline(0.0, label="Treat None", color="#b3b3b3", linestyle=":", linewidth=2)

    # 축 설정 (0~0.30 확대)
    plt.xlim(thr.min(), thr.max())

    # y축을 데이터에 맞춰 좁게 잡기 (약간의 margin 포함)
    y_min = min(nb_m.min(), nb_a.min(), 0.0)
    y_max = max(nb_m.max(), nb_a.max(), 0.0)
    if y_max - y_min < 0.01:  # 너무 좁으면 최소 폭 보장
        center = 0.5 * (y_max + y_min)
        y_min = center - 0.01
        y_max = center + 0.01
    margin = 0.1 * (y_max - y_min)
    plt.ylim(y_min - margin, y_max + margin)

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print(f"[DCA] saved → {out_png}")

    if out_csv:
        df = pd.DataFrame({
            "threshold": thr,
            "net_benefit_model": nb_m,
            "net_benefit_treat_all": nb_a,
            "net_benefit_treat_none": nb_n,
        })
        df.to_csv(out_csv, index=False)
        print(f"[DCA CSV] saved → {out_csv}")


# ============================
#  MAIN
# ============================
def parse_args():
    p = argparse.ArgumentParser(description="Medical-style ROC & DCA plotter")
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    y_true, y_prob = load_data(args.csv_path)

    # ROC
    roc_path = os.path.join(args.out_dir, "roc_curve_medical.png")
    plot_roc_curve(y_true, y_prob, roc_path)

    # DCA
    dca_png = os.path.join(args.out_dir, "dca_curve_medical.png")
    dca_csv = os.path.join(args.out_dir, "dca_values.csv")
    plot_dca_curve(y_true, y_prob, dca_png, dca_csv)


if __name__ == "__main__":
    main()
