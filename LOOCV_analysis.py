import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


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
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 1.5,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "grid.color": "#bfbfbf",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 2.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================
#  1) PR Curve
# ============================
def plot_pr_curve(y_true, y_prob, out_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    set_medical_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(recall, precision, label=f"Model (AP = {ap:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================
#  2) Cut-off별 성능표
# ============================
def compute_cutoff_table(y_true, y_prob, thresholds=None):
    if thresholds is None:
        # 필요하면 여기에서 cut-off 조정
        thresholds = np.arange(0.1, 1.0, 0.1)

    rows = []
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob, dtype=float)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        # 작은 데이터라 0 division 조심
        sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # sensitivity / recall
        spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan  # specificity
        ppv  = TP / (TP + FP) if (TP + FP) > 0 else np.nan  # PPV
        npv  = TN / (TN + FN) if (TN + FN) > 0 else np.nan  # NPV
        acc  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else np.nan

        rows.append({
            "threshold": thr,
            "sensitivity": sens,
            "specificity": spec,
            "PPV": ppv,
            "NPV": npv,
            "accuracy": acc,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        })

    df = pd.DataFrame(rows)
    return df


def save_cutoff_table_png(df: pd.DataFrame, out_png_path: str):
    """
    cut-off 성능표를 PNG figure로 저장
    (논문용 figure table 느낌)
    """
    set_medical_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    # 표시할 열 선택 (원하면 TP,FP 등은 빼도 됨)
    display_cols = ["threshold", "sensitivity", "specificity", "PPV", "NPV", "accuracy"]

    # 소수점 포맷
    df_disp = df.copy()
    for col in display_cols:
        if col == "threshold":
            df_disp[col] = df_disp[col].map(lambda x: f"{x:.2f}")
        else:
            df_disp[col] = df_disp[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "NA")

    table = ax.table(
        cellText=df_disp[display_cols].values,
        colLabels=display_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 1.3)

    fig.tight_layout()
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


# ============================
#  3) Calibration Plot
# ============================
def plot_calibration(y_true, y_prob, out_path: str, n_bins: int = 10):
    """
    quantile binning으로 calibration curve 계산
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    set_medical_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    # reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")

    # model calibration
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2.3, label="Model")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration Plot")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================
#  Main
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True,
                        help="LOOCV 결과 result.csv 경로")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="그래프 / 표 png 출력 폴더")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="calibration plot bin 개수")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if not {"label", "prob_pos"}.issubset(df.columns):
        raise ValueError("CSV에 'label', 'prob_pos' 컬럼이 필요합니다.")

    y_true = df["label"].values
    y_prob = df["prob_pos"].values

    # 1) PR curve
    pr_path = os.path.join(args.out_dir, "pr_curve.png")
    plot_pr_curve(y_true, y_prob, pr_path)
    print(f"[INFO] Saved PR curve -> {pr_path}")

    # 2) Cut-off table (csv + png)
    cutoff_df = compute_cutoff_table(y_true, y_prob)
    cutoff_csv_path = os.path.join(args.out_dir, "cutoff_table.csv")
    cutoff_df.to_csv(cutoff_csv_path, index=False)
    print(f"[INFO] Saved cutoff table CSV -> {cutoff_csv_path}")

    cutoff_png_path = os.path.join(args.out_dir, "cutoff_table.png")
    save_cutoff_table_png(cutoff_df, cutoff_png_path)
    print(f"[INFO] Saved cutoff table PNG -> {cutoff_png_path}")

    # 3) Calibration plot
    calib_path = os.path.join(args.out_dir, "calibration_plot.png")
    plot_calibration(y_true, y_prob, calib_path, n_bins=args.n_bins)
    print(f"[INFO] Saved calibration plot -> {calib_path}")


if __name__ == "__main__":
    main()
