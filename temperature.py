import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
#  Global
# ============================
EPS = 1e-12


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
    y_true = df["label"].values.astype(int)
    y_prob = df["prob_pos"].values.astype(float)
    return df, y_true, y_prob


# ============================
#  Temperature Scaling
# ============================
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1.0 - p))


def nll_with_T(T, z, y):
    """
    T: scalar > 0
    z: (N,)  원래 logit difference
    y: (N,)  0/1 label
    """
    p_T = 1.0 / (1.0 + np.exp(-z / T))   # sigmoid(z/T)
    p_T = np.clip(p_T, EPS, 1.0 - EPS)
    nll = -np.mean(y * np.log(p_T) + (1 - y) * np.log(1.0 - p_T))
    return nll


def find_best_T(z, y, t_min=0.2, t_max=5.0, num_grid=200):
    """
    단순 1차원 그리드 서치 + 로컬 refine
    """
    Ts = np.linspace(t_min, t_max, num_grid)
    losses = [nll_with_T(T, z, y) for T in Ts]
    idx = int(np.argmin(losses))
    T_best = Ts[idx]

    # 주변에서 한 번 더 촘촘히
    local_min = max(t_min, T_best - 0.5)
    local_max = min(t_max, T_best + 0.5)
    Ts_local = np.linspace(local_min, local_max, num_grid)
    losses_local = [nll_with_T(T, z, y) for T in Ts_local]
    idx2 = int(np.argmin(losses_local))
    T_refined = Ts_local[idx2]

    return T_refined


def brier_score(p, y):
    p = np.asarray(p)
    y = np.asarray(y)
    return np.mean((p - y) ** 2)


# ============================
#  Calibration Plot
# ============================
def reliability_curve(p, y, n_bins=10):
    """
    p: (N,) predicted prob
    y: (N,) label
    return: bin_centers, mean_pred, frac_pos
    """
    p = np.asarray(p)
    y = np.asarray(y)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1  # 0 ~ n_bins-1

    bin_centers = []
    bin_mean_pred = []
    bin_frac_pos = []

    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) == 0:
            continue
        bin_p = p[mask]
        bin_y = y[mask]
        bin_centers.append(0.5 * (bins[b] + bins[b + 1]))
        bin_mean_pred.append(np.mean(bin_p))
        bin_frac_pos.append(np.mean(bin_y))

    return np.array(bin_centers), np.array(bin_mean_pred), np.array(bin_frac_pos)


def plot_calibration_medical(y_true, p_raw, p_cal, out_path):
    """
    NEJM/JAMA 스타일 Calibration plot
    - 대각선: Ideal
    - 동그라미: Before TS
    - 네모: After TS
    """
    set_medical_style()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Ideal line
    ax.plot([0, 1], [0, 1], "--", linewidth=1.5, color="#7a7a7a", label="Ideal")

    # Before TS
    _, mean_raw, frac_raw = reliability_curve(p_raw, y_true, n_bins=10)
    ax.plot(mean_raw, frac_raw, "o-", label="Before TS", color="#003f5c")

    # After TS
    _, mean_cal, frac_cal = reliability_curve(p_cal, y_true, n_bins=10)
    ax.plot(mean_cal, frac_cal, "s-", label="After TS", color="#bc5090")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Proportion")
    ax.set_title("Calibration Plot (Temperature Scaling)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Calibration] saved → {out_path}")


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


def plot_dca_curve_medical(y_true, y_prob, out_png, out_csv=None):
    """
    DCA를 임상 저널 스타일로 플로팅 (Temperature Scaling 적용 후 확률 사용)
    - x축: threshold 0 ~ 0.30
    - y축: net benefit를 작은 범위로 자동 확대
    - curves: Model (solid), Treat All (dashed), Treat None (0 baseline)
    """
    set_medical_style()

    thr, nb_m, nb_a, nb_n = decision_curve_analysis(y_true, y_prob)

    plt.figure(figsize=(6.5, 6))

    # Model & Treat All
    plt.plot(thr, nb_m, label="Model (Calibrated)", color="#003f5c")
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
    plt.title("Decision Curve Analysis (Calibrated)")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print(f"[DCA] saved → {out_png}")

    if out_csv:
        df = pd.DataFrame({
            "threshold": thr,
            "net_benefit_model_calibrated": nb_m,
            "net_benefit_treat_all": nb_a,
            "net_benefit_treat_none": nb_n,
        })
        df.to_csv(out_csv, index=False)
        print(f"[DCA CSV] saved → {out_csv}")


# ============================
#  MAIN
# ============================
def parse_args():
    p = argparse.ArgumentParser(
        description="Temperature Scaling + Medical-style Calibration & DCA"
    )
    p.add_argument("--csv_path", type=str, required=True,
                   help="LOOCV result.csv path (columns: label, prob_pos)")
    p.add_argument("--out_dir", type=str, default="plots_ts",
                   help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 데이터 로드
    df, y_true, p_raw = load_data(args.csv_path)

    # 2) logit 복원
    z = logit(p_raw)

    # 3) 최적 T 탐색
    T_star = find_best_T(z, y_true)
    nll_before = nll_with_T(1.0, z, y_true)
    nll_after = nll_with_T(T_star, z, y_true)

    # 4) Temperature Scaling 적용
    p_cal = 1.0 / (1.0 + np.exp(-z / T_star))
    p_cal = np.clip(p_cal, EPS, 1.0 - EPS)

    brier_before = brier_score(p_raw, y_true)
    brier_after = brier_score(p_cal, y_true)

    print(f"[INFO] Best T* = {T_star:.6f}")
    print(f"[INFO] NLL before: {nll_before:.6f}  |  after: {nll_after:.6f}")
    print(f"[INFO] Brier before: {brier_before:.6f}  |  after: {brier_after:.6f}")

    # 5) 결과 저장 (CSV + T 정보)
    df["prob_pos_calibrated"] = p_cal
    out_csv = os.path.join(args.out_dir, "result_with_calibrated_prob.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved calibrated CSV → {out_csv}")

    t_info_path = os.path.join(args.out_dir, "temperature_info.txt")
    with open(t_info_path, "w") as f:
        f.write(f"Best T: {T_star:.6f}\n")
        f.write(f"NLL before: {nll_before:.6f}\n")
        f.write(f"NLL after: {nll_after:.6f}\n")
        f.write(f"Brier before: {brier_before:.6f}\n")
        f.write(f"Brier after: {brier_after:.6f}\n")
    print(f"[INFO] Saved temperature info → {t_info_path}")

    # 6) Calibration Plot (Before vs After)
    calib_png = os.path.join(args.out_dir, "calibration_temperature_scaling.png")
    plot_calibration_medical(y_true, p_raw, p_cal, calib_png)

    # 7) DCA (Calibrated probability 기반)
    dca_png = os.path.join(args.out_dir, "dca_curve_calibrated.png")
    dca_csv = os.path.join(args.out_dir, "dca_values_calibrated.csv")
    plot_dca_curve_medical(y_true, p_cal, dca_png, dca_csv)


if __name__ == "__main__":
    main()
