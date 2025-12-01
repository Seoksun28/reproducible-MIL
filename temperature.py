import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def nll_with_T(T, z, y):
    """
    T: scalar > 0
    z: (N,)  원래 logit difference
    y: (N,)  0/1 label
    """
    p_T = 1.0 / (1.0 + np.exp(-z / T))   # sigmoid(z/T)
    p_T = np.clip(p_T, EPS, 1 - EPS)
    nll = -np.mean(y * np.log(p_T) + (1 - y) * np.log(1 - p_T))
    return nll


def find_best_T(z, y, t_min=0.2, t_max=5.0, num_grid=200):
    """
    아주 단순한 1차원 그리드 서치 + 로컬 refine
    """
    Ts = np.linspace(t_min, t_max, num_grid)
    losses = [nll_with_T(T, z, y) for T in Ts]
    idx = int(np.argmin(losses))
    T_best = Ts[idx]

    # 주변에서 한 번 더 촘촘히 탐색
    local_min = max(t_min, T_best - 0.5)
    local_max = min(t_max, T_best + 0.5)
    Ts_local = np.linspace(local_min, local_max, num_grid)
    losses_local = [nll_with_T(T, z, y) for T in Ts_local]
    idx2 = int(np.argmin(losses_local))
    T_refined = Ts_local[idx2]

    return T_refined


def brier_score(p, y):
    return np.mean((p - y) ** 2)


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


def plot_calibration(p_raw, p_cal, y, out_path):
    """
    before vs after temperature scaling calibration plot
    """
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.linewidth": 1.5,
        "axes.grid": True,
        "grid.linestyle": ":",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6, 6))

    # Ideal line
    ax.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Ideal")

    # Before calibration
    c_raw, m_raw, f_raw = reliability_curve(p_raw, y, n_bins=10)
    ax.plot(m_raw, f_raw, "o-", label="Before TS")

    # After calibration
    c_cal, m_cal, f_cal = reliability_curve(p_cal, y, n_bins=10)
    ax.plot(m_cal, f_cal, "s-", label="After TS")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration plot (Temperature Scaling)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True,
                        help="LOOCV result.csv path")
    parser.add_argument("--out_dir", type=str, default="calibration_out",
                        help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    # 컬럼 이름이 slide,label,pred,prob_pos 라고 가정
    y = df["label"].values.astype(int)
    p_raw = df["prob_pos"].values.astype(float)

    # logit 복원
    z = logit(p_raw)

    # Before metric
    nll_before = nll_with_T(1.0, z, y)
    brier_before = brier_score(p_raw, y)

    # 최적 T 찾기
    T_star = find_best_T(z, y)
    print(f"[INFO] Best temperature T* = {T_star:.4f}")
    df_T_info = os.path.join(args.out_dir, "temperature_info.txt")
    with open(df_T_info, "w") as f:
        f.write(f"Best T: {T_star:.6f}\n")

    # After scaling
    p_cal = 1.0 / (1.0 + np.exp(-z / T_star))
    p_cal = np.clip(p_cal, EPS, 1 - EPS)

    nll_after = nll_with_T(T_star, z, y)
    brier_after = brier_score(p_cal, y)

    print(f"NLL before: {nll_before:.6f}, after: {nll_after:.6f}")
    print(f"Brier before: {brier_before:.6f}, after: {brier_after:.6f}")

    # 결과 CSV 저장
    df["prob_pos_calibrated"] = p_cal
    out_csv = os.path.join(args.out_dir, "result_with_calibrated_prob.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved calibrated CSV to {out_csv}")

    # Calibration plot 저장
    out_plot = os.path.join(args.out_dir, "calibration_temperature_scaling.png")
    plot_calibration(p_raw, p_cal, y, out_plot)
    print(f"[INFO] Saved calibration plot to {out_plot}")


if __name__ == "__main__":
    main()
