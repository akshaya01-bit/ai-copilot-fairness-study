import pandas as pd
import numpy as np

from sklearn.metrics import brier_score_loss

# ----------------------------
# Fairness + calibration demo
# ----------------------------
# This script:
#  - reads data/synthetic_copilot_logs_v1.csv
#  - computes overall accuracy
#  - computes TPR/FPR by A_group (fairness)
#  - computes Equalized Odds-style gap
#  - computes ECE (Expected Calibration Error)
# ----------------------------

DATA_PATH = "data/synthetic_copilot_logs_v1.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def compute_accuracy(df):
    correct = (df["human_decision"] == df["y_true"]).mean()
    return correct

def tpr_fpr_by_group(df, group_col="A_group"):
    results = {}
    for g in sorted(df[group_col].unique()):
        sub = df[df[group_col] == g]
        y_true = sub["y_true"].values
        y_pred = sub["human_decision"].values

        # TPR: P(pred=1 | y=1)
        mask_pos = (y_true == 1)
        if mask_pos.sum() > 0:
            tpr = (y_pred[mask_pos] == 1).mean()
        else:
            tpr = np.nan

        # FPR: P(pred=1 | y=0)
        mask_neg = (y_true == 0)
        if mask_neg.sum() > 0:
            fpr = (y_pred[mask_neg] == 1).mean()
        else:
            fpr = np.nan

        results[g] = {"TPR": tpr, "FPR": fpr}

    return results

def eo_gap(results):
    """
    Equalized-odds style gap: |TPR0 - TPR1| + |FPR0 - FPR1|
    assumes binary group {0,1}
    """
    g0, g1 = 0, 1
    tpr_gap = abs(results[g0]["TPR"] - results[g1]["TPR"])
    fpr_gap = abs(results[g0]["FPR"] - results[g1]["FPR"])
    return tpr_gap, fpr_gap, tpr_gap + fpr_gap

def expected_calibration_error(y_true, p_hat, n_bins=10):
    """
    Standard ECE with equally spaced bins in [0,1].
    """
    y_true = np.asarray(y_true)
    p_hat = np.asarray(p_hat)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(p_hat, bin_edges, right=True) - 1  # 0..n_bins-1

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = (indices == b)
        if mask.sum() == 0:
            continue
        conf_avg = p_hat[mask].mean()
        acc_avg = (y_true[mask] == 1).mean()
        weight = mask.sum() / n
        ece += weight * abs(acc_avg - conf_avg)

    return ece

def main():
    df = load_data()
    print("=== Synthetic Co-pilot Logs (v1) ===")
    print(f"Shape: {df.shape}")
    print(df.head())

    # Overall accuracy
    acc = compute_accuracy(df)
    print(f"\nOverall human decision accuracy: {acc:.3f}")

    # Fairness: TPR/FPR by A_group
    fair = tpr_fpr_by_group(df, group_col="A_group")
    print("\nTPR/FPR by A_group:")
    for g, stats in fair.items():
        print(f"  Group {g}: TPR={stats['TPR']:.3f}, FPR={stats['FPR']:.3f}")

    tpr_gap, fpr_gap, eo = eo_gap(fair)
    print(f"\nEqualized-odds style gaps:")
    print(f"  |TPR_0 - TPR_1| = {tpr_gap:.3f}")
    print(f"  |FPR_0 - FPR_1| = {fpr_gap:.3f}")
    print(f"  EO gap (sum)   = {eo:.3f}")

    # Calibration (ECE) w.r.t. model probabilities p_model
    y_true = df["y_true"].values
    p_model = df["p_model"].values
    ece = expected_calibration_error(y_true, p_model, n_bins=10)
    brier = brier_score_loss(y_true, p_model)
    print(f"\nCalibration metrics (using model probabilities p_model):")
    print(f"  ECE  = {ece:.3f}")
    print(f"  Brier score = {brier:.3f}")

if __name__ == "__main__":
    main()
