"""
analyze_fairness_calibration.py

Loads the richer synthetic data and:
- prints fairness metrics (TPR/FPR gaps)
- computes an access-aware disparity index (ADI)
- saves two plots to figures/:
    1) EO gap by access tier
    2) simple reliability diagram
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW = Path("data/raw/sample_decisions_v2.csv")
METRICS = Path("data/processed/session_metrics_v2.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
df = pd.read_csv(RAW)
metrics = pd.read_csv(METRICS)

print("Raw decisions shape:", df.shape)
print("Session metrics shape:", metrics.shape)

# ---- Equalized-odds style gap by access tier ----
rows = []
for access, sub in metrics.groupby("access_tier"):
    g0 = sub[sub["group"] == 0]
    g1 = sub[sub["group"] == 1]
    if g0.empty or g1.empty:
        continue

    tpr_gap = (g1["tpr"].mean() - g0["tpr"].mean())
    fpr_gap = (g1["fpr"].mean() - g0["fpr"].mean())
    eo_gap = abs(tpr_gap) + abs(fpr_gap)

    rows.append(
        {
            "access_tier": access,
            "tpr_gap": tpr_gap,
            "fpr_gap": fpr_gap,
            "eo_gap": eo_gap,
        }
    )

eo_df = pd.DataFrame(rows)
print("\nEqualized-odds style gaps by access tier:")
print(eo_df.sort_values("eo_gap", ascending=False))

ADI = eo_df["eo_gap"].abs().max()
print(f"\nAccess-aware Disparity Index (ADI) ≈ {ADI:.3f}")

# ---- Plot EO gap by access tier ----
plt.figure()
x = np.arange(len(eo_df))
plt.bar(x - 0.15, eo_df["tpr_gap"], width=0.3, label="TPR gap")
plt.bar(x + 0.15, eo_df["fpr_gap"], width=0.3, label="FPR gap")
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xticks(x, eo_df["access_tier"], rotation=20)
plt.ylabel("Gap (group 1 − group 0)")
plt.title("Equalized-odds style gaps by access tier (synthetic)")
plt.legend()
plt.tight_layout()
eo_fig = FIG_DIR / "eo_gaps_by_access_tier.png"
plt.savefig(eo_fig, dpi=200)
print(f"Saved EO gap plot → {eo_fig}")

# ---- Reliability diagram (overall) ----
bins = np.linspace(0, 1, 11)
df["conf_bin"] = pd.cut(df["ai_conf"], bins=bins, include_lowest=True)
calib = df.groupby("conf_bin").agg(
    mean_conf=("ai_conf", "mean"),
    mean_corr=("ai_correct", "mean"),
    n=("ai_correct", "size"),
).reset_index()

plt.figure()
plt.plot(calib["mean_conf"], calib["mean_corr"], marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Predicted probability (binned mean)")
plt.ylabel("Empirical accuracy")
plt.title("Reliability diagram (synthetic AI co-pilot)")
plt.tight_layout()
rel_fig = FIG_DIR / "reliability_overall.png"
plt.savefig(rel_fig, dpi=200)
print(f"Saved reliability diagram → {rel_fig}")

print("\nDone. Version 2 analysis complete.")

