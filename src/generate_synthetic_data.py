"""
generate_synthetic_data.py

Richer synthetic dataset for the AI co-pilot fairness study.

Creates:
- data/raw/sample_decisions_v2.csv  (row-level decisions)
- data/processed/session_metrics_v2.csv  (session x group x access metrics)
"""

from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ---- basic design choices ----
n_workers = 24
n_sessions = 6
items_per_session = 40

workers = [f"W{idx:03d}" for idx in range(1, n_workers + 1)]
groups = [0, 1]  # e.g., community / region
access_tiers = ["feature_2G", "smart_3G", "smart_4G"]
modalities = ["voice", "text"]
arms = ["None", "Rationale", "Calib", "Counter"]

rows = []

for w in workers:
    group = np.random.choice(groups)
    access = np.random.choice(access_tiers, p=[0.4, 0.4, 0.2])
    modality = np.random.choice(modalities, p=[0.6, 0.4])

    for s in range(1, n_sessions + 1):
        for item in range(1, items_per_session + 1):
            arm = np.random.choice(arms, p=[0.2, 0.3, 0.3, 0.2])

            # base difficulty: easier items later in the study
            base_difficulty = 0.4 + 0.4 * (item / items_per_session)

            # ground-truth label: should escalate (1) vs not (0)
            y_true = np.random.binomial(1, base_difficulty)

            # AI quality slightly better for smart_4G + text
            ai_base = 0.72
            if access == "feature_2G":
                ai_base -= 0.08
            if modality == "text":
                ai_base += 0.05

            if arm == "Calib":
                ai_base += 0.03
            elif arm == "Counter":
                ai_base += 0.01

            ai_correct_prob = np.clip(ai_base, 0.55, 0.93)
            ai_correct = np.random.binomial(1, ai_correct_prob)

            # AI confidence: higher if model is correct
            if ai_correct:
                ai_conf = np.random.normal(0.78, 0.08)
            else:
                ai_conf = np.random.normal(0.60, 0.12)
            ai_conf = float(np.clip(ai_conf, 0.05, 0.99))

            # human acceptance depends on arm & confidence
            accept_logit = -0.2 + 1.5 * (ai_conf - 0.5)
            if arm == "Rationale":
                accept_logit += 0.2
            if arm == "Counter":
                # exposed to counter-arguments ⇒ a bit more selective
                accept_logit -= 0.1

            accept_prob = 1 / (1 + np.exp(-accept_logit))
            human_accept = np.random.binomial(1, accept_prob)

            # human unaided base skill (slightly better for experienced sessions)
            human_base = 0.65 + 0.05 * (s / n_sessions)
            if arm == "None":
                human_correct_prob = human_base
            else:
                # with assistance, they improve but can over-rely
                human_correct_prob = human_base + 0.06 * human_accept - 0.03 * (1 - human_accept)

            human_correct_prob = np.clip(human_correct_prob, 0.40, 0.98)
            human_correct = np.random.binomial(1, human_correct_prob)

            high_conf_error = int(ai_conf >= 0.8 and ai_correct == 0)
            caught_high_conf_error = int(high_conf_error and human_correct == 1)

            rows.append(
                {
                    "worker_id": w,
                    "session": s,
                    "item_id": item,
                    "group": group,
                    "access_tier": access,
                    "modality": modality,
                    "assist_arm": arm,
                    "y_true": y_true,
                    "ai_conf": round(ai_conf, 3),
                    "ai_correct": ai_correct,
                    "human_accept": human_accept,
                    "human_correct": human_correct,
                    "high_conf_error": high_conf_error,
                    "caught_high_conf_error": caught_high_conf_error,
                }
            )

df = pd.DataFrame(rows)
raw_path = DATA_RAW / "sample_decisions_v2.csv"
df.to_csv(raw_path, index=False)
print(f"Saved raw decisions: {raw_path}  shape={df.shape}")

# ---- aggregate metrics by session x group x access ----

def tpr_fpr(group_df: pd.DataFrame):
    # treat y_true = 1 as "positive", human_correct as success
    positives = group_df[group_df["y_true"] == 1]
    negatives = group_df[group_df["y_true"] == 0]

    if len(positives) == 0:
        tpr = np.nan
    else:
        tpr = (positives["human_correct"] == 1).mean()

    if len(negatives) == 0:
        fpr = np.nan
    else:
        fpr = (negatives["human_correct"] == 1).mean()

    return tpr, fpr


metric_rows = []

for (s, access), sub in df.groupby(["session", "access_tier"]):
    for g in [0, 1]:
        gdf = sub[sub["group"] == g]
        if len(gdf) == 0:
            continue

        acc = (gdf["human_correct"] == 1).mean()
        tpr, fpr = tpr_fpr(gdf)
        ece_bin = pd.cut(gdf["ai_conf"], bins=np.linspace(0, 1, 11), include_lowest=True)
        calib = gdf.groupby(ece_bin).agg(
            mean_conf=("ai_conf", "mean"),
            mean_corr=("ai_correct", "mean"),
        )
        ece = (calib["mean_conf"] - calib["mean_corr"]).abs().mean()

        metric_rows.append(
            {
                "session": s,
                "access_tier": access,
                "group": g,
                "acc": acc,
                "tpr": tpr,
                "fpr": fpr,
                "ece": ece,
                "n_items": len(gdf),
            }
        )

metrics_df = pd.DataFrame(metric_rows)
metrics_path = DATA_PROCESSED / "session_metrics_v2.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved session metrics: {metrics_path}  shape={metrics_df.shape}")

