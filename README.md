# AI Co-pilot: Synthetic Fairness & Skill-Drift Dataset

> Status: **Research prototype** supporting a PhD application writing sample.  
> All data are **synthetic only**; no real human or patient data are included.

This repository contains a small, self-contained synthetic dataset and generator
script for an AI co-pilot study on:

- fairness across groups and access strata (device / connectivity / modality),
- calibration (how well confidence matches accuracy),
- and indicators of **skill drift vs. skill lift** for human decision-makers.

It is designed to accompany a research **writing sample** on AI co-pilots,
community health workers (ASHAs), and access-aware fairness in low-resource
settings.

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── src
│   └── generate_synthetic_data.py
└── data
    ├── raw
    │   └── sample_decisions_v2.csv        # row-level synthetic decisions
    └── processed
        └── session_metrics_v2.csv         # session × group × access metrics
src/generate_synthetic_data.py
Script that generates a richer synthetic dataset aligned with the paper’s
setting: multiple workers, sessions, assistance arms, groups, and access tiers.

data/raw/sample_decisions_v2.csv
Row-level synthetic decisions.

data/processed/session_metrics_v2.csv
Aggregated session-level metrics used for fairness / calibration summaries.Data Design (High-Level)

The generator script simulates:

Workers: 24 workers (W001–W024)

Sessions: 6 sessions per worker

Items per session: 40 tasks
→ Total: 24 × 6 × 40 = 5,760 synthetic decisions

Each decision row includes:

worker_id – worker identifier

session – session index

item_id – item within the session

group – protected group indicator (e.g., community/region)

access_tier – "feature_2G", "smart_3G", "smart_4G"

modality – "voice" vs "text"

assist_arm – assistance style: "None", "Rationale", "Calib", "Counter"

y_true – ground-truth label (e.g., escalate vs. not)

ai_conf – AI confidence score in [0,1]

ai_correct – whether the AI suggestion is correct

human_accept – whether the human accepted the AI suggestion

human_correct – whether the final human decision is correct

high_conf_error – cases where confident AI is wrong

caught_high_conf_error – whether the human caught those errors

The processed file aggregates these into session-level metrics by
session × access_tier × group
cd ~/Documents
git clone <YOUR-REPO-URL> Copilot
cd Copilot

