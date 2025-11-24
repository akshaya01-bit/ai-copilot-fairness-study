# Copilot: Fairness- and Calibration-Aware AI Assistance (Version 1)

This repository contains **Version 1** of the research artifact that accompanies a
writing sample on **AI co-pilots for community health workers (ASHAs) in low-resource
settings**. The focus of this version is to provide a **minimal but complete synthetic
dataset and analysis script** that mirror the methods described in the paper:

- Multi-arm assistance (`None`, `Rationale`, `Calib`, `Counter`)
- Protected group **A** for fairness analysis
- Access modalities (**feature phone / 2G**, **smartphone / 3G**, **IVR (Kannada)**, **text (Hindi)**)
- Metrics: accuracy, TPR/FPR gaps, EO-style gap, calibration (ECE, Brier)

Later versions (v2, v3) will extend this with bandit-style adaptation, dynamic skill
indices (SDI), and richer governance dashboards.

---

## Repository structure (v1)

```text
Copilot/
  data/
    synthetic_copilot_logs_v1.csv      # synthetic deployment-like logs
  src/
    generate_synthetic_data_v1.py      # script to generate synthetic data
    analyze_fairness_calibration_v1.py # fairness + calibration analysis
  requirements.txt
  README.md

data/synthetic_copilot_logs_v1.csv
Synthetic session- and item-level logs reflecting the planned ASHA context:
teams, sessions, assistance arm, protected group A_group, access modality
(modality), human decisions, model probabilities, and timing.

src/generate_synthetic_data_v1.py
Generates the synthetic dataset aligned with the paper’s setting:

Teams and sessions

Assistance arms (None, Rationale, Calib, Counter)

Access modalities (feature_2G, smartphone_3G, ivr_kannada, text_hindi)

Group-level fairness structure (A_group)

Model probabilities p_model with mild miscalibration and group gaps

src/analyze_fairness_calibration_v1.py
Reads the synthetic CSV and computes:

Overall human decision accuracy

TPR/FPR by protected group A_group

Equalized-odds style gap: |TPR₀ − TPR₁| + |FPR₀ − FPR₁|

Expected Calibration Error (ECE) and Brier score using p_model

This mirrors the fairness and calibration measures described in the Methods
section of the writing sample.