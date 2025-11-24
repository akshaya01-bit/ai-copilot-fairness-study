import numpy as np
import pandas as pd

# ----------------------------
# Synthetic log generator (v1)
# ----------------------------
# This script generates a synthetic dataset that mirrors the planned
# ASHA setting in the paper:
# - Teams, sessions
# - Assistance arms: None / Rationale / Calib / Counter
# - Access modalities: feature phone / smartphone / IVR / text
# - Protected group A for fairness
# - Outcomes + probabilities for calibration / EO gap / ECE
#
# Output: data/synthetic_copilot_logs_v1.csv
# ----------------------------

RNG_SEED = 42
np.random.seed(RNG_SEED)

# Config
N_TEAMS = 8
N_SESSIONS = 10
TASKS_PER_SESSION = 40  # total rows = 8 * 10 * 40 = 3200

teams = np.arange(1, N_TEAMS + 1)
sessions = np.arange(1, N_SESSIONS + 1)

arms = ["None", "Rationale", "Calib", "Counter"]
modalities = [
    "feature_2G",      # feature phone / 2G
    "smartphone_3G",   # smartphone / 3G
    "ivr_kannada",     # voice / IVR (Kannada)
    "text_hindi"       # text UI (Hindi)
]

rows = []

for team_id in teams:
    # Assign half of teams to group A=0 and half to A=1, just as a simple pattern
    group_A = 0 if team_id <= N_TEAMS / 2 else 1

    for session_id in sessions:
        for task_idx in range(TASKS_PER_SESSION):
            # Random ASHA (worker) id within team
            asha_id = f"T{team_id}_W{np.random.randint(1, 6)}"  # up to 5 per team

            # Assistance arm for this team x session (simple pattern: cycle arms)
            arm = arms[(team_id + session_id + task_idx) % len(arms)]

            # Access modality: random but biased by group A
            # e.g., group A=1 slightly more likely to have better access
            if group_A == 0:
                modality = np.random.choice(
                    modalities,
                    p=[0.4, 0.25, 0.2, 0.15]  # more feature_2G for group 0
                )
            else:
                modality = np.random.choice(
                    modalities,
                    p=[0.2, 0.4, 0.2, 0.2]   # more smartphone_3G for group 1
                )

            # Base difficulty of the task
            difficulty = np.random.beta(2.0, 2.0)  # between 0 and 1

            # True label probability depends on difficulty + group + modality
            # (just a toy structure)
            base_logit = -0.5 + 1.0 * (1 - difficulty)
            base_logit += 0.2 * (1 if group_A == 1 else 0)
            base_logit += 0.1 * (1 if modality in ["smartphone_3G", "ivr_kannada"] else 0)

            def sigmoid(z):
                return 1 / (1 + np.exp(-z))

            p_y1 = sigmoid(base_logit)
            y_true = np.random.binomial(1, p_y1)

            # Model base probability (uncalibrated) as a function of difficulty + arm
            arm_effect = {
                "None": 0.0,
                "Rationale": 0.1,
                "Calib": 0.2,
                "Counter": 0.15
            }[arm]

            model_logit = -0.3 + arm_effect + 0.8 * (1 - difficulty)
            # small fairness issue: group 0 systematically under-estimated
            model_logit += -0.15 if group_A == 0 else 0.0

            p_model_raw = sigmoid(model_logit)

            # Add some miscalibration: e.g., model overconfident for "Calib" is smaller drift
            if arm == "Calib":
                p_model = 0.05 + 0.9 * p_model_raw  # gently calibrated
            else:
                p_model = 0.10 + 1.1 * p_model_raw
            p_model = np.clip(p_model, 1e-3, 1 - 1e-3)

            # AI suggestion label
            ai_suggestion = int(p_model >= 0.5)

            # Human decision:
            # - With assistance arms != "None", humans more likely to follow AI
            # - But occasionally override based on difficulty
            if arm == "None":
                # Unaided: noisy but depends on difficulty & group
                p_human_correct = 0.6 - 0.3 * difficulty + 0.1 * (1 if group_A == 1 else 0)
                p_human_correct = np.clip(p_human_correct, 0.1, 0.9)
                human_decision = np.random.binomial(1, p_human_correct if y_true == 1 else 1 - p_human_correct)
                accepted_suggestion = 0  # no AI suggestion
                second_look = 0
            else:
                # Aided: start from AI suggestion then possibly override
                # override probability higher on harder tasks
                override_prob = 0.05 + 0.4 * difficulty
                did_override = np.random.rand() < override_prob

                if did_override:
                    # with some probability, the override is correct
                    p_override_correct = 0.55 + 0.2 * (1 if group_A == 1 else 0)
                    override_label = np.random.binomial(1, p_override_correct if y_true == 1 else 1 - p_override_correct)
                    human_decision = override_label
                    accepted_suggestion = 0
                else:
                    human_decision = ai_suggestion
                    accepted_suggestion = 1

                # "Second look" triggered when AI is high confidence but difficulty high
                second_look = int((p_model > 0.8) and (difficulty > 0.5))

            # High-confidence AI error
            high_conf_error = int((p_model > 0.8) and (ai_suggestion != y_true))

            # Task time (seconds) – longer for harder tasks, slightly faster with Calib
            base_time = 60 + 40 * difficulty
            if arm == "Calib":
                base_time *= 0.9
            elif arm == "Counter":
                base_time *= 1.05
            time_seconds = np.random.normal(loc=base_time, scale=10.0)

            rows.append({
                "team_id": team_id,
                "session_id": session_id,
                "task_index": task_idx,
                "asha_id": asha_id,
                "A_group": group_A,             # protected group A
                "arm": arm,                     # assistance style
                "modality": modality,           # access modality (M)
                "difficulty": difficulty,       # latent difficulty
                "y_true": y_true,               # true label
                "p_model": float(p_model),      # model probability for y=1
                "ai_suggestion": ai_suggestion,
                "human_decision": human_decision,
                "accepted_suggestion": accepted_suggestion,
                "second_look": second_look,
                "high_conf_error": high_conf_error,
                "time_seconds": time_seconds
            })

df = pd.DataFrame(rows)

output_path = "data/synthetic_copilot_logs_v1.csv"
df.to_csv(output_path, index=False)

print(f"Saved synthetic dataset with {len(df)} rows to {output_path}")
