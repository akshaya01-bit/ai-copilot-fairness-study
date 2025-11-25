"""
03_causal_and_power_demo.py

Simple causal + power sketch for the AI co-pilot study.

What it does:
- Simulates a team × session RCT with 2 arms: Control vs Calibrated AI
- Fits a linear model with team and session fixed effects
- Uses robust (HC1) standard errors to avoid cluster-weight bugs
- Prints the estimated treatment effect and an approximate power sketch
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def simulate_dataset(
    n_teams: int = 20,
    n_sessions: int = 10,
    tasks_per_session: int = 40,
    base_acc: float = 0.70,
    true_effect: float = 0.06,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Simulate a team × session × task panel.

    - Half the team–sessions get Control, half get Calibrated assistance.
    - Outcome is a binary "correct decision" indicator.
    """

    rng = np.random.default_rng(seed)

    rows = []
    for team in range(n_teams):
        for s in range(n_sessions):
            # randomize assistance arm at team×session
            arm = rng.choice(["Control", "Calibrated"])
            # baseline ability variation per team
            team_shift = rng.normal(0.0, 0.05)
            # session noise
            session_shift = rng.normal(0.0, 0.03)

            # mean accuracy for this cell
            mu = base_acc + team_shift + session_shift
            if arm == "Calibrated":
                mu += true_effect

            # clip to [0,1]
            mu = float(np.clip(mu, 0.01, 0.99))

            # generate task-level outcomes
            y = rng.binomial(1, mu, size=tasks_per_session)

            for t in range(tasks_per_session):
                rows.append(
                    {
                        "team": team,
                        "session": s,
                        "task": t,
                        "arm": arm,
                        "correct": y[t],
                    }
                )

    df = pd.DataFrame(rows)
    return df


def fit_itt(df: pd.DataFrame):
    """
    Intent-to-treat estimate:
    correct ~ 1 + Calibrated + C(team) + C(session)
    with robust HC1 standard errors.
    """

    df = df.copy()
    df["Calibrated"] = (df["arm"] == "Calibrated").astype(int)

    model = smf.ols(
        "correct ~ Calibrated + C(team) + C(session)", data=df
    ).fit(cov_type="HC1")

    est = model.params["Calibrated"]
    se = model.bse["Calibrated"]
    ci_low = est - 1.96 * se
    ci_high = est + 1.96 * se

    return est, se, ci_low, ci_high, model


def power_simulation(
    n_sims: int = 200,
    true_effect: float = 0.06,
    alpha: float = 0.05,
) -> None:
    """
    Run repeated simulations and report empirical power
    for detecting the calibrated-arm effect at level alpha.
    """

    estimates = []
    p_values = []

    for s in range(n_sims):
        df = simulate_dataset(true_effect=true_effect, seed=100 + s)
        est, se, ci_low, ci_high, model = fit_itt(df)
        estimates.append(est)
        p_values.append(model.pvalues["Calibrated"])

    estimates = np.array(estimates)
    p_values = np.array(p_values)

    power = float(np.mean(p_values < alpha))

    print("=== Power Simulation Summary ===")
    print(f"  True effect (Δ accuracy): {true_effect:.3f}")
    print(f"  Mean estimated effect   : {estimates.mean():.3f}")
    print(f"  SD of estimates         : {estimates.std(ddof=1):.3f}")
    print(f"  Empirical power (α={alpha:.2f}): {power:.3f}")
    print("  2.5% / 97.5% quantiles  : "
          f"[{np.quantile(estimates, 0.025):.3f}, {np.quantile(estimates, 0.975):.3f}]")


def main():
    print("Running one example dataset + ITT estimate...")
    df = simulate_dataset()
    est, se, ci_low, ci_high, model = fit_itt(df)

    print("\n=== ITT estimate for one simulated study ===")
    print(f"Calibrated vs Control effect (Δ accuracy): {est:.3f}")
    print(f"Standard error (HC1)                     : {se:.3f}")
    print(f"95% CI                                   : [{ci_low:.3f}, {ci_high:.3f}]")
    print("\nModel summary (truncated):")
    print(model.summary().tables[1])  # just coefficients table

    print("\nNow running a small power simulation (this may take a few seconds)...\n")
    power_simulation()


if __name__ == "__main__":
    main()
