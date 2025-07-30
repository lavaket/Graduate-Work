"""Run an example pharmacoepidemiology analysis."""
from __future__ import annotations

import matplotlib.pyplot as plt

from pharmacoepi.data import simulate_cohort
from pharmacoepi.analysis import (
    fit_propensity_score_model,
    calculate_ip_weights,
    fit_weighted_cox_model,
)
from pharmacoepi.plotting import plot_kaplan_meier


def main() -> None:
    df = simulate_cohort(n=2000, seed=42)

    covariates = ["age", "sex"]
    df["propensity_score"] = fit_propensity_score_model(df, covariates)
    df["ip_weight"] = calculate_ip_weights(df["treatment"], df["propensity_score"])

    cph = fit_weighted_cox_model(
        df,
        duration_col="time",
        event_col="event",
        treatment_col="treatment",
        weight_col="ip_weight",
        covariates=covariates,
    )

    print(cph.summary)

    plot_kaplan_meier(df)
    plt.show()


if __name__ == "__main__":
    main()
