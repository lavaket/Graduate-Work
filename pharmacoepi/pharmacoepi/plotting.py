"""Plotting utilities for pharmacoepidemiology analyses."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter


def plot_kaplan_meier(
    df: pd.DataFrame,
    duration_col: str = "time",
    event_col: str = "event",
    group_col: str = "treatment",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot Kaplan-Meier curves by group."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    kmf = KaplanMeierFitter()
    for group, group_df in df.groupby(group_col):
        kmf.fit(group_df[duration_col], group_df[event_col], label=str(group))
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier curves")
    return ax
