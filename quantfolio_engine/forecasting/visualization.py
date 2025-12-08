"""
Visualization utilities for forecasting models.

Provides plotting functions for forecasts, diagnostics, and model comparisons.
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_forecasts(
    y_train: pd.Series,
    y_test: Optional[pd.Series],
    forecast_df: pd.DataFrame,
    model_name: str = "Model",
    title: Optional[str] = None,
    show_intervals: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot forecasts with prediction intervals.

    Args:
        y_train: Training data
        y_test: Test data (actual values)
        forecast_df: DataFrame with 'forecast', 'lower', 'upper', 'lower_80', 'upper_80'
        model_name: Name of the model
        title: Plot title (auto-generated if None)
        show_intervals: Whether to show prediction intervals
        ax: Matplotlib axes (creates new if None)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Plot training data
    ax.plot(
        y_train.index, y_train.values, label="Training Data", color="blue", alpha=0.7
    )

    # Plot test data if available
    if y_test is not None:
        ax.plot(y_test.index, y_test.values, label="Actual", color="green", linewidth=2)

    # Plot forecast
    ax.plot(
        forecast_df.index,
        forecast_df["forecast"],
        label=f"{model_name} Forecast",
        color="red",
        linewidth=2,
        linestyle="--",
    )

    # Plot prediction intervals
    if show_intervals:
        # 95% interval
        if "lower" in forecast_df.columns and "upper" in forecast_df.columns:
            ax.fill_between(
                forecast_df.index,
                forecast_df["lower"],
                forecast_df["upper"],
                alpha=0.2,
                color="red",
                label="95% Prediction Interval",
            )

        # 80% interval
        if "lower_80" in forecast_df.columns and "upper_80" in forecast_df.columns:
            ax.fill_between(
                forecast_df.index,
                forecast_df["lower_80"],
                forecast_df["upper_80"],
                alpha=0.3,
                color="red",
                label="80% Prediction Interval",
            )

    # Add vertical line separating train/test
    if y_test is not None:
        split_date = y_train.index[-1]
        ax.axvline(x=split_date, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    if title is None:
        title = f"{model_name} Forecast"
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_diagnostics(
    residuals: pd.Series,
    fitted_values: pd.Series,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot diagnostic plots for model residuals.

    Creates 4-panel diagnostic plot:
    1. Residuals over time
    2. Q-Q plot for normality
    3. ACF of residuals
    4. Residuals vs fitted values

    Args:
        residuals: Model residuals
        fitted_values: Fitted values
        model_name: Name of the model
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"{model_name} - Diagnostic Plots", fontsize=14, fontweight="bold")

    # 1. Residuals over time
    ax1 = axes[0, 0]
    ax1.plot(residuals.index, residuals.values, color="blue", alpha=0.7)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals Over Time")
    ax1.grid(True, alpha=0.3)

    # 2. Q-Q plot
    ax2 = axes[0, 1]
    from scipy import stats

    stats.probplot(residuals.dropna(), dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normality Check)")
    ax2.grid(True, alpha=0.3)

    # 3. ACF of residuals
    ax3 = axes[1, 0]
    try:
        plot_acf(
            residuals.dropna(), lags=min(20, len(residuals) // 2), ax=ax3, alpha=0.05
        )
        ax3.set_title("ACF of Residuals")
    except Exception as e:
        logger.warning(f"ACF plot failed: {e}")
        ax3.text(0.5, 0.5, "ACF plot unavailable", ha="center", va="center")
        ax3.set_title("ACF of Residuals")

    # 4. Residuals vs fitted values
    ax4 = axes[1, 1]
    # Align indices
    common_idx = residuals.index.intersection(fitted_values.index)
    if len(common_idx) > 0:
        ax4.scatter(
            fitted_values.loc[common_idx],
            residuals.loc[common_idx],
            alpha=0.6,
            s=20,
        )
        ax4.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax4.set_xlabel("Fitted Values")
        ax4.set_ylabel("Residuals")
        ax4.set_title("Residuals vs Fitted Values")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No overlapping data", ha="center", va="center")

    plt.tight_layout()
    return fig


def plot_model_comparison(
    comparison_results: Dict[str, Dict],
    metric: str = "rmse",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot model comparison bar chart.

    Args:
        comparison_results: Dictionary mapping model names to evaluation results
        metric: Metric to compare ('rmse', 'mae', 'mape', 'aic', 'bic')
        title: Plot title (auto-generated if None)
        ax: Matplotlib axes (creates new if None)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    model_names = []
    metric_values = []

    for model_name, results in comparison_results.items():
        model_names.append(model_name)

        if metric in ["aic", "bic"]:
            # Get from model_info
            value = results.get("model_info", {}).get(metric)
        else:
            # Get from metrics
            value = results.get("metrics", {}).get(metric)

        metric_values.append(value)

    # Filter out None values
    valid_data = [
        (name, val) for name, val in zip(model_names, metric_values) if val is not None
    ]
    if not valid_data:
        logger.warning("No valid data for comparison")
        return ax

    model_names, metric_values = zip(*valid_data)

    # Create bar plot
    bars = ax.bar(model_names, metric_values, alpha=0.7, edgecolor="black")
    ax.set_ylabel(metric.upper())
    if title is None:
        title = f"Model Comparison - {metric.upper()}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return ax


def plot_time_series_exploration(
    y: pd.Series,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot time series exploration (time plot, ACF, PACF, decomposition).

    Args:
        y: Time series
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if title is None:
        title = "Time Series Exploration"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Time series plot
    ax1 = axes[0, 0]
    ax1.plot(y.index, y.values, color="blue", linewidth=1.5)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.set_title("Time Series")
    ax1.grid(True, alpha=0.3)

    # 2. ACF
    ax2 = axes[0, 1]
    try:
        plot_acf(y.dropna(), lags=min(20, len(y) // 2), ax=ax2, alpha=0.05)
        ax2.set_title("ACF")
    except Exception as e:
        logger.warning(f"ACF plot failed: {e}")
        ax2.text(0.5, 0.5, "ACF plot unavailable", ha="center", va="center")

    # 3. PACF
    ax3 = axes[1, 0]
    try:
        plot_pacf(y.dropna(), lags=min(20, len(y) // 2), ax=ax3, alpha=0.05)
        ax3.set_title("PACF")
    except Exception as e:
        logger.warning(f"PACF plot failed: {e}")
        ax3.text(0.5, 0.5, "PACF plot unavailable", ha="center", va="center")

    # 4. Decomposition (if enough data)
    ax4 = axes[1, 1]
    try:
        if len(y.dropna()) >= 24:  # Need at least 2 periods for decomposition
            freq = pd.infer_freq(y.index) or "M"
            if freq.startswith("M"):
                period = 12
            elif freq.startswith("Q"):
                period = 4
            else:
                period = 12

            decomposition = seasonal_decompose(
                y.dropna(), model="additive", period=period
            )
            ax4.plot(
                decomposition.trend.index,
                decomposition.trend.values,
                label="Trend",
                color="blue",
            )
            ax4.plot(
                decomposition.seasonal.index,
                decomposition.seasonal.values,
                label="Seasonal",
                color="green",
            )
            ax4.plot(
                decomposition.resid.index,
                decomposition.resid.values,
                label="Residual",
                color="red",
                alpha=0.5,
            )
            ax4.set_xlabel("Date")
            ax4.set_ylabel("Value")
            ax4.set_title("Decomposition")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "Insufficient data\nfor decomposition",
                ha="center",
                va="center",
            )
    except Exception as e:
        logger.warning(f"Decomposition failed: {e}")
        ax4.text(0.5, 0.5, "Decomposition unavailable", ha="center", va="center")

    plt.tight_layout()
    return fig
