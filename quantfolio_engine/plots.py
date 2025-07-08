"""
Plotting utilities for quantfolio engine.

This module provides various plotting functions for portfolio analysis,
backtesting results, and optimization outputs.
"""

from typing import Dict, Optional, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_backtest_results(
    performance_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Plot comprehensive backtest results with benchmark comparison.

    Args:
        performance_df: DataFrame with backtest performance data
        benchmark_df: Optional DataFrame with benchmark performance data
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle("Walk-Forward Backtest Results", fontsize=16, fontweight="bold")

    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    cumulative_returns = (1 + performance_df["total_return"]).cumprod()
    ax1.plot(
        cumulative_returns.index,
        cumulative_returns.values,
        label="Portfolio",
        linewidth=2,
        color="blue",
    )

    if benchmark_df is not None and "benchmark_total_return" in performance_df.columns:
        benchmark_cumulative = (1 + performance_df["benchmark_total_return"]).cumprod()
        ax1.plot(
            benchmark_cumulative.index,
            benchmark_cumulative.values,
            label="Benchmark",
            linewidth=2,
            color="red",
            linestyle="--",
        )

    ax1.set_title("Cumulative Returns")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Rolling Sharpe Ratio
    ax2 = axes[0, 1]
    if "sharpe_ratio" in performance_df.columns:
        ax2.plot(
            performance_df.index,
            performance_df["sharpe_ratio"],
            color="green",
            linewidth=2,
        )
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.set_title("Rolling Sharpe Ratio")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.grid(True, alpha=0.3)

    # 3. Drawdown
    ax3 = axes[1, 0]
    if "max_drawdown" in performance_df.columns:
        ax3.fill_between(
            performance_df.index,
            performance_df["max_drawdown"],
            0,
            alpha=0.3,
            color="red",
        )
        ax3.plot(
            performance_df.index,
            performance_df["max_drawdown"],
            color="red",
            linewidth=1,
        )
        ax3.set_title("Drawdown")
        ax3.set_ylabel("Drawdown")
        ax3.grid(True, alpha=0.3)

    # 4. Volatility
    ax4 = axes[1, 1]
    if "volatility" in performance_df.columns:
        ax4.plot(
            performance_df.index,
            performance_df["volatility"],
            color="purple",
            linewidth=2,
        )
        ax4.set_title("Rolling Volatility")
        ax4.set_ylabel("Volatility")
        ax4.grid(True, alpha=0.3)

    # 5. Turnover
    ax5 = axes[2, 0]
    if "turnover" in performance_df.columns:
        ax5.bar(
            performance_df.index, performance_df["turnover"], alpha=0.7, color="orange"
        )
        ax5.set_title("Portfolio Turnover")
        ax5.set_ylabel("Turnover")
        ax5.grid(True, alpha=0.3)

    # 6. Transaction Costs
    ax6 = axes[2, 1]
    if "transaction_cost" in performance_df.columns:
        cumulative_costs = performance_df["transaction_cost"].cumsum()
        ax6.plot(
            cumulative_costs.index, cumulative_costs.values, color="brown", linewidth=2
        )
        ax6.set_title("Cumulative Transaction Costs")
        ax6.set_ylabel("Transaction Costs")
        ax6.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Backtest plot saved to {save_path}")

    plt.show()


def plot_performance_comparison(
    performance_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot performance comparison between portfolio and benchmark.

    Args:
        performance_df: DataFrame with performance data
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Portfolio vs Benchmark Performance", fontsize=14, fontweight="bold")

    # 1. Returns Comparison
    ax1 = axes[0, 0]
    if (
        "total_return" in performance_df.columns
        and "benchmark_total_return" in performance_df.columns
    ):
        x = range(len(performance_df))
        ax1.bar(
            [i - 0.2 for i in x],
            performance_df["total_return"],
            width=0.4,
            label="Portfolio",
            alpha=0.7,
            color="blue",
        )
        ax1.bar(
            [i + 0.2 for i in x],
            performance_df["benchmark_total_return"],
            width=0.4,
            label="Benchmark",
            alpha=0.7,
            color="red",
        )
        ax1.set_title("Period Returns Comparison")
        ax1.set_ylabel("Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Sharpe Ratio Comparison
    ax2 = axes[0, 1]
    if (
        "sharpe_ratio" in performance_df.columns
        and "benchmark_sharpe_ratio" in performance_df.columns
    ):
        ax2.scatter(
            performance_df["benchmark_sharpe_ratio"],
            performance_df["sharpe_ratio"],
            alpha=0.6,
            color="green",
        )
        ax2.plot([-2, 2], [-2, 2], "r--", alpha=0.5)  # 45-degree line
        ax2.set_xlabel("Benchmark Sharpe Ratio")
        ax2.set_ylabel("Portfolio Sharpe Ratio")
        ax2.set_title("Sharpe Ratio Comparison")
        ax2.grid(True, alpha=0.3)

    # 3. Risk-Return Scatter
    ax3 = axes[1, 0]
    if (
        "volatility" in performance_df.columns
        and "total_return" in performance_df.columns
    ):
        ax3.scatter(
            performance_df["volatility"],
            performance_df["total_return"],
            alpha=0.6,
            color="purple",
        )
        ax3.set_xlabel("Volatility")
        ax3.set_ylabel("Return")
        ax3.set_title("Risk-Return Scatter")
        ax3.grid(True, alpha=0.3)

    # 4. Hit Rate Analysis
    ax4 = axes[1, 1]
    if (
        "total_return" in performance_df.columns
        and "benchmark_total_return" in performance_df.columns
    ):
        outperformance = (
            performance_df["total_return"] > performance_df["benchmark_total_return"]
        )
        hit_rate = outperformance.mean()
        ax4.pie(
            [hit_rate, 1 - hit_rate],
            labels=["Outperform", "Underperform"],
            autopct="%1.1f%%",
            colors=["green", "red"],
        )
        ax4.set_title(f"Hit Rate: {hit_rate:.1%}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Performance comparison plot saved to {save_path}")

    plt.show()


def plot_weight_evolution(
    weight_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 8),
) -> None:
    """
    Plot portfolio weight evolution over time.

    Args:
        weight_df: DataFrame with weight history
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    plt.style.use("seaborn-v0_8")

    # Get asset columns (exclude date)
    asset_cols = [col for col in weight_df.columns if col != "date"]

    fig, ax = plt.subplots(figsize=figsize)

    # Stack plot of weights
    weight_df.set_index("date")[asset_cols].plot(
        kind="area", stacked=True, alpha=0.7, ax=ax
    )

    ax.set_title("Portfolio Weight Evolution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Weight evolution plot saved to {save_path}")

    plt.show()


def plot_aggregate_metrics(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot aggregate performance metrics.

    Args:
        metrics: Dictionary of aggregate metrics
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Aggregate Performance Metrics", fontsize=14, fontweight="bold")

    # 1. Key Metrics Bar Chart
    ax1 = axes[0, 0]
    key_metrics = [
        "avg_total_return",
        "avg_sharpe_ratio",
        "avg_sortino_ratio",
        "hit_ratio",
    ]
    metric_names = ["Total Return", "Sharpe Ratio", "Sortino Ratio", "Hit Ratio"]
    values = [metrics.get(metric, 0) for metric in key_metrics]

    bars = ax1.bar(
        metric_names, values, color=["blue", "green", "orange", "red"], alpha=0.7
    )
    ax1.set_title("Key Performance Metrics")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    # 2. Risk Metrics
    ax2 = axes[0, 1]
    risk_metrics = ["avg_volatility", "worst_max_drawdown"]
    risk_names = ["Volatility", "Max Drawdown"]
    risk_values = [metrics.get(metric, 0) for metric in risk_metrics]

    ax2.bar(risk_names, risk_values, color=["purple", "brown"], alpha=0.7)
    ax2.set_title("Risk Metrics")
    ax2.set_ylabel("Value")
    ax2.grid(True, alpha=0.3)

    # 3. Transaction Costs
    ax3 = axes[1, 0]
    cost_metrics = ["total_transaction_costs", "avg_transaction_cost"]
    cost_names = ["Total Costs", "Avg Cost/Period"]
    cost_values = [metrics.get(metric, 0) for metric in cost_metrics]

    ax3.bar(cost_names, cost_values, color=["darkblue", "lightblue"], alpha=0.7)
    ax3.set_title("Transaction Costs")
    ax3.set_ylabel("Cost")
    ax3.grid(True, alpha=0.3)

    # 4. Turnover Metrics
    ax4 = axes[1, 1]
    turnover_metrics = ["total_turnover", "avg_turnover"]
    turnover_names = ["Total Turnover", "Avg Turnover/Period"]
    turnover_values = [metrics.get(metric, 0) for metric in turnover_metrics]

    ax4.bar(
        turnover_names, turnover_values, color=["darkgreen", "lightgreen"], alpha=0.7
    )
    ax4.set_title("Portfolio Turnover")
    ax4.set_ylabel("Turnover")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Aggregate metrics plot saved to {save_path}")

    plt.show()


def plot_return_distribution(
    performance_df: pd.DataFrame,
    save_path: str = None,
    bins: int = 30,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot histogram comparing the distribution of portfolio and benchmark returns.

    Args:
        performance_df: DataFrame with 'total_return' and 'benchmark_total_return' columns
        save_path: Optional path to save the plot
        bins: Number of bins for the histogram
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots(figsize=figsize)

    if "total_return" in performance_df.columns:
        sns.histplot(
            performance_df["total_return"],
            bins=bins,
            color="blue",
            label="Portfolio",
            kde=True,
            stat="density",
            alpha=0.6,
            ax=ax,
        )
    if "benchmark_total_return" in performance_df.columns:
        sns.histplot(
            performance_df["benchmark_total_return"],
            bins=bins,
            color="red",
            label="Benchmark",
            kde=True,
            stat="density",
            alpha=0.4,
            ax=ax,
        )

    ax.set_title("Distribution of Returns: Portfolio vs Benchmark")
    ax.set_xlabel("Return per Period")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Return distribution plot saved to {save_path}")
    plt.show()
