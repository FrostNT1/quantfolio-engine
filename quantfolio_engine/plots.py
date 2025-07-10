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

    # Plot all three benchmarks
    benchmark_colors = ["red", "green", "orange"]
    benchmark_labels = ["60/40 SPY/TLT", "SPY", "TLT"]
    benchmark_columns = [
        "benchmark_6040_total_return",
        "benchmark_spy_total_return",
        "benchmark_tlt_total_return",
    ]

    for i, (col, label, color) in enumerate(
        zip(benchmark_columns, benchmark_labels, benchmark_colors)
    ):
        if col in performance_df.columns:
            benchmark_cumulative = (1 + performance_df[col]).cumprod()
            ax1.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values,
                label=label,
                linewidth=2,
                color=color,
                linestyle="--" if i == 0 else "-",
                alpha=0.8,
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
    x = range(len(performance_df))
    width = 0.18
    # Portfolio
    if "total_return" in performance_df.columns:
        ax1.bar(
            [i - 0.27 for i in x],
            performance_df["total_return"],
            width=width,
            label="Portfolio",
            color="blue",
            alpha=0.8,
        )
    # Benchmarks
    benchmarks = [
        ("benchmark_6040_total_return", "60/40 SPY/TLT", "red", -0.09),
        ("benchmark_spy_total_return", "SPY", "green", 0.09),
        ("benchmark_tlt_total_return", "TLT", "orange", 0.27),
    ]
    for col, label, color, offset in benchmarks:
        if col in performance_df.columns:
            ax1.bar(
                [i + offset for i in x],
                performance_df[col],
                width=width,
                label=label,
                color=color,
                alpha=0.7,
            )
    ax1.set_title("Period Returns Comparison")
    ax1.set_ylabel("Return")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Sharpe Ratio Comparison
    ax2 = axes[0, 1]
    if "sharpe_ratio" in performance_df.columns:
        benchmark_colors = ["red", "green", "orange"]
        benchmark_labels = ["60/40 SPY/TLT", "SPY", "TLT"]
        benchmark_columns = [
            "benchmark_6040_sharpe_ratio",
            "benchmark_spy_sharpe_ratio",
            "benchmark_tlt_sharpe_ratio",
        ]

        for col, label, color in zip(
            benchmark_columns, benchmark_labels, benchmark_colors
        ):
            if col in performance_df.columns:
                ax2.scatter(
                    performance_df[col],
                    performance_df["sharpe_ratio"],
                    alpha=0.6,
                    color=color,
                    label=label,
                    s=50,
                )

        # Add 45-degree line
        ax2.plot([-2, 2], [-2, 2], "k--", alpha=0.5, label="Equal Performance")
        ax2.set_xlabel("Benchmark Sharpe Ratio")
        ax2.set_ylabel("Portfolio Sharpe Ratio")
        ax2.set_title("Sharpe Ratio Comparison")
        ax2.legend()
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
    if "total_return" in performance_df.columns:
        benchmark_columns = [
            "benchmark_6040_total_return",
            "benchmark_spy_total_return",
            "benchmark_tlt_total_return",
        ]
        benchmark_labels = ["60/40 SPY/TLT", "SPY", "TLT"]

        hit_rates = []
        available_labels = []

        for col, label in zip(benchmark_columns, benchmark_labels):
            if col in performance_df.columns:
                outperformance = performance_df["total_return"] > performance_df[col]
                hit_rate = outperformance.mean()
                hit_rates.append(hit_rate)
                available_labels.append(f"{label}\n{hit_rate:.1%}")

        if hit_rates:
            colors = ["green", "blue", "orange"]
            ax4.pie(
                hit_rates,
                labels=available_labels,
                autopct="%1.1f%%",
                colors=colors[: len(hit_rates)],
                startangle=90,
            )
            ax4.set_title("Hit Rate vs Benchmarks")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Performance comparison plot saved to {save_path}")

    plt.show()


def plot_benchmark_comparison(
    performance_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Plot comprehensive benchmark comparison with all three benchmarks.

    Args:
        performance_df: DataFrame with performance data including all benchmarks
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Multi-Benchmark Performance Comparison", fontsize=16, fontweight="bold"
    )

    # 1. Cumulative Returns Comparison
    ax1 = axes[0, 0]
    portfolio_cumulative = (1 + performance_df["total_return"]).cumprod()
    ax1.plot(
        portfolio_cumulative.index,
        portfolio_cumulative.values,
        label="Portfolio",
        linewidth=3,
        color="blue",
    )

    benchmark_colors = ["red", "green", "orange"]
    benchmark_labels = ["60/40 SPY/TLT", "SPY", "TLT"]
    benchmark_columns = [
        "benchmark_6040_total_return",
        "benchmark_spy_total_return",
        "benchmark_tlt_total_return",
    ]

    for col, label, color in zip(benchmark_columns, benchmark_labels, benchmark_colors):
        if col in performance_df.columns:
            benchmark_cumulative = (1 + performance_df[col]).cumprod()
            ax1.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values,
                label=label,
                linewidth=2,
                color=color,
                linestyle="--",
                alpha=0.8,
            )

    ax1.set_title("Cumulative Returns Comparison")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Excess Returns Over Time
    ax2 = axes[0, 1]
    for col, label, color in zip(benchmark_columns, benchmark_labels, benchmark_colors):
        if col in performance_df.columns:
            excess_returns = performance_df["total_return"] - performance_df[col]
            ax2.plot(
                excess_returns.index,
                excess_returns.cumsum(),
                label=f"vs {label}",
                color=color,
                linewidth=2,
            )

    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_title("Cumulative Excess Returns")
    ax2.set_ylabel("Cumulative Excess Return")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Sharpe Ratio Comparison
    ax3 = axes[1, 0]
    if "sharpe_ratio" in performance_df.columns:
        benchmark_sharpe_cols = [
            "benchmark_6040_sharpe_ratio",
            "benchmark_spy_sharpe_ratio",
            "benchmark_tlt_sharpe_ratio",
        ]

        for col, label, color in zip(
            benchmark_sharpe_cols, benchmark_labels, benchmark_colors
        ):
            if col in performance_df.columns:
                ax3.scatter(
                    performance_df[col],
                    performance_df["sharpe_ratio"],
                    alpha=0.6,
                    color=color,
                    label=label,
                    s=50,
                )

        ax3.plot([-2, 2], [-2, 2], "k--", alpha=0.5, label="Equal Performance")
        ax3.set_xlabel("Benchmark Sharpe Ratio")
        ax3.set_ylabel("Portfolio Sharpe Ratio")
        ax3.set_title("Sharpe Ratio Comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Hit Rate Analysis
    ax4 = axes[1, 1]
    if "total_return" in performance_df.columns:
        hit_rates = []
        available_labels = []

        for col, label in zip(benchmark_columns, benchmark_labels):
            if col in performance_df.columns:
                outperformance = performance_df["total_return"] > performance_df[col]
                hit_rate = outperformance.mean()
                hit_rates.append(hit_rate)
                available_labels.append(f"{label}\n{hit_rate:.1%}")

        if hit_rates:
            colors = ["green", "blue", "orange"]
            ax4.pie(
                hit_rates,
                labels=available_labels,
                autopct="%1.1f%%",
                colors=colors[: len(hit_rates)],
                startangle=90,
            )
            ax4.set_title("Hit Rate vs Benchmarks")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Benchmark comparison plot saved to {save_path}")

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

    # 4. Benchmark Comparison
    ax4 = axes[1, 1]
    benchmark_metrics = [
        "excess_return_vs_6040",
        "excess_return_vs_spy",
        "excess_return_vs_tlt",
    ]
    benchmark_names = ["vs 60/40", "vs SPY", "vs TLT"]
    benchmark_values = [metrics.get(metric, 0) for metric in benchmark_metrics]
    benchmark_colors = ["red", "green", "orange"]

    bars = ax4.bar(benchmark_names, benchmark_values, color=benchmark_colors, alpha=0.7)
    ax4.set_title("Excess Returns vs Benchmarks")
    ax4.set_ylabel("Excess Return")
    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, benchmark_values):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.001 if height >= 0 else -0.001),
            f"{value:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Aggregate metrics plot saved to {save_path}")

    plt.show()


def plot_return_distribution(
    performance_df: pd.DataFrame,
    save_path: Optional[str] = None,
    bins: int = 30,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot histogram and KDE of period returns for portfolio and benchmarks.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=figsize)
    plt.style.use("seaborn-v0_8")

    # Portfolio
    if "total_return" in performance_df.columns:
        sns.histplot(
            performance_df["total_return"],
            bins=bins,
            kde=True,
            color="blue",
            label="Portfolio",
            stat="density",
            alpha=0.5,
        )

    # Benchmarks
    benchmark_info = [
        ("benchmark_6040_total_return", "60/40 SPY/TLT", "red"),
        ("benchmark_spy_total_return", "SPY", "green"),
        ("benchmark_tlt_total_return", "TLT", "orange"),
    ]
    for col, label, color in benchmark_info:
        if col in performance_df.columns:
            sns.histplot(
                performance_df[col],
                bins=bins,
                kde=True,
                color=color,
                label=label,
                stat="density",
                alpha=0.4,
            )

    plt.title("Distribution of Returns: Portfolio vs Benchmarks")
    plt.xlabel("Period Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
