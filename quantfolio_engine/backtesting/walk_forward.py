"""
Walk-forward backtesting framework.

This module provides a comprehensive walk-forward backtesting framework
with configurable train/test windows and rebalance frequencies.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd

from quantfolio_engine.optimizer.portfolio_engine import PortfolioOptimizationEngine

from .data_validator import DataValidator


def set_log_level(debug: bool):
    from loguru import logger

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="DEBUG" if debug else "INFO",
        colorize=True,
    )


class WalkForwardBacktester:
    """
    Walk-forward backtesting framework.

    Args:
        debug (bool): If True, sets logger to DEBUG level for verbose output. Default is False.
    """

    def __init__(
        self,
        train_years: int = 8,
        test_years: int = 2,
        rebalance_frequency: str = "monthly",
        transaction_costs: Optional[Dict[str, float]] = None,
        risk_free_rate: float = 0.045,
        max_weight: float = 0.3,
        min_weight: float = 0.05,
        max_volatility: float = 0.15,
        random_state: Optional[int] = None,
        debug: bool = False,
    ):
        set_log_level(debug)
        self.train_years = train_years
        self.test_years = test_years
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_volatility = max_volatility
        self.random_state = random_state

        # Data frequency and periods per year (will be set when data is provided)
        self.data_frequency: str = (
            "monthly"  # Default value, will be updated by _infer_data_frequency
        )
        self.periods_per_year = 12  # Default to monthly if not inferred

        # Set default transaction costs if not provided
        if transaction_costs is None:
            self.transaction_costs = {
                "ETFs": 0.0005,  # 5 bps (SPY, TLT, GLD)
                "Large_Cap": 0.001,  # 10 bps (AAPL, MSFT, JPM)
                "Small_Cap": 0.002,  # 20 bps (IWM)
                "International": 0.0025,  # 25 bps (EFA)
                "Commodities": 0.0015,  # 15 bps (XLE)
            }
        else:
            self.transaction_costs = transaction_costs

        # Initialize components
        self.data_validator = DataValidator(
            min_training_years=train_years,
            min_testing_years=test_years,
            rebalance_frequency=rebalance_frequency,
        )

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.weight_history: List[Dict[str, Any]] = []
        self.benchmark_history: List[Dict[str, Any]] = []

    def run_backtest(
        self,
        returns_df: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame] = None,
        factor_regimes: Optional[pd.DataFrame] = None,
        sentiment_scores: Optional[pd.DataFrame] = None,
        macro_data: Optional[pd.DataFrame] = None,
        method: str = "combined",
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Run walk-forward backtest.

        Args:
            returns_df: Asset returns DataFrame
            factor_exposures: Factor exposures DataFrame
            factor_regimes: Factor regimes DataFrame
            sentiment_scores: Sentiment scores DataFrame
            macro_data: Macroeconomic data DataFrame
            method: Optimization method ('black_litterman', 'monte_carlo', 'combined')

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting walk-forward backtest with {method} method...")

        # 0. Infer data frequency and set periods per year
        self._infer_data_frequency(returns_df)

        # 1. Validate data
        is_valid, validation_messages = (
            self.data_validator.validate_data_for_backtesting(
                returns_df,
                factor_exposures,
                factor_regimes,
                sentiment_scores,
                macro_data,
            )
        )

        if not is_valid:
            logger.error("Data validation failed. Cannot proceed with backtest.")
            return {
                "error": "Data validation failed",
                "validation_messages": validation_messages,
            }

        # 2. Get train/test split
        train_end, test_start = self.data_validator.suggest_train_test_split(returns_df)
        logger.info(
            f"Train period: {returns_df.index.min().date()} to {train_end.date()}"
        )
        logger.info(
            f"Test period: {test_start.date()} to {returns_df.index.max().date()}"
        )

        # 3. Initialize performance tracking
        self.performance_history = []
        self.weight_history = []
        self.benchmark_history = []

        # 4. Run walk-forward backtest
        self._run_walk_forward(
            returns_df,
            factor_exposures,
            factor_regimes,
            sentiment_scores,
            macro_data,
            method,
        )

        # 5. Calculate aggregate performance metrics
        aggregate_metrics = self._calculate_aggregate_metrics()

        return {
            "performance_history": pd.DataFrame(self.performance_history),
            "weight_history": pd.DataFrame(self.weight_history),
            "benchmark_history": pd.DataFrame(self.benchmark_history),
            "aggregate_metrics": aggregate_metrics,
            "validation_messages": validation_messages,
        }

    def _run_walk_forward(
        self,
        returns_df: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame],
        factor_regimes: Optional[pd.DataFrame],
        sentiment_scores: Optional[pd.DataFrame],
        macro_data: Optional[pd.DataFrame],
        method: str,
    ) -> None:
        """Run the actual walk-forward backtest."""

        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(returns_df)
        logger.info(f"Running {len(rebalance_dates)} rebalance periods...")
        prev_weights = None
        for i, rebalance_date in enumerate(rebalance_dates):
            logger.info(
                f"Rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date.date()}"
            )
            # Get training data (expanding window)
            # FIXED: Prevent peeking by ensuring train_end is before test_start
            train_end = rebalance_date - pd.Timedelta(days=1)
            train_start = returns_df.index.min()
            train_data = self._get_training_data(
                returns_df,
                factor_exposures,
                factor_regimes,
                sentiment_scores,
                macro_data,
                train_start,
                train_end,
            )
            # Get testing data (next period)
            test_end = self._get_next_rebalance_date(returns_df, rebalance_date)
            # FIXED: Use index-based slicing to get proper test window with month-end data
            # Get the slice from rebalance_date to next_reb, then exclude the first row
            test_data = returns_df.loc[rebalance_date:test_end].iloc[1:]
            if test_data.empty:
                logger.warning(f"No test data available for {rebalance_date.date()}")
                continue
            # Run optimization
            try:
                portfolio_result = self._run_optimization(train_data, method)
                if portfolio_result is None:
                    continue
                # Calculate portfolio returns for transaction cost calculation
                weights = portfolio_result["weights"]
                # FIXED: Handle Series alignment properly to prevent silent errors
                if isinstance(weights, pd.Series):
                    w = weights
                else:
                    w = pd.Series(weights, index=returns_df.columns)
                portfolio_returns = test_data.mul(w, axis=1).sum(axis=1)

                # Calculate test performance
                test_performance = self._calculate_test_performance(
                    portfolio_result["weights"],
                    test_data,
                    rebalance_date,
                    portfolio_returns,
                )

                # Calculate turnover and transaction cost
                weights = portfolio_result["weights"]
                if isinstance(weights, np.ndarray):
                    weights = pd.Series(weights, index=returns_df.columns)
                turnover = 0.0
                tc = 0.0

                if prev_weights is not None:
                    # Calculate turnover
                    turnover = (weights - prev_weights).abs().sum()

                    # Calculate transaction costs
                    weight_changes = (weights - prev_weights).abs()
                    for asset, weight_change in weight_changes.items():
                        tc += self._get_transaction_cost(asset) * weight_change
                else:
                    # First rebalance - no turnover
                    turnover = 0.0
                    tc = 0.0

                # Store performance metrics
                performance_data = {
                    "date": rebalance_date,
                    "method": method,
                    "total_return": test_performance["total_return"],
                    "avg_return": test_performance["avg_return"],
                    "volatility": test_performance["volatility"],
                    "sharpe_ratio": test_performance["sharpe_ratio"],
                    "sortino_ratio": test_performance["sortino_ratio"],
                    "max_drawdown": test_performance["max_drawdown"],
                    "calmar_ratio": test_performance["calmar_ratio"],
                    "turnover": turnover,
                    "transaction_cost": tc,
                    "net_total_return": test_performance["total_return"] - tc,
                    "period_returns": portfolio_returns.tolist(),  # Store individual period returns
                    # Add annualized metrics
                    "avg_return_annual": test_performance["avg_return_annual"],
                    "volatility_annual": test_performance["volatility_annual"],
                    "sharpe_ratio_annual": test_performance["sharpe_ratio_annual"],
                    "sortino_ratio_annual": test_performance["sortino_ratio_annual"],
                }

                # Add benchmark performance
                benchmark_performance = self._calculate_benchmark_performance(
                    test_data, rebalance_date
                )
                performance_data.update(benchmark_performance)

                self.performance_history.append(performance_data)

                # Update prev_weights for next iteration
                prev_weights = weights.copy()
                # Store weights
                weights_dict = {"date": rebalance_date}
                if isinstance(portfolio_result["weights"], np.ndarray):
                    asset_names = list(returns_df.columns)
                    for i, asset in enumerate(asset_names):
                        weights_dict[asset] = portfolio_result["weights"][i]
                else:
                    for asset, weight in portfolio_result["weights"].items():
                        weights_dict[asset] = weight
                self.weight_history.append(weights_dict)
            except Exception as e:
                logger.error(f"Error in rebalance {rebalance_date.date()}: {str(e)}")
                continue
        logger.success(
            f"Walk-forward backtest completed with {len(self.performance_history)} periods"
        )

    def _get_rebalance_dates(self, returns_df: pd.DataFrame) -> List[pd.Timestamp]:
        """Get rebalance dates based on frequency."""
        start_date = returns_df.index.min()
        end_date = returns_df.index.max()

        if self.rebalance_frequency == "monthly":
            freq = "ME"
        elif self.rebalance_frequency == "quarterly":
            freq = "QE"
        elif self.rebalance_frequency == "annual":
            freq = "YE"
        else:
            raise ValueError(f"Invalid rebalance frequency: {self.rebalance_frequency}")

        # Generate rebalance dates
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Filter to dates that have sufficient training data
        min_train_start = start_date + timedelta(days=self.train_years * 365.25)
        rebalance_dates = rebalance_dates[rebalance_dates >= min_train_start]

        return rebalance_dates.tolist()

    def _get_next_rebalance_date(
        self, returns_df: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Timestamp:
        """Get the next rebalance date."""
        if self.rebalance_frequency == "monthly":
            next_date = current_date + pd.DateOffset(months=1)
        elif self.rebalance_frequency == "quarterly":
            next_date = current_date + pd.DateOffset(months=3)
        elif self.rebalance_frequency == "annual":
            next_date = current_date + pd.DateOffset(years=1)
        else:
            raise ValueError(f"Invalid rebalance frequency: {self.rebalance_frequency}")

        # Ensure we don't go beyond available data
        return min(next_date, returns_df.index.max())

    def _get_training_data(
        self,
        returns_df: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame],
        factor_regimes: Optional[pd.DataFrame],
        sentiment_scores: Optional[pd.DataFrame],
        macro_data: Optional[pd.DataFrame],
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        """Get training data for the specified period."""
        train_data = {
            "returns": returns_df.loc[train_start:train_end],
        }

        if factor_exposures is not None:
            # Filter factor data by date column
            factor_exposures["date"] = pd.to_datetime(factor_exposures["date"])
            mask = (factor_exposures["date"] >= train_start) & (
                factor_exposures["date"] <= train_end
            )
            train_data["factor_exposures"] = factor_exposures[mask]

        if factor_regimes is not None:
            # Filter regime data by date column
            factor_regimes["date"] = pd.to_datetime(factor_regimes["date"])
            mask = (factor_regimes["date"] >= train_start) & (
                factor_regimes["date"] <= train_end
            )
            train_data["factor_regimes"] = factor_regimes[mask]

        if sentiment_scores is not None:
            train_data["sentiment_scores"] = sentiment_scores.loc[train_start:train_end]

        if macro_data is not None:
            train_data["macro_data"] = macro_data.loc[train_start:train_end]

        return train_data

    def _get_testing_data(
        self, returns_df: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
    ) -> pd.DataFrame:
        """Get testing data for the specified period."""
        return returns_df.loc[test_start:test_end]

    def _run_optimization(
        self, train_data: Dict[str, pd.DataFrame], method: str
    ) -> Optional[Dict]:
        """Run portfolio optimization on training data."""
        try:
            # Initialize optimization engine
            engine = PortfolioOptimizationEngine(
                method=method,
                risk_free_rate=self.risk_free_rate,
                random_state=self.random_state,
            )

            # Set constraints
            constraints = {
                "max_weight": self.max_weight,
                "min_weight": self.min_weight,
            }

            # Run optimization
            result = engine.optimize_portfolio(
                data=train_data,
                constraints=constraints,
                max_volatility=self.max_volatility,
            )

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return None

    def _calculate_test_performance(
        self,
        weights: Union[pd.Series, np.ndarray],
        test_data: pd.DataFrame,
        rebalance_date: pd.Timestamp,
        portfolio_returns: pd.Series,
    ) -> Dict[str, float]:
        """Calculate performance metrics for test period."""
        try:
            # Define canonical units once per function
            periods = (
                self.periods_per_year
            )  # Based on data frequency, not rebalance frequency

            # Work in per-period units first
            total_return = (1 + portfolio_returns).prod() - 1
            mean_r = portfolio_returns.mean()
            stdev_r = (
                0.0 if len(portfolio_returns) < 2 else portfolio_returns.std(ddof=0)
            )
            rf_period = self.risk_free_rate / periods

            # Calculate per-period Sharpe ratio
            sharpe_ratio = (mean_r - rf_period) / stdev_r if stdev_r > 0 else 0.0

            # Calculate max drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calculate Sortino ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            if len(downside_returns) > 0:
                downside_std = (
                    0.0
                    if len(downside_returns) < 2
                    else abs(downside_returns.std(ddof=0))
                )
                sortino_ratio = (
                    (mean_r - rf_period) / downside_std if downside_std > 0 else 0.0
                )
            else:
                sortino_ratio = 0.0
                downside_std = 0.0

            # Calculate Calmar ratio (per-period cumulative return / max drawdown)
            calmar_ratio = (
                total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            )

            # Annualized versions (only for reporting)
            # Use linear approximation for annualized return: avg_period_return * periods
            mean_r_ann = mean_r * periods
            stdev_r_ann = stdev_r * np.sqrt(periods)
            sharpe_ratio_annual = sharpe_ratio * np.sqrt(periods)

            # Annualized Sortino ratio (ensure positive)
            sortino_ratio_annual = abs(sortino_ratio) * np.sqrt(periods)

            return {
                "total_return": total_return,
                "avg_return": mean_r,
                "volatility": stdev_r,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "calmar_ratio": calmar_ratio,
                # Annualized versions
                "avg_return_annual": mean_r_ann,
                "volatility_annual": stdev_r_ann,
                "sharpe_ratio_annual": sharpe_ratio_annual,
                "sortino_ratio_annual": sortino_ratio_annual,
            }

        except Exception as e:
            logger.error(f"Error calculating test performance: {str(e)}")
            return {
                "total_return": 0.0,
                "avg_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "avg_return_annual": 0.0,
                "volatility_annual": 0.0,
                "sharpe_ratio_annual": 0.0,
                "sortino_ratio_annual": 0.0,
            }

    def _calculate_benchmark_performance(
        self, test_data: pd.DataFrame, rebalance_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate benchmark performance for multiple benchmarks."""
        try:
            # Find SPY and TLT indices
            spy_idx = None
            tlt_idx = None
            for i, col in enumerate(test_data.columns):
                if col == "SPY":
                    spy_idx = i
                elif col == "TLT":
                    tlt_idx = i

            periods = self.periods_per_year if self.periods_per_year is not None else 12
            rf_period = self.risk_free_rate / periods
            results = {}

            # 1. 60/40 SPY/TLT benchmark
            if spy_idx is not None and tlt_idx is not None:
                benchmark_weights_6040 = np.zeros(len(test_data.columns))
                benchmark_weights_6040[spy_idx] = 0.6
                benchmark_weights_6040[tlt_idx] = 0.4
                benchmark_returns_6040 = (test_data * benchmark_weights_6040).sum(
                    axis=1
                )

                benchmark_total_return_6040 = (1 + benchmark_returns_6040).prod() - 1
                benchmark_mean_r_6040 = benchmark_returns_6040.mean()
                benchmark_stdev_r_6040 = (
                    0.0
                    if len(benchmark_returns_6040) < 2
                    else benchmark_returns_6040.std(ddof=0)
                )
                benchmark_sharpe_6040 = (
                    (benchmark_mean_r_6040 - rf_period) / benchmark_stdev_r_6040
                    if benchmark_stdev_r_6040 > 0
                    else 0.0
                )

                results.update(
                    {
                        "benchmark_6040_total_return": benchmark_total_return_6040,
                        "benchmark_6040_avg_return": benchmark_mean_r_6040,
                        "benchmark_6040_volatility": benchmark_stdev_r_6040,
                        "benchmark_6040_sharpe_ratio": benchmark_sharpe_6040,
                        # Legacy keys for backward compatibility
                        "benchmark_total_return": benchmark_total_return_6040,
                        "benchmark_avg_return": benchmark_mean_r_6040,
                        "benchmark_volatility": benchmark_stdev_r_6040,
                        "benchmark_sharpe_ratio": benchmark_sharpe_6040,
                    }
                )
            else:
                # Fallback to equal weight if SPY/TLT not available
                benchmark_weights_6040 = np.ones(len(test_data.columns)) / len(
                    test_data.columns
                )
                benchmark_returns_6040 = (test_data * benchmark_weights_6040).sum(
                    axis=1
                )

                benchmark_total_return_6040 = (1 + benchmark_returns_6040).prod() - 1
                benchmark_mean_r_6040 = benchmark_returns_6040.mean()
                benchmark_stdev_r_6040 = (
                    0.0
                    if len(benchmark_returns_6040) < 2
                    else benchmark_returns_6040.std(ddof=0)
                )
                benchmark_sharpe_6040 = (
                    (benchmark_mean_r_6040 - rf_period) / benchmark_stdev_r_6040
                    if benchmark_stdev_r_6040 > 0
                    else 0.0
                )

                results.update(
                    {
                        "benchmark_6040_total_return": benchmark_total_return_6040,
                        "benchmark_6040_avg_return": benchmark_mean_r_6040,
                        "benchmark_6040_volatility": benchmark_stdev_r_6040,
                        "benchmark_6040_sharpe_ratio": benchmark_sharpe_6040,
                        # Legacy keys for backward compatibility
                        "benchmark_total_return": benchmark_total_return_6040,
                        "benchmark_avg_return": benchmark_mean_r_6040,
                        "benchmark_volatility": benchmark_stdev_r_6040,
                        "benchmark_sharpe_ratio": benchmark_sharpe_6040,
                    }
                )

            # 2. Equity market benchmark (SPY only)
            if spy_idx is not None:
                benchmark_returns_spy = test_data.iloc[:, spy_idx]
                benchmark_total_return_spy = (1 + benchmark_returns_spy).prod() - 1
                benchmark_mean_r_spy = benchmark_returns_spy.mean()
                benchmark_stdev_r_spy = (
                    0.0
                    if len(benchmark_returns_spy) < 2
                    else benchmark_returns_spy.std(ddof=0)
                )
                benchmark_sharpe_spy = (
                    (benchmark_mean_r_spy - rf_period) / benchmark_stdev_r_spy
                    if benchmark_stdev_r_spy > 0
                    else 0.0
                )

                results.update(
                    {
                        "benchmark_spy_total_return": benchmark_total_return_spy,
                        "benchmark_spy_avg_return": benchmark_mean_r_spy,
                        "benchmark_spy_volatility": benchmark_stdev_r_spy,
                        "benchmark_spy_sharpe_ratio": benchmark_sharpe_spy,
                    }
                )
            else:
                # If SPY not available, use first asset as equity proxy
                benchmark_returns_spy = test_data.iloc[:, 0]
                benchmark_total_return_spy = (1 + benchmark_returns_spy).prod() - 1
                benchmark_mean_r_spy = benchmark_returns_spy.mean()
                benchmark_stdev_r_spy = (
                    0.0
                    if len(benchmark_returns_spy) < 2
                    else benchmark_returns_spy.std(ddof=0)
                )
                benchmark_sharpe_spy = (
                    (benchmark_mean_r_spy - rf_period) / benchmark_stdev_r_spy
                    if benchmark_stdev_r_spy > 0
                    else 0.0
                )

                results.update(
                    {
                        "benchmark_spy_total_return": benchmark_total_return_spy,
                        "benchmark_spy_avg_return": benchmark_mean_r_spy,
                        "benchmark_spy_volatility": benchmark_stdev_r_spy,
                        "benchmark_spy_sharpe_ratio": benchmark_sharpe_spy,
                    }
                )

            # 3. Treasury benchmark (TLT only)
            if tlt_idx is not None:
                benchmark_returns_tlt = test_data.iloc[:, tlt_idx]
                benchmark_total_return_tlt = (1 + benchmark_returns_tlt).prod() - 1
                benchmark_mean_r_tlt = benchmark_returns_tlt.mean()
                benchmark_stdev_r_tlt = (
                    0.0
                    if len(benchmark_returns_tlt) < 2
                    else benchmark_returns_tlt.std(ddof=0)
                )
                benchmark_sharpe_tlt = (
                    (benchmark_mean_r_tlt - rf_period) / benchmark_stdev_r_tlt
                    if benchmark_stdev_r_tlt > 0
                    else 0.0
                )

                results.update(
                    {
                        "benchmark_tlt_total_return": benchmark_total_return_tlt,
                        "benchmark_tlt_avg_return": benchmark_mean_r_tlt,
                        "benchmark_tlt_volatility": benchmark_stdev_r_tlt,
                        "benchmark_tlt_sharpe_ratio": benchmark_sharpe_tlt,
                    }
                )
            else:
                # If TLT not available, use second asset as treasury proxy
                if len(test_data.columns) > 1:
                    benchmark_returns_tlt = test_data.iloc[:, 1]
                else:
                    benchmark_returns_tlt = test_data.iloc[
                        :, 0
                    ]  # Use first asset if only one available

                benchmark_total_return_tlt = (1 + benchmark_returns_tlt).prod() - 1
                benchmark_mean_r_tlt = benchmark_returns_tlt.mean()
                benchmark_stdev_r_tlt = (
                    0.0
                    if len(benchmark_returns_tlt) < 2
                    else benchmark_returns_tlt.std(ddof=0)
                )
                benchmark_sharpe_tlt = (
                    (benchmark_mean_r_tlt - rf_period) / benchmark_stdev_r_tlt
                    if benchmark_stdev_r_tlt > 0
                    else 0.0
                )

                results.update(
                    {
                        "benchmark_tlt_total_return": benchmark_total_return_tlt,
                        "benchmark_tlt_avg_return": benchmark_mean_r_tlt,
                        "benchmark_tlt_volatility": benchmark_stdev_r_tlt,
                        "benchmark_tlt_sharpe_ratio": benchmark_sharpe_tlt,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error calculating benchmark performance: {str(e)}")
            return {
                "benchmark_6040_total_return": 0.0,
                "benchmark_6040_avg_return": 0.0,
                "benchmark_6040_volatility": 0.0,
                "benchmark_6040_sharpe_ratio": 0.0,
                "benchmark_spy_total_return": 0.0,
                "benchmark_spy_avg_return": 0.0,
                "benchmark_spy_volatility": 0.0,
                "benchmark_spy_sharpe_ratio": 0.0,
                "benchmark_tlt_total_return": 0.0,
                "benchmark_tlt_avg_return": 0.0,
                "benchmark_tlt_volatility": 0.0,
                "benchmark_tlt_sharpe_ratio": 0.0,
                # Legacy keys for backward compatibility
                "benchmark_total_return": 0.0,
                "benchmark_avg_return": 0.0,
                "benchmark_volatility": 0.0,
                "benchmark_sharpe_ratio": 0.0,
            }

    def _calculate_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate performance metrics across all periods."""
        if not self.performance_history:
            return {}

        df = pd.DataFrame(self.performance_history)

        # Calculate aggregate metrics - average per-period metrics, annualize only for final display
        periods = self.periods_per_year if self.periods_per_year is not None else 12

        # Average per-period metrics
        avg_period_return = df["avg_return"].mean()
        avg_period_volatility = df["volatility"].mean(skipna=True)
        avg_period_sharpe = df["sharpe_ratio"].mean()
        # avg_period_sortino = df["sortino_ratio"].mean() # Unused variable, remove

        # Annualize for final display
        avg_annual_return = avg_period_return * periods
        avg_annual_volatility = avg_period_volatility * np.sqrt(periods)
        avg_annual_sharpe = avg_period_sharpe * np.sqrt(periods)

        aggregate_metrics = {
            "total_periods": len(df),
            "avg_total_return": df["total_return"].mean(),
            "avg_sharpe_ratio": avg_period_sharpe,  # Per-period average
            "avg_calmar_ratio": df["calmar_ratio"].mean(),
            "worst_max_drawdown": df["max_drawdown"].min(),
            "avg_volatility": avg_period_volatility,  # Per-period average
            "hit_ratio": (df["total_return"] > 0).mean(),
            # 60/40 benchmark metrics
            "benchmark_6040_avg_return": df["benchmark_6040_total_return"].mean(),
            "benchmark_6040_avg_sharpe": df["benchmark_6040_sharpe_ratio"].mean(),
            "excess_return_vs_6040": df["total_return"].mean()
            - df["benchmark_6040_total_return"].mean(),
            "excess_sharpe_vs_6040": avg_period_sharpe
            - df["benchmark_6040_sharpe_ratio"].mean(),
            # SPY benchmark metrics
            "benchmark_spy_avg_return": df["benchmark_spy_total_return"].mean(),
            "benchmark_spy_avg_sharpe": df["benchmark_spy_sharpe_ratio"].mean(),
            "excess_return_vs_spy": df["total_return"].mean()
            - df["benchmark_spy_total_return"].mean(),
            "excess_sharpe_vs_spy": avg_period_sharpe
            - df["benchmark_spy_sharpe_ratio"].mean(),
            # TLT benchmark metrics
            "benchmark_tlt_avg_return": df["benchmark_tlt_total_return"].mean(),
            "benchmark_tlt_avg_sharpe": df["benchmark_tlt_sharpe_ratio"].mean(),
            "excess_return_vs_tlt": df["total_return"].mean()
            - df["benchmark_tlt_total_return"].mean(),
            "excess_sharpe_vs_tlt": avg_period_sharpe
            - df["benchmark_tlt_sharpe_ratio"].mean(),
            # Add transaction cost metrics
            "total_transaction_costs": df["transaction_cost"].sum(),
            "avg_transaction_cost": df["transaction_cost"].mean(),
            "total_turnover": df["turnover"].sum(),
            "avg_turnover": df["turnover"].mean(),
            "net_total_return": df["net_total_return"].mean(),
            # Annualized aggregate metrics (for final display)
            "avg_return_annual": avg_annual_return,
            "avg_volatility_annual": avg_annual_volatility,
            "avg_sharpe_ratio_annual": avg_annual_sharpe,
            "avg_sortino_ratio_annual": df["sortino_ratio_annual"].abs().mean(),
            "avg_calmar_ratio_annual": (
                df["total_return"].mean() * periods / abs(df["max_drawdown"].min())
                if df["max_drawdown"].min() != 0
                else 0.0
            ),
        }

        # Calculate aggregate Sortino ratio properly
        # Use cumulative returns and downside deviation across all periods
        try:
            # Get all portfolio returns across all periods
            all_returns = []
            for period_data in self.performance_history:
                period_returns = period_data.get("period_returns", [])
                if period_returns:
                    all_returns.extend(period_returns)

            if all_returns:
                all_returns_array = np.array(all_returns)
                avg_return = np.mean(all_returns_array)
                downside_returns = all_returns_array[all_returns_array < 0]

                if len(downside_returns) > 0:
                    downside_deviation = abs(np.std(downside_returns))
                    if downside_deviation > 0:
                        # FIXED: Use correct risk-free rate periodicity
                        rf_period = self.risk_free_rate / self.periods_per_year
                        aggregate_sortino = (
                            avg_return - rf_period
                        ) / downside_deviation
                    else:
                        aggregate_sortino = 0.0
                else:
                    aggregate_sortino = 0.0
            else:
                aggregate_sortino = 0.0

        except Exception as e:
            logger.warning(f"Error calculating aggregate Sortino ratio: {e}")
            aggregate_sortino = 0.0

        aggregate_metrics["avg_sortino_ratio"] = aggregate_sortino

        # Calculate max drawdown across entire backtest period
        try:
            # Get all portfolio returns across all periods and build cumulative equity curve
            all_returns = []
            for period_data in self.performance_history:
                period_returns = period_data.get("period_returns", [])
                if period_returns:
                    all_returns.extend(period_returns)

            if all_returns:
                # Build cumulative equity curve
                cumulative_returns = np.array(all_returns)
                equity_curve = (1 + cumulative_returns).cumprod()

                # Calculate max drawdown across entire backtest
                running_max = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - running_max) / running_max
                max_drawdown_overall = drawdown.min()

                # Calculate Calmar ratio using overall max drawdown
                total_return_overall = equity_curve[-1] - 1
                calmar_ratio_overall = (
                    total_return_overall / abs(max_drawdown_overall)
                    if max_drawdown_overall != 0
                    else 0.0
                )

                aggregate_metrics["max_drawdown_overall"] = max_drawdown_overall
                aggregate_metrics["calmar_ratio_overall"] = calmar_ratio_overall
            else:
                aggregate_metrics["max_drawdown_overall"] = 0.0
                aggregate_metrics["calmar_ratio_overall"] = 0.0

        except Exception as e:
            logger.warning(f"Error calculating overall max drawdown: {e}")
            aggregate_metrics["max_drawdown_overall"] = 0.0
            aggregate_metrics["calmar_ratio_overall"] = 0.0

        return aggregate_metrics

    def _infer_data_frequency(self, returns_df: pd.DataFrame) -> None:
        """Infer data frequency from the index and set periods_per_year."""
        try:
            # Get the frequency of the index
            freq = pd.infer_freq(returns_df.index)

            if freq is None:
                # Try to infer from the average time difference
                time_diffs = returns_df.index.to_series().diff().dropna()
                avg_days = time_diffs.dt.days.mean()

                if avg_days <= 1:
                    self.data_frequency = "daily"
                    self.periods_per_year = 252
                elif avg_days <= 7:
                    self.data_frequency = "weekly"
                    self.periods_per_year = 52
                elif avg_days <= 35:
                    self.data_frequency = "monthly"
                    self.periods_per_year = 12
                elif avg_days <= 100:
                    self.data_frequency = "quarterly"
                    self.periods_per_year = 4
                else:
                    self.data_frequency = "annual"
                    self.periods_per_year = 1
            else:
                # Use pandas frequency inference
                if freq.startswith("D"):
                    self.data_frequency = "daily"
                    self.periods_per_year = 252
                elif freq.startswith("W"):
                    self.data_frequency = "weekly"
                    self.periods_per_year = 52
                elif freq.startswith("M"):
                    self.data_frequency = "monthly"
                    self.periods_per_year = 12
                elif freq.startswith("Q"):
                    self.data_frequency = "quarterly"
                    self.periods_per_year = 4
                elif freq.startswith("A") or freq.startswith("Y"):
                    self.data_frequency = "annual"
                    self.periods_per_year = 1
                else:
                    # Default to monthly if unclear
                    self.data_frequency = "monthly"
                    self.periods_per_year = 12

            logger.info(
                f"Inferred data frequency: {self.data_frequency} ({self.periods_per_year} periods per year)"
            )

        except Exception as e:
            logger.warning(
                f"Error inferring data frequency: {e}. Defaulting to monthly."
            )
            self.data_frequency = "monthly"
            self.periods_per_year = 12

    def _get_transaction_cost(self, asset: str) -> float:
        """Return transaction cost (as a decimal) for a given asset based on type."""
        # FIXED: Use user-provided transaction costs instead of hard-coded values
        # Try to find exact asset match first
        if asset in self.transaction_costs:
            return self.transaction_costs[asset]

        # Try to match by asset type
        asset_type = self._get_asset_type(asset)
        if asset_type in self.transaction_costs:
            return self.transaction_costs[asset_type]

        # Default fallback
        return self.transaction_costs.get("default", 0.002)

    def _get_asset_type(self, asset: str) -> str:
        """Determine asset type for transaction cost lookup."""
        # Default mapping (can be extended)
        if asset in ["SPY", "TLT", "GLD", "EFA", "IWM", "XLE"]:
            return "ETFs"
        elif asset in ["AAPL", "MSFT", "JPM", "UNH", "WMT", "BA"]:
            return "Large_Cap"
        elif "IWM" in asset or "small" in asset.lower():
            return "Small_Cap"
        elif "EFA" in asset or "international" in asset.lower():
            return "International"
        elif "XLE" in asset or "commodity" in asset.lower():
            return "Commodities"
        else:
            return "default"
