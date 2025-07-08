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


class WalkForwardBacktester:
    """
    Walk-forward backtesting framework.

    Implements:
    - Configurable train/test windows
    - Multiple rebalance frequencies
    - Transaction cost modeling
    - Performance tracking
    - Benchmark comparison
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
    ):
        """
        Initialize walk-forward backtester.

        Args:
            train_years: Years of data to use for training
            test_years: Years of data to use for testing
            rebalance_frequency: Rebalance frequency ('monthly', 'quarterly', 'annual')
            transaction_costs: Transaction costs by asset type (bps)
            risk_free_rate: Annual risk-free rate
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            max_volatility: Maximum portfolio volatility
            random_state: Random seed for reproducibility
        """
        self.train_years = train_years
        self.test_years = test_years
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_volatility = max_volatility
        self.random_state = random_state

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
            train_end = rebalance_date
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
            test_start = rebalance_date
            test_end = self._get_next_rebalance_date(returns_df, rebalance_date)
            test_data = self._get_testing_data(returns_df, test_start, test_end)
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
                if isinstance(weights, np.ndarray):
                    weights_array = weights
                else:
                    weights_array = weights.values
                portfolio_returns = (test_data * weights_array).sum(axis=1)

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
            # Calculate metrics using provided portfolio returns
            total_return = (1 + portfolio_returns).prod() - 1
            avg_return = portfolio_returns.mean()
            volatility = portfolio_returns.std()

            if volatility > 0:
                sharpe_ratio = (avg_return - self.risk_free_rate / 12) / volatility
            else:
                sharpe_ratio = 0.0

            # Calculate max drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calculate Sortino ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std()
                if downside_deviation > 0:
                    sortino_ratio = (
                        avg_return - self.risk_free_rate / 12
                    ) / downside_deviation
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = 0.0

            # Calculate Calmar ratio
            if max_drawdown != 0:
                calmar_ratio = total_return / abs(max_drawdown)
            else:
                calmar_ratio = 0.0

            return {
                "total_return": total_return,
                "avg_return": avg_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "calmar_ratio": calmar_ratio,
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
            }

    def _calculate_benchmark_performance(
        self, test_data: pd.DataFrame, rebalance_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate benchmark performance (60/40 portfolio)."""
        try:
            # Simple 60/40 benchmark (60% SPY, 40% TLT if available)
            benchmark_weights = np.zeros(len(test_data.columns))

            # Find SPY and TLT if available
            spy_idx = None
            tlt_idx = None
            for i, col in enumerate(test_data.columns):
                if col == "SPY":
                    spy_idx = i
                elif col == "TLT":
                    tlt_idx = i

            if spy_idx is not None and tlt_idx is not None:
                benchmark_weights[spy_idx] = 0.6
                benchmark_weights[tlt_idx] = 0.4
            else:
                # Equal weight if SPY/TLT not available
                benchmark_weights = np.ones(len(test_data.columns)) / len(
                    test_data.columns
                )

            # Calculate benchmark returns
            benchmark_returns = (test_data * benchmark_weights).sum(axis=1)

            # Calculate benchmark metrics
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            benchmark_avg_return = benchmark_returns.mean()
            benchmark_volatility = benchmark_returns.std()

            if benchmark_volatility > 0:
                benchmark_sharpe = (
                    benchmark_avg_return - self.risk_free_rate / 12
                ) / benchmark_volatility
            else:
                benchmark_sharpe = 0.0

            return {
                "benchmark_total_return": benchmark_total_return,
                "benchmark_avg_return": benchmark_avg_return,
                "benchmark_volatility": benchmark_volatility,
                "benchmark_sharpe_ratio": benchmark_sharpe,
            }

        except Exception as e:
            logger.error(f"Error calculating benchmark performance: {str(e)}")
            return {
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

        # Calculate aggregate metrics
        aggregate_metrics = {
            "total_periods": len(df),
            "avg_total_return": df["total_return"].mean(),
            "avg_sharpe_ratio": df["sharpe_ratio"].mean(),
            "avg_calmar_ratio": df["calmar_ratio"].mean(),
            "worst_max_drawdown": df["max_drawdown"].min(),
            "avg_volatility": df["volatility"].mean(),
            "hit_ratio": (df["total_return"] > 0).mean(),
            "benchmark_avg_return": df["benchmark_total_return"].mean(),
            "benchmark_avg_sharpe": df["benchmark_sharpe_ratio"].mean(),
            "excess_return": df["total_return"].mean()
            - df["benchmark_total_return"].mean(),
            "excess_sharpe": df["sharpe_ratio"].mean()
            - df["benchmark_sharpe_ratio"].mean(),
            # Add transaction cost metrics
            "total_transaction_costs": df["transaction_cost"].sum(),
            "avg_transaction_cost": df["transaction_cost"].mean(),
            "total_turnover": df["turnover"].sum(),
            "avg_turnover": df["turnover"].mean(),
            "net_total_return": df["net_total_return"].mean(),
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
                    downside_deviation = np.std(downside_returns)
                    if downside_deviation > 0:
                        # Use annualized risk-free rate
                        risk_free_rate_monthly = self.risk_free_rate / 12
                        aggregate_sortino = (
                            avg_return - risk_free_rate_monthly
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

        return aggregate_metrics

    def _get_transaction_cost(self, asset: str) -> float:
        """Return transaction cost (as a decimal) for a given asset based on type."""
        # Default mapping (can be extended)
        if asset in ["SPY", "TLT", "GLD", "EFA", "IWM", "XLE"]:
            return 0.0005  # ETF
        elif asset in ["AAPL", "MSFT", "JPM", "UNH", "WMT", "BA"]:
            return 0.001  # Large Cap
        else:
            return 0.002  # Default for others
