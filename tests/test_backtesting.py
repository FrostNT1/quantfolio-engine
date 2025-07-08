"""
Tests for backtesting framework.

Tests data validation, walk-forward backtesting, and performance metrics calculation.
"""

import numpy as np
import pandas as pd
import pytest

from quantfolio_engine.backtesting import DataValidator, WalkForwardBacktester


class TestDataValidator:
    """Test data validation functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create sample returns data
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="ME")
        np.random.seed(42)

        self.returns_df = pd.DataFrame(
            np.random.normal(0.01, 0.05, (len(dates), 5)),
            index=dates,
            columns=["SPY", "TLT", "GLD", "AAPL", "MSFT"],
        )

        # Create sample factor exposures in long format
        assets = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
        factor_data = []
        for date in dates:
            for asset in assets:
                factor_data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "MKT": np.random.normal(0, 1),
                        "SMB": np.random.normal(0, 1),
                        "HML": np.random.normal(0, 1),
                    }
                )
        self.factor_exposures = pd.DataFrame(factor_data)

        # Create sample factor regimes in long format
        regime_data = []
        for date in dates:
            for asset in assets:
                regime_data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "regime_0_prob": np.random.uniform(0, 1),
                        "regime_1_prob": np.random.uniform(0, 1),
                        "regime_2_prob": np.random.uniform(0, 1),
                        "regime": np.random.randint(0, 3),
                    }
                )
        self.factor_regimes = pd.DataFrame(regime_data)

        # Create sample sentiment data
        self.sentiment_scores = pd.DataFrame(
            np.random.uniform(-1, 1, (len(dates), 2)),
            index=dates,
            columns=["SPY_sentiment", "TLT_sentiment"],
        )

        # Create sample macro data
        self.macro_data = pd.DataFrame(
            np.random.normal(0, 0.1, (len(dates), 2)),
            index=dates,
            columns=["GDP_growth", "Inflation"],
        )

    def test_validate_returns_data_valid(self):
        """Test validation of valid returns data."""
        validator = DataValidator(max_gap_days=35)  # More lenient for test data
        is_valid, message = validator._validate_returns_data(self.returns_df)

        assert is_valid
        assert "PASSED" in message
        assert "5 assets" in message

    def test_validate_returns_data_empty(self):
        """Test validation of empty returns data."""
        validator = DataValidator()
        empty_df = pd.DataFrame()
        is_valid, message = validator._validate_returns_data(empty_df)

        assert not is_valid
        assert "FAILED" in message
        assert "empty" in message

    def test_validate_returns_data_insufficient_assets(self):
        """Test validation with insufficient assets."""
        validator = DataValidator()
        single_asset_df = self.returns_df.iloc[:, :1]
        is_valid, message = validator._validate_returns_data(single_asset_df)

        assert not is_valid
        assert "FAILED" in message
        assert "at least 2 assets" in message

    def test_validate_returns_data_extreme_values(self):
        """Test validation with extreme return values."""
        validator = DataValidator()

        # Test values > 100%
        extreme_df = self.returns_df.copy()
        extreme_df.iloc[0, 0] = 1.5  # 150% return
        is_valid, message = validator._validate_returns_data(extreme_df)

        assert not is_valid
        assert "FAILED" in message
        assert "> 100%" in message

    def test_validate_factor_data_exposures(self):
        """Test validation of factor exposures data."""
        validator = DataValidator()
        is_valid, message = validator._validate_factor_data(
            self.factor_exposures, "exposures"
        )

        assert is_valid
        assert "PASSED" in message

    def test_validate_factor_data_regimes(self):
        """Test validation of factor regimes data."""
        validator = DataValidator()
        is_valid, message = validator._validate_factor_data(
            self.factor_regimes, "regimes"
        )

        assert is_valid
        assert "PASSED" in message

    def test_validate_sentiment_data(self):
        """Test validation of sentiment data."""
        validator = DataValidator()
        is_valid, message = validator._validate_sentiment_data(self.sentiment_scores)

        assert is_valid
        assert "PASSED" in message

    def test_validate_macro_data(self):
        """Test validation of macro data."""
        validator = DataValidator()
        is_valid, message = validator._validate_macro_data(self.macro_data)

        assert is_valid
        assert "PASSED" in message

    def test_validate_data_alignment(self):
        """Test validation of data alignment."""
        # Use a validator with a lower min_training_years and min_testing_years for test data
        validator = DataValidator(
            min_training_years=2, min_testing_years=1, max_gap_days=35
        )
        is_valid, message = validator._validate_data_alignment(
            self.returns_df,
            self.factor_exposures,
            self.factor_regimes,
            self.sentiment_scores,
            self.macro_data,
        )

        print(f"Data alignment validation result: {is_valid}, message: {message}")
        assert is_valid
        assert "PASSED" in message

    def test_validate_sufficient_history(self):
        """Test validation of sufficient history."""
        validator = DataValidator(min_training_years=2, min_testing_years=1)
        is_valid, message = validator._validate_sufficient_history(self.returns_df)

        assert is_valid
        assert "PASSED" in message

    def test_validate_sufficient_history_insufficient(self):
        """Test validation with insufficient history."""
        # Create short dataset
        short_dates = pd.date_range(start="2020-01-01", end="2021-06-30", freq="ME")
        short_df = pd.DataFrame(
            np.random.normal(0.01, 0.05, (len(short_dates), 3)),
            index=short_dates,
            columns=["SPY", "TLT", "GLD"],
        )

        validator = DataValidator(min_training_years=8, min_testing_years=2)
        is_valid, message = validator._validate_sufficient_history(short_df)

        assert not is_valid
        assert "FAILED" in message

    def test_suggest_train_test_split(self):
        """Test train/test split suggestion."""
        validator = DataValidator(min_training_years=2, min_testing_years=1)
        train_end, test_start = validator.suggest_train_test_split(self.returns_df)

        assert train_end < test_start
        assert train_end <= self.returns_df.index.max()
        assert test_start <= self.returns_df.index.max()

    def test_validate_data_for_backtesting_complete(self):
        """Test complete data validation for backtesting."""
        validator = DataValidator(
            min_training_years=2, min_testing_years=1, max_gap_days=35
        )
        is_valid, messages = validator.validate_data_for_backtesting(
            self.returns_df,
            self.factor_exposures,
            self.factor_regimes,
            self.sentiment_scores,
            self.macro_data,
        )

        assert is_valid
        assert len(messages) > 0
        assert all("PASSED" in msg for msg in messages.values())


class TestWalkForwardBacktester:
    """Test walk-forward backtesting functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create sample returns data with longer history
        dates = pd.date_range(start="2010-01-01", end="2023-12-31", freq="ME")
        np.random.seed(42)

        self.returns_df = pd.DataFrame(
            np.random.normal(0.01, 0.05, (len(dates), 5)),
            index=dates,
            columns=["SPY", "TLT", "GLD", "AAPL", "MSFT"],
        )

        # Create sample factor exposures in long format
        assets = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
        factor_data = []
        for date in dates:
            for asset in assets:
                factor_data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "MKT": np.random.normal(0, 1),
                        "SMB": np.random.normal(0, 1),
                        "HML": np.random.normal(0, 1),
                    }
                )
        self.factor_exposures = pd.DataFrame(factor_data)

        # Create sample factor regimes in long format
        regime_data = []
        for date in dates:
            for asset in assets:
                regime_data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "regime_0_prob": np.random.uniform(0, 1),
                        "regime_1_prob": np.random.uniform(0, 1),
                        "regime_2_prob": np.random.uniform(0, 1),
                        "regime": np.random.randint(0, 3),
                    }
                )
        self.factor_regimes = pd.DataFrame(regime_data)

        # Create sample sentiment data
        self.sentiment_scores = pd.DataFrame(
            np.random.uniform(-1, 1, (len(dates), 2)),
            index=dates,
            columns=["SPY_sentiment", "TLT_sentiment"],
        )

        # Create sample macro data
        self.macro_data = pd.DataFrame(
            np.random.normal(0, 0.1, (len(dates), 2)),
            index=dates,
            columns=["GDP_growth", "Inflation"],
        )

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        backtester = WalkForwardBacktester(
            train_years=5,
            test_years=1,
            rebalance_frequency="monthly",
            risk_free_rate=0.03,
            max_weight=0.3,
            min_weight=0.05,
            max_volatility=0.15,
            random_state=42,
        )

        assert backtester.train_years == 5
        assert backtester.test_years == 1
        assert backtester.rebalance_frequency == "monthly"
        assert backtester.risk_free_rate == 0.03
        assert backtester.max_weight == 0.3
        assert backtester.min_weight == 0.05
        assert backtester.max_volatility == 0.15
        assert backtester.random_state == 42

    def test_get_rebalance_dates_monthly(self):
        """Test monthly rebalance date generation."""
        backtester = WalkForwardBacktester(train_years=5, test_years=1)
        rebalance_dates = backtester._get_rebalance_dates(self.returns_df)

        assert len(rebalance_dates) > 0
        assert all(isinstance(date, pd.Timestamp) for date in rebalance_dates)

        # Check that dates are properly spaced
        for i in range(1, len(rebalance_dates)):
            diff = rebalance_dates[i] - rebalance_dates[i - 1]
            assert diff.days >= 28  # At least 28 days between rebalances

    def test_get_rebalance_dates_quarterly(self):
        """Test quarterly rebalance date generation."""
        backtester = WalkForwardBacktester(
            train_years=5, test_years=1, rebalance_frequency="quarterly"
        )
        rebalance_dates = backtester._get_rebalance_dates(self.returns_df)

        assert len(rebalance_dates) > 0
        # Check quarterly spacing (approximately 90 days)
        for i in range(1, len(rebalance_dates)):
            diff = rebalance_dates[i] - rebalance_dates[i - 1]
            assert diff.days >= 80  # At least 80 days between quarterly rebalances

    def test_get_next_rebalance_date(self):
        """Test next rebalance date calculation."""
        backtester = WalkForwardBacktester(rebalance_frequency="monthly")
        current_date = pd.Timestamp("2022-01-01")
        next_date = backtester._get_next_rebalance_date(self.returns_df, current_date)

        assert next_date > current_date
        assert next_date <= self.returns_df.index.max()

    def test_get_training_data(self):
        """Test training data extraction."""
        backtester = WalkForwardBacktester()
        train_start = pd.Timestamp("2020-01-01")
        train_end = pd.Timestamp("2022-01-01")

        train_data = backtester._get_training_data(
            self.returns_df,
            self.factor_exposures,
            self.factor_regimes,
            None,
            None,
            train_start,
            train_end,
        )

        assert "returns" in train_data
        assert "factor_exposures" in train_data
        assert "factor_regimes" in train_data
        assert "sentiment_scores" not in train_data
        assert "macro_data" not in train_data

        # Check data ranges
        assert train_data["returns"].index.min() >= train_start
        assert train_data["returns"].index.max() <= train_end

    def test_get_testing_data(self):
        """Test testing data extraction."""
        backtester = WalkForwardBacktester()
        test_start = pd.Timestamp("2022-01-01")
        test_end = pd.Timestamp("2022-02-01")

        test_data = backtester._get_testing_data(self.returns_df, test_start, test_end)

        assert not test_data.empty
        assert test_data.index.min() >= test_start
        assert test_data.index.max() <= test_end

    def test_calculate_test_performance(self):
        """Test performance calculation for test period."""
        backtester = WalkForwardBacktester()

        # Create test data
        test_dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="ME")
        test_data = pd.DataFrame(
            np.random.normal(0.01, 0.05, (len(test_dates), 3)),
            index=test_dates,
            columns=["SPY", "TLT", "GLD"],
        )

        # Create weights
        weights = np.array([0.4, 0.3, 0.3])
        rebalance_date = pd.Timestamp("2022-01-01")

        # Calculate portfolio returns first
        portfolio_returns = (test_data * weights).sum(axis=1)

        performance = backtester._calculate_test_performance(
            weights, test_data, rebalance_date, portfolio_returns
        )

        assert "total_return" in performance
        assert "avg_return" in performance
        assert "volatility" in performance
        assert "sharpe_ratio" in performance
        assert "sortino_ratio" in performance
        assert "max_drawdown" in performance
        assert "calmar_ratio" in performance

        # Check that metrics are reasonable
        assert isinstance(performance["total_return"], float)
        assert isinstance(performance["volatility"], float)
        assert performance["volatility"] >= 0

    def test_calculate_benchmark_performance(self):
        """Test benchmark performance calculation."""
        backtester = WalkForwardBacktester()

        # Create test data with SPY and TLT
        test_dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="ME")
        test_data = pd.DataFrame(
            np.random.normal(0.01, 0.05, (len(test_dates), 3)),
            index=test_dates,
            columns=["SPY", "TLT", "GLD"],
        )

        rebalance_date = pd.Timestamp("2022-01-01")
        benchmark_performance = backtester._calculate_benchmark_performance(
            test_data, rebalance_date
        )

        assert "benchmark_total_return" in benchmark_performance
        assert "benchmark_avg_return" in benchmark_performance
        assert "benchmark_volatility" in benchmark_performance
        assert "benchmark_sharpe_ratio" in benchmark_performance

        # Check that metrics are reasonable
        assert isinstance(benchmark_performance["benchmark_total_return"], float)
        assert isinstance(benchmark_performance["benchmark_volatility"], float)
        assert benchmark_performance["benchmark_volatility"] >= 0

    def test_calculate_aggregate_metrics(self):
        """Test aggregate metrics calculation."""
        backtester = WalkForwardBacktester()

        # Create sample performance history
        backtester.performance_history = [
            {
                "date": pd.Timestamp("2022-01-01"),
                "total_return": 0.05,
                "avg_return": 0.01,
                "volatility": 0.15,
                "sharpe_ratio": 0.5,
                "sortino_ratio": 0.6,
                "max_drawdown": -0.1,
                "calmar_ratio": 0.5,
                "benchmark_total_return": 0.03,
                "benchmark_sharpe_ratio": 0.3,
                "transaction_cost": 0.001,
                "turnover": 0.15,
                "net_total_return": 0.049,  # total_return - transaction_cost
            },
            {
                "date": pd.Timestamp("2022-02-01"),
                "total_return": 0.03,
                "avg_return": 0.008,
                "volatility": 0.12,
                "sharpe_ratio": 0.4,
                "sortino_ratio": 0.5,
                "max_drawdown": -0.08,
                "calmar_ratio": 0.375,
                "benchmark_total_return": 0.02,
                "benchmark_sharpe_ratio": 0.25,
                "transaction_cost": 0.002,
                "turnover": 0.12,
                "net_total_return": 0.028,  # total_return - transaction_cost
            },
        ]

        aggregate_metrics = backtester._calculate_aggregate_metrics()

        assert "total_periods" in aggregate_metrics
        assert "avg_total_return" in aggregate_metrics
        assert "avg_sharpe_ratio" in aggregate_metrics
        assert "avg_sortino_ratio" in aggregate_metrics
        assert "avg_calmar_ratio" in aggregate_metrics
        assert "worst_max_drawdown" in aggregate_metrics
        assert "avg_volatility" in aggregate_metrics
        assert "hit_ratio" in aggregate_metrics
        assert "excess_return" in aggregate_metrics
        assert "excess_sharpe" in aggregate_metrics

        # Check calculated values
        assert aggregate_metrics["total_periods"] == 2
        assert aggregate_metrics["avg_total_return"] == 0.04  # (0.05 + 0.03) / 2
        assert aggregate_metrics["hit_ratio"] == 1.0  # Both periods positive

    def test_run_backtest_basic(self):
        """Test basic backtest execution."""
        backtester = WalkForwardBacktester(
            train_years=3, test_years=1, rebalance_frequency="quarterly"
        )

        # Override the data validator to be more lenient for tests
        backtester.data_validator = DataValidator(max_gap_days=35)

        # Run backtest with minimal data
        results = backtester.run_backtest(
            returns_df=self.returns_df,
            factor_exposures=self.factor_exposures,
            factor_regimes=self.factor_regimes,
            method="combined",
        )

        # Check that results are returned
        assert "performance_history" in results
        assert "weight_history" in results
        assert "aggregate_metrics" in results
        assert "validation_messages" in results

        # Check that performance history is a DataFrame
        assert isinstance(results["performance_history"], pd.DataFrame)

        # Check that we have some results
        if len(results["performance_history"]) > 0:
            assert "date" in results["performance_history"].columns
            assert "total_return" in results["performance_history"].columns
            assert "sharpe_ratio" in results["performance_history"].columns

    def test_run_backtest_invalid_data(self):
        """Test backtest with invalid data."""
        backtester = WalkForwardBacktester()

        # Create invalid data (empty DataFrame)
        empty_df = pd.DataFrame()

        results = backtester.run_backtest(returns_df=empty_df, method="combined")

        assert "error" in results
        assert "Data validation failed" in results["error"]

    def test_transaction_costs_default(self):
        """Test default transaction costs."""
        backtester = WalkForwardBacktester()

        expected_costs = {
            "ETFs": 0.0005,
            "Large_Cap": 0.001,
            "Small_Cap": 0.002,
            "International": 0.0025,
            "Commodities": 0.0015,
        }

        assert backtester.transaction_costs == expected_costs

    def test_transaction_costs_custom(self):
        """Test custom transaction costs."""
        custom_costs = {"ETFs": 0.001, "Stocks": 0.002}
        backtester = WalkForwardBacktester(transaction_costs=custom_costs)

        assert backtester.transaction_costs == custom_costs

    def test_transaction_cost_modeling(self):
        """Test transaction cost calculation and application."""
        # Create test data
        test_returns_df = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.03],
                "TLT": [0.005, -0.01, 0.02, 0.01],
                "AAPL": [0.02, 0.01, -0.02, 0.04],
            },
            index=pd.date_range("2020-01-01", periods=4, freq="ME"),
        )

        # Test transaction cost calculation
        backtester = WalkForwardBacktester(
            train_years=1,
            test_years=1,
            transaction_costs={
                "ETFs": 0.0005,  # 5 bps
                "Large_Cap": 0.001,  # 10 bps
                "Small_Cap": 0.002,  # 20 bps
            },
        )

        # Test transaction cost lookup
        assert backtester._get_transaction_cost("SPY") == 0.0005  # ETF
        assert backtester._get_transaction_cost("TLT") == 0.0005  # ETF
        assert backtester._get_transaction_cost("AAPL") == 0.001  # Large Cap
        assert backtester._get_transaction_cost("UNKNOWN") == 0.002  # Default

        # Test transaction cost application in performance calculation
        weights = pd.Series({"SPY": 0.4, "TLT": 0.3, "AAPL": 0.3})
        prev_weights = pd.Series({"SPY": 0.5, "TLT": 0.3, "AAPL": 0.2})

        # Calculate turnover
        turnover = (weights - prev_weights).abs().sum()
        assert abs(turnover - 0.2) < 1e-10  # Handle floating point precision

        # Calculate transaction cost
        tc = 0.0
        for asset, weight_change in (weights - prev_weights).abs().items():
            tc += backtester._get_transaction_cost(asset) * weight_change

        expected_tc = 0.1 * 0.0005 + 0.0 * 0.0005 + 0.1 * 0.001
        assert abs(tc - expected_tc) < 1e-6

    def test_benchmark_performance_calculation(self):
        """Test benchmark performance calculation (60/40 vs equal-weight)."""
        # Create test data with SPY and TLT
        returns_df = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.03],
                "TLT": [0.005, -0.01, 0.02, 0.01],
                "AAPL": [0.02, 0.01, -0.02, 0.04],
            },
            index=pd.date_range("2020-01-01", periods=4, freq="ME"),
        )

        backtester = WalkForwardBacktester(train_years=1, test_years=1)

        # Test benchmark calculation with SPY/TLT available
        benchmark_perf = backtester._calculate_benchmark_performance(
            returns_df, pd.Timestamp("2020-01-01")
        )

        assert "benchmark_total_return" in benchmark_perf
        assert "benchmark_avg_return" in benchmark_perf
        assert "benchmark_volatility" in benchmark_perf
        assert "benchmark_sharpe_ratio" in benchmark_perf

        # Test benchmark calculation without SPY/TLT (should use equal-weight)
        returns_df_no_spy = returns_df.drop(columns=["SPY", "TLT"])
        benchmark_perf_equal = backtester._calculate_benchmark_performance(
            returns_df_no_spy, pd.Timestamp("2020-01-01")
        )

        assert "benchmark_total_return" in benchmark_perf_equal
        assert "benchmark_avg_return" in benchmark_perf_equal

    def test_walk_forward_edge_cases(self):
        """Test walk-forward backtesting with edge cases."""
        # Create minimal test data
        returns_df = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.03, 0.01, 0.02],
                "TLT": [0.005, -0.01, 0.02, 0.01, 0.005, -0.01],
            },
            index=pd.date_range("2020-01-01", periods=6, freq="ME"),
        )

        # Test with insufficient data (should handle gracefully)
        backtester = WalkForwardBacktester(train_years=5, test_years=2)

        # This should not raise an error but return empty results
        result = backtester.run_backtest(returns_df)

        # Should have validation error due to insufficient data
        assert "error" in result or len(result.get("performance_history", [])) == 0

        # Test with valid data but short history
        backtester_short = WalkForwardBacktester(train_years=1, test_years=1)
        result_short = backtester_short.run_backtest(returns_df)

        # Should have some results but limited periods
        if "performance_history" in result_short:
            assert len(result_short["performance_history"]) <= 2  # Limited periods

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation in backtest context."""
        # Create test data with known returns
        returns_df = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.03, 0.01, 0.02],
                "TLT": [0.005, -0.01, 0.02, 0.01, 0.005, -0.01],
            },
            index=pd.date_range("2020-01-01", periods=6, freq="ME"),
        )

        backtester = WalkForwardBacktester(train_years=1, test_years=1)

        # Test performance calculation with known weights
        weights = np.array([0.6, 0.4])  # 60/40 portfolio
        test_data = returns_df.iloc[2:4]  # 2 periods of test data

        # Calculate portfolio returns first
        portfolio_returns = (test_data * weights).sum(axis=1)

        performance = backtester._calculate_test_performance(
            weights, test_data, pd.Timestamp("2020-03-01"), portfolio_returns
        )

        # Check that all expected metrics are present
        expected_metrics = [
            "total_return",
            "avg_return",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
        ]

        for metric in expected_metrics:
            assert metric in performance
            assert isinstance(performance[metric], (int, float))

        # Test with zero volatility (edge case)
        zero_vol_returns = pd.DataFrame(
            {"SPY": [0.01, 0.01, 0.01], "TLT": [0.005, 0.005, 0.005]},
            index=pd.date_range("2020-01-01", periods=3, freq="ME"),
        )

        # Calculate portfolio returns for zero volatility case
        zero_vol_portfolio_returns = (zero_vol_returns * weights).sum(axis=1)

        zero_vol_performance = backtester._calculate_test_performance(
            weights,
            zero_vol_returns,
            pd.Timestamp("2020-01-01"),
            zero_vol_portfolio_returns,
        )

        assert zero_vol_performance["volatility"] == 0.0
        assert zero_vol_performance["sharpe_ratio"] == 0.0

    def test_aggregate_metrics_calculation(self):
        """Test aggregate metrics calculation across multiple periods."""
        backtester = WalkForwardBacktester(train_years=1, test_years=1)

        # Mock performance history
        backtester.performance_history = [
            {
                "total_return": 0.05,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.5,
                "calmar_ratio": 2.0,
                "max_drawdown": -0.025,
                "volatility": 0.15,
                "benchmark_total_return": 0.03,
                "benchmark_sharpe_ratio": 0.8,
                "transaction_cost": 0.001,
                "turnover": 0.15,
                "net_total_return": 0.049,  # total_return - transaction_cost
            },
            {
                "total_return": 0.03,
                "sharpe_ratio": 0.8,
                "sortino_ratio": 1.0,
                "calmar_ratio": 1.5,
                "max_drawdown": -0.02,
                "volatility": 0.12,
                "benchmark_total_return": 0.02,
                "benchmark_sharpe_ratio": 0.6,
                "transaction_cost": 0.002,
                "turnover": 0.12,
                "net_total_return": 0.028,  # total_return - transaction_cost
            },
        ]

        aggregate_metrics = backtester._calculate_aggregate_metrics()

        # Check that all expected aggregate metrics are present
        expected_metrics = [
            "total_periods",
            "avg_total_return",
            "avg_sharpe_ratio",
            "avg_sortino_ratio",
            "avg_calmar_ratio",
            "worst_max_drawdown",
            "avg_volatility",
            "hit_ratio",
            "benchmark_avg_return",
            "benchmark_avg_sharpe",
            "excess_return",
            "excess_sharpe",
        ]

        for metric in expected_metrics:
            assert metric in aggregate_metrics
            assert isinstance(aggregate_metrics[metric], (int, float))

        # Check specific calculations
        assert aggregate_metrics["total_periods"] == 2
        assert (
            abs(aggregate_metrics["avg_total_return"] - 0.04) < 1e-10
        )  # (0.05 + 0.03) / 2
        assert aggregate_metrics["hit_ratio"] == 1.0  # Both periods positive
        assert abs(aggregate_metrics["excess_return"] - 0.015) < 1e-10  # 0.04 - 0.025

    def test_rebalance_date_generation(self):
        """Test rebalance date generation for different frequencies."""
        returns_df = pd.DataFrame(
            {
                "SPY": [0.01] * 24,  # 2 years of monthly data
                "TLT": [0.005] * 24,
            },
            index=pd.date_range("2020-01-01", periods=24, freq="ME"),
        )

        # Test monthly rebalancing
        backtester_monthly = WalkForwardBacktester(
            train_years=1, test_years=1, rebalance_frequency="monthly"
        )
        monthly_dates = backtester_monthly._get_rebalance_dates(returns_df)
        assert len(monthly_dates) > 0
        assert all(isinstance(d, pd.Timestamp) for d in monthly_dates)

        # Test quarterly rebalancing
        backtester_quarterly = WalkForwardBacktester(
            train_years=1, test_years=1, rebalance_frequency="quarterly"
        )
        quarterly_dates = backtester_quarterly._get_rebalance_dates(returns_df)
        assert len(quarterly_dates) > 0
        assert len(quarterly_dates) < len(monthly_dates)  # Fewer quarterly dates

        # Test annual rebalancing
        backtester_annual = WalkForwardBacktester(
            train_years=1, test_years=1, rebalance_frequency="annual"
        )
        annual_dates = backtester_annual._get_rebalance_dates(returns_df)
        assert len(annual_dates) > 0
        assert len(annual_dates) < len(quarterly_dates)  # Fewer annual dates

    def test_next_rebalance_date_calculation(self):
        """Test next rebalance date calculation."""
        backtester = WalkForwardBacktester(rebalance_frequency="monthly")

        current_date = pd.Timestamp("2020-01-31")
        returns_df = pd.DataFrame(
            {"SPY": [0.01] * 12},
            index=pd.date_range("2020-01-01", periods=12, freq="ME"),
        )

        next_date = backtester._get_next_rebalance_date(returns_df, current_date)
        expected_next = current_date + pd.DateOffset(months=1)
        assert next_date == expected_next

        # Test with annual frequency
        backtester_annual = WalkForwardBacktester(rebalance_frequency="annual")
        next_date_annual = backtester_annual._get_next_rebalance_date(
            returns_df, current_date
        )
        expected_next_annual = pd.Timestamp("2020-12-31")  # YE logic
        assert next_date_annual == expected_next_annual

    def test_training_data_filtering(self):
        """Test training data filtering for different data formats."""
        returns_df = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.03],
                "TLT": [0.005, -0.01, 0.02, 0.01],
            },
            index=pd.date_range("2020-01-01", periods=4, freq="ME"),
        )

        # Create factor data in long format
        factor_data = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"] * 2,
                "asset": ["SPY"] * 4 + ["TLT"] * 4,
                "factor1": [0.1, 0.2, -0.1, 0.3] * 2,
                "factor2": [0.05, -0.1, 0.2, 0.1] * 2,
            }
        )

        backtester = WalkForwardBacktester(train_years=1, test_years=1)

        train_start = pd.Timestamp("2020-01-01")
        train_end = pd.Timestamp("2020-02-29")

        train_data = backtester._get_training_data(
            returns_df, factor_data, None, None, None, train_start, train_end
        )

        assert "returns" in train_data
        assert "factor_exposures" in train_data
        assert len(train_data["returns"]) == 2  # Jan and Feb
        assert len(train_data["factor_exposures"]) == 4  # 2 assets * 2 periods


if __name__ == "__main__":
    pytest.main([__file__])
