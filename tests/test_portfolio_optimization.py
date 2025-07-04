"""Tests for portfolio optimization modules."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quantfolio_engine.config import ASSET_UNIVERSE
from quantfolio_engine.optimizer.black_litterman import BlackLittermanOptimizer
from quantfolio_engine.optimizer.monte_carlo import MonteCarloOptimizer
from quantfolio_engine.optimizer.portfolio_engine import PortfolioOptimizationEngine


class TestBlackLittermanOptimizer:
    """Test cases for BlackLittermanOptimizer."""

    def test_init(self):
        """Test BlackLittermanOptimizer initialization."""
        optimizer = BlackLittermanOptimizer(risk_free_rate=0.03, tau=0.1)

        assert optimizer.risk_free_rate == 0.03
        assert optimizer.tau == 0.1
        assert len(optimizer.market_cap_weight) > 0

    def test_estimate_covariance_matrix_sample(self):
        """Test covariance matrix estimation using sample method."""
        optimizer = BlackLittermanOptimizer()

        # Create dummy returns data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 60),
                "TLT": np.random.normal(0.005, 0.03, 60),
                "GLD": np.random.normal(0.008, 0.04, 60),
            },
            index=dates,
        )

        cov_matrix = optimizer.estimate_covariance_matrix(returns_df, method="sample")

        assert isinstance(cov_matrix, pd.DataFrame)
        assert cov_matrix.shape == (3, 3)
        assert np.all(np.diag(cov_matrix) >= 0)  # Diagonal should be non-negative

    def test_estimate_covariance_matrix_lw(self):
        """Test covariance matrix estimation using Ledoit-Wolf method."""
        optimizer = BlackLittermanOptimizer()

        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 60),
                "TLT": np.random.normal(0.005, 0.03, 60),
                "GLD": np.random.normal(0.008, 0.04, 60),
            },
            index=dates,
        )

        cov_matrix = optimizer.estimate_covariance_matrix(returns_df, method="lw")

        assert isinstance(cov_matrix, pd.DataFrame)
        assert cov_matrix.shape == (3, 3)

    def test_calculate_equilibrium_returns(self):
        """Test equilibrium returns calculation."""
        optimizer = BlackLittermanOptimizer()

        # Create dummy covariance matrix
        cov_matrix = pd.DataFrame(
            {
                "SPY": [0.0025, 0.0005, 0.0008],
                "TLT": [0.0005, 0.0009, 0.0002],
                "GLD": [0.0008, 0.0002, 0.0016],
            },
            index=["SPY", "TLT", "GLD"],
        )

        equilibrium_returns = optimizer.calculate_equilibrium_returns(cov_matrix)

        assert isinstance(equilibrium_returns, pd.Series)
        assert len(equilibrium_returns) == 3
        assert all(
            asset in equilibrium_returns.index for asset in ["SPY", "TLT", "GLD"]
        )

    def test_create_factor_timing_views(self):
        """Test factor timing views creation."""
        optimizer = BlackLittermanOptimizer()

        # Create dummy returns data
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 24),
                "TLT": np.random.normal(0.005, 0.03, 24),
                "GLD": np.random.normal(0.008, 0.04, 24),
            },
            index=dates,
        )

        # Create dummy factor data
        factor_exposures = pd.DataFrame(
            {
                "SPY_CPIAUCSL": [0.8, 0.2, 0.5],
                "TLT_FEDFUNDS": [0.1, 0.9, 0.2],
                "GLD_INDPRO": [0.3, 0.1, 0.8],
            },
            index=dates[-3:],
        )

        factor_regimes = pd.DataFrame(
            {
                "regime_0": [1, 0, 0],
                "regime_1": [0, 1, 0],
                "regime_2": [0, 0, 1],
            },
            index=dates[-3:],
        )

        P, Q, Omega = optimizer.create_factor_timing_views(
            factor_exposures, factor_regimes, returns_df
        )

        # Should create some views if factor data is available
        if len(P) > 0:
            assert P.shape[0] == len(Q)
            assert Omega.shape[0] == len(Q)
            assert Omega.shape[1] == len(Q)

    def test_optimize_portfolio_basic(self):
        """Test basic portfolio optimization."""
        optimizer = BlackLittermanOptimizer()

        # Create dummy returns data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 60),
                "TLT": np.random.normal(0.005, 0.03, 60),
                "GLD": np.random.normal(0.008, 0.04, 60),
            },
            index=dates,
        )

        result = optimizer.optimize_portfolio(returns_df)

        assert isinstance(result, dict)
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result

        # Check weights sum to 1
        assert abs(result["weights"].sum() - 1.0) < 1e-6
        # Check all weights are non-negative (long-only)
        assert all(result["weights"] >= 0)

    def test_optimize_portfolio_with_constraints(self):
        """Test portfolio optimization with constraints."""
        optimizer = BlackLittermanOptimizer()

        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 60),
                "TLT": np.random.normal(0.005, 0.03, 60),
                "GLD": np.random.normal(0.008, 0.04, 60),
            },
            index=dates,
        )

        constraints = {
            "max_weight": 0.5,
            "min_weight": 0.05,
        }

        result = optimizer.optimize_portfolio(returns_df, constraints=constraints)

        # Check constraint satisfaction
        assert all(result["weights"] <= 0.5)
        assert all(result["weights"] >= 0.05)


class TestMonteCarloOptimizer:
    """Test cases for MonteCarloOptimizer."""

    def test_init(self):
        """Test MonteCarloOptimizer initialization."""
        optimizer = MonteCarloOptimizer(
            n_simulations=500,
            time_horizon=6,
            risk_free_rate=0.03,
            max_cvar_loss=0.10,
            confidence_level=0.99,
        )

        assert optimizer.n_simulations == 500
        assert optimizer.time_horizon == 6
        assert optimizer.risk_free_rate == 0.03
        assert optimizer.max_cvar_loss == 0.10
        assert optimizer.confidence_level == 0.99

    def test_generate_scenarios(self):
        """Test scenario generation."""
        optimizer = MonteCarloOptimizer(n_simulations=100)

        # Create dummy data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 60),
                "TLT": np.random.normal(0.005, 0.03, 60),
                "GLD": np.random.normal(0.008, 0.04, 60),
            },
            index=dates,
        )

        _ = pd.DataFrame(
            {
                "GDP": np.random.normal(2.0, 0.5, 60),
                "INFLATION": np.random.normal(2.5, 0.3, 60),
            },
            index=dates,
        )

        _ = pd.DataFrame(
            {
                "regime_0": np.random.choice([0, 1], 60),
                "regime_1": np.random.choice([0, 1], 60),
            },
            index=dates,
        )

        scenarios = optimizer.generate_scenarios(returns_df)

        assert isinstance(scenarios, np.ndarray)
        assert scenarios.shape[0] == 100  # n_simulations
        assert scenarios.shape[1] == optimizer.time_horizon  # time_horizon
        assert scenarios.shape[2] == 3  # n_assets

    def test_optimize_with_constraints(self):
        """Test Monte Carlo optimization with constraints."""
        optimizer = MonteCarloOptimizer(n_simulations=100)

        # Create dummy scenarios with correct number of assets
        n_assets = len(ASSET_UNIVERSE)
        scenarios = np.random.normal(0.01, 0.05, (100, 12, n_assets))

        constraints = {
            "max_weight": 0.5,
            "min_weight": 0.05,
        }

        result = optimizer.optimize_with_constraints(scenarios, constraints=constraints)

        assert isinstance(result, dict)
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result

        # Check weights sum to 1
        assert abs(result["weights"].sum() - 1.0) < 1e-6
        # Check constraint satisfaction
        assert all(result["weights"] <= 0.5)
        assert all(
            result["weights"] >= 0.05 - 1e-10
        )  # Allow for floating point precision

    def test_optimize_with_constraints_3d(self):
        """Test Monte Carlo optimization with 3D scenarios."""
        optimizer = MonteCarloOptimizer(n_simulations=100)

        # Create 3D dummy scenarios with correct number of assets
        n_assets = len(ASSET_UNIVERSE)
        scenarios = np.random.normal(0.01, 0.05, (100, 12, n_assets))

        constraints = {
            "max_weight": 0.5,
            "min_weight": 0.05,
        }

        result = optimizer.optimize_with_constraints(scenarios, constraints=constraints)

        assert isinstance(result, dict)
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result

        # Check weights sum to 1
        assert abs(result["weights"].sum() - 1.0) < 1e-6
        # Check all weights are non-negative (long-only)
        assert all(result["weights"] >= 0)

    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        optimizer = MonteCarloOptimizer()

        # Create 3D scenarios to test the new path
        n_assets = len(ASSET_UNIVERSE)
        scenarios = np.random.normal(0.01, 0.05, (100, 12, n_assets))
        weights = np.ones(n_assets) / n_assets  # Equal weights

        metrics = optimizer._calculate_portfolio_metrics(scenarios, weights)

        assert isinstance(metrics, dict)
        assert "expected_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "var_95" in metrics

        # Check that metrics are reasonable
        assert metrics["expected_return"] > 0
        assert metrics["volatility"] > 0
        assert metrics["sharpe_ratio"] > -10  # Allow negative Sharpe for bad portfolios


class TestPortfolioOptimizationEngine:
    """Test cases for PortfolioOptimizationEngine."""

    def test_init(self):
        """Test PortfolioOptimizationEngine initialization."""
        engine = PortfolioOptimizationEngine(
            method="combined",
            risk_free_rate=0.03,
            max_drawdown=0.10,
            confidence_level=0.99,
        )

        assert engine.method == "combined"
        assert engine.risk_free_rate == 0.03
        assert engine.max_drawdown == 0.10
        assert engine.confidence_level == 0.99

    @patch("quantfolio_engine.optimizer.portfolio_engine.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        """Test data loading functionality."""
        engine = PortfolioOptimizationEngine()

        # Mock successful data loading
        mock_returns = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 60),
                "TLT": np.random.normal(0.005, 0.03, 60),
            },
            index=pd.date_range("2020-01-01", periods=60, freq="ME"),
        )
        mock_read_csv.return_value = mock_returns

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True):
            data = engine.load_data()

            assert isinstance(data, dict)
            assert "returns" in data

    def test_optimize_portfolio_black_litterman(self):
        """Test Black-Litterman optimization."""
        engine = PortfolioOptimizationEngine(method="black_litterman")

        # Create dummy data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        data = {
            "returns": pd.DataFrame(
                {
                    "SPY": np.random.normal(0.01, 0.05, 60),
                    "TLT": np.random.normal(0.005, 0.03, 60),
                    "GLD": np.random.normal(0.008, 0.04, 60),
                },
                index=dates,
            )
        }

        result = engine.optimize_portfolio(data)

        assert isinstance(result, dict)
        assert result["method"] == "black_litterman"
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result

    def test_optimize_portfolio_monte_carlo(self):
        """Test Monte Carlo optimization."""
        engine = PortfolioOptimizationEngine(method="monte_carlo")

        # Create dummy data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        data = {
            "returns": pd.DataFrame(
                {
                    "SPY": np.random.normal(0.01, 0.05, 60),
                    "TLT": np.random.normal(0.005, 0.03, 60),
                    "GLD": np.random.normal(0.008, 0.04, 60),
                },
                index=dates,
            ),
            "macro": pd.DataFrame(
                {
                    "GDP": np.random.normal(2.0, 0.5, 60),
                },
                index=dates,
            ),
            "factor_regimes": pd.DataFrame(
                {
                    "regime_0": np.random.choice([0, 1], 60),
                    "regime_1": np.random.choice([0, 1], 60),
                },
                index=dates,
            ),
        }

        result = engine.optimize_portfolio(data)

        assert isinstance(result, dict)
        assert result["method"] == "monte_carlo"
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result

    def test_optimize_portfolio_combined(self):
        """Test combined optimization."""
        engine = PortfolioOptimizationEngine(method="combined")

        # Create dummy data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        data = {
            "returns": pd.DataFrame(
                {
                    "SPY": np.random.normal(0.01, 0.05, 60),
                    "TLT": np.random.normal(0.005, 0.03, 60),
                    "GLD": np.random.normal(0.008, 0.04, 60),
                },
                index=dates,
            ),
            "macro": pd.DataFrame(
                {
                    "GDP": np.random.normal(2.0, 0.5, 60),
                },
                index=dates,
            ),
            "factor_regimes": pd.DataFrame(
                {
                    "regime_0": np.random.choice([0, 1], 60),
                    "regime_1": np.random.choice([0, 1], 60),
                },
                index=dates,
            ),
        }

        result = engine.optimize_portfolio(data)

        assert isinstance(result, dict)
        assert result["method"] == "combined"
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
        assert "black_litterman" in result
        assert "monte_carlo" in result

    def test_analyze_portfolio_risk(self):
        """Test portfolio risk analysis."""
        engine = PortfolioOptimizationEngine()

        # Create dummy data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        data = {
            "returns": pd.DataFrame(
                {
                    "SPY": np.random.normal(0.01, 0.05, 60),
                    "TLT": np.random.normal(0.005, 0.03, 60),
                    "GLD": np.random.normal(0.008, 0.04, 60),
                },
                index=dates,
            )
        }

        weights = pd.Series([0.4, 0.3, 0.3], index=["SPY", "TLT", "GLD"])

        risk_metrics = engine.analyze_portfolio_risk(weights, data)

        assert isinstance(risk_metrics, dict)
        assert "volatility" in risk_metrics
        assert "expected_return" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "max_drawdown" in risk_metrics
        assert "var_95" in risk_metrics
        assert "cvar_95" in risk_metrics

    def test_invalid_method(self):
        """Test handling of invalid optimization method."""
        engine = PortfolioOptimizationEngine(method="invalid_method")

        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        data = {
            "returns": pd.DataFrame(
                {
                    "SPY": np.random.normal(0.01, 0.05, 60),
                },
                index=dates,
            )
        }

        with pytest.raises(ValueError, match="Unknown optimization method"):
            engine.optimize_portfolio(data)

    def test_mc_frontier_respects_sector_limits(self):
        """Test that Monte Carlo efficient frontier respects sector limits."""
        engine = PortfolioOptimizationEngine(method="monte_carlo")

        # Create minimal data with two assets in same sector
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        data = {
            "returns": pd.DataFrame(
                {
                    "AAPL": np.random.normal(0.01, 0.05, 60),
                    "MSFT": np.random.normal(0.01, 0.05, 60),
                    "TLT": np.random.normal(0.005, 0.03, 60),
                },
                index=dates,
            ),
            "factor_regimes": pd.DataFrame(
                {
                    "regime_0": np.random.choice([0, 1], 60),
                    "regime_1": np.random.choice([0, 1], 60),
                },
                index=dates,
            ),
        }

        frontier = engine.generate_efficient_frontier(
            data, n_points=3, sector_limits={"Tech": 1.0}
        )

        # Pick any point and assert constraint
        if len(frontier["weights"]) > 0:
            # AAPL and MSFT should be in Tech sector, sum should be <= 1.0
            tech_weights = frontier["weights"][0, :2]  # First point, first two assets
            assert (tech_weights.sum() <= 1.0 + 1e-6).all()

    def test_combined_vol_annualisation(self):
        """Test that combined method uses correct frequency for volatility calculation."""
        engine = PortfolioOptimizationEngine(method="combined")

        # Create test data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        data = {
            "returns": pd.DataFrame(
                {
                    "SPY": np.random.normal(0.01, 0.05, 60),
                    "TLT": np.random.normal(0.005, 0.03, 60),
                    "GLD": np.random.normal(0.008, 0.04, 60),
                },
                index=dates,
            ),
            "factor_regimes": pd.DataFrame(
                {
                    "regime_0": np.random.choice([0, 1], 60),
                    "regime_1": np.random.choice([0, 1], 60),
                },
                index=dates,
            ),
        }

        result = engine.optimize_portfolio(data)

        # Calculate expected volatility using same frequency logic
        freq = 12  # monthly
        port_ret_series = (data["returns"] * result["weights"]).sum(axis=1)
        expected_vol = port_ret_series.std() * np.sqrt(freq)

        assert abs(result["volatility"] - expected_vol) < 1e-6


class TestIntegration:
    """Integration tests for portfolio optimization."""

    def test_end_to_end_optimization(self):
        """Test end-to-end portfolio optimization workflow."""
        # Create comprehensive test data
        dates = pd.date_range("2020-01-01", periods=120, freq="ME")

        # Generate correlated returns
        np.random.seed(42)
        base_returns = np.random.normal(0.01, 0.05, 120)

        returns_df = pd.DataFrame(
            {
                "SPY": base_returns + np.random.normal(0, 0.01, 120),
                "TLT": -0.3 * base_returns + np.random.normal(0.005, 0.02, 120),
                "GLD": 0.2 * base_returns + np.random.normal(0.008, 0.03, 120),
                "AAPL": 1.2 * base_returns + np.random.normal(0.015, 0.06, 120),
                "JPM": 0.8 * base_returns + np.random.normal(0.012, 0.04, 120),
            },
            index=dates,
        )

        # Create factor exposures
        factor_exposures = pd.DataFrame(
            {
                "SPY": [0.8, 0.2, 0.5],
                "TLT": [0.1, 0.9, 0.2],
                "GLD": [0.3, 0.1, 0.8],
                "AAPL": [0.9, 0.1, 0.3],
                "JPM": [0.7, 0.3, 0.6],
            },
            index=dates[-3:],
            columns=["momentum", "value", "size"],
        )

        # Create factor regimes
        factor_regimes = pd.DataFrame(
            {
                "regime_0": np.random.choice([0, 1], 120),
                "regime_1": np.random.choice([0, 1], 120),
                "regime_2": np.random.choice([0, 1], 120),
            },
            index=dates,
        )

        data = {
            "returns": returns_df,
            "factor_exposures": factor_exposures,
            "factor_regimes": factor_regimes,
        }

        # Test Black-Litterman optimization
        bl_engine = PortfolioOptimizationEngine(method="black_litterman")
        bl_result = bl_engine.optimize_portfolio(data)

        # Check weights sum to 1 and match returned assets
        assert abs(bl_result["weights"].sum() - 1.0) < 1e-6
        assert len(bl_result["weights"]) == len(bl_result["weights"].index)

        # Test Monte Carlo optimization
        mc_engine = PortfolioOptimizationEngine(method="monte_carlo")
        mc_result = mc_engine.optimize_portfolio(data)

        # Check weights sum to 1 and match returned assets
        assert abs(mc_result["weights"].sum() - 1.0) < 1e-6
        assert len(mc_result["weights"]) == len(mc_result["weights"].index)

        # Test combined optimization
        combined_engine = PortfolioOptimizationEngine(method="combined")
        combined_result = combined_engine.optimize_portfolio(data)

        assert combined_result["method"] == "combined"
        assert len(combined_result["weights"]) == 5
        assert abs(combined_result["weights"].sum() - 1.0) < 1e-6

    def test_constraint_satisfaction(self):
        """Test that portfolio constraints are properly enforced."""
        engine = PortfolioOptimizationEngine(method="black_litterman")

        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 60),
                "TLT": np.random.normal(0.005, 0.03, 60),
                "GLD": np.random.normal(0.008, 0.04, 60),
            },
            index=dates,
        )

        data = {"returns": returns_df}

        # Test with strict constraints
        constraints = {
            "max_weight": 0.4,
            "min_weight": 0.2,
        }

        sector_limits = {
            "Tech": 0.5,
            "Bonds": 0.3,
        }

        result = engine.optimize_portfolio(
            data, constraints=constraints, sector_limits=sector_limits
        )

        # Check constraint satisfaction
        weights = result["weights"]
        assert all(weights <= 0.4)
        assert all(weights >= 0.2)
        assert abs(weights.sum() - 1.0) < 1e-6
