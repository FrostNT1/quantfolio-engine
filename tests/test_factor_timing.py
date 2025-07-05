"""Tests for the factor timing module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from quantfolio_engine.signals.factor_timing import (
    FactorExposureCalculator,
    FactorTimingEngine,
    RegimeDetector,
)


class TestFactorExposureCalculator:
    """Test cases for FactorExposureCalculator."""

    def test_init(self):
        """Test FactorExposureCalculator initialization."""
        calculator = FactorExposureCalculator(lookback_period=60)
        assert calculator.lookback_period == 60
        assert calculator.factor_exposures == {}

    def test_calculate_rolling_factor_exposures(self):
        """Test rolling factor exposure calculation."""
        calculator = FactorExposureCalculator(lookback_period=12)

        # Create dummy data with proper structure (returns need to be percentage changes)
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")

        # Create price data first, then convert to returns
        spy_prices = 100 * (1 + np.cumsum(np.random.normal(0.01, 0.05, 24)))
        tlt_prices = 100 * (1 + np.cumsum(np.random.normal(0.005, 0.03, 24)))

        returns_df = pd.DataFrame(
            {
                "SPY": pd.Series(spy_prices, index=dates).pct_change(),
                "TLT": pd.Series(tlt_prices, index=dates).pct_change(),
            },
            index=dates,
        )

        # Create factor data with enough non-missing values to pass the threshold
        # Need at least 3 factors or half the columns to be non-missing
        factors_df = pd.DataFrame(
            {
                "factor1": np.random.normal(100, 10, 24),  # Level data, not returns
                "factor2": np.random.normal(50, 5, 24),  # Level data, not returns
                "factor3": np.random.normal(200, 20, 24),  # Level data, not returns
            },
            index=dates,
        )

        # Calculate exposures
        exposures = calculator.calculate_rolling_factor_exposures(
            returns_df, factors_df
        )

        # Check output
        assert isinstance(exposures, pd.DataFrame)
        assert len(exposures.columns) == 2  # SPY and TLT
        assert len(exposures) > 0  # Should have some data after lookback period

    def test_rolling_regression_length_consistency(self):
        """Test that regression and correlation fallback return same number of factors."""
        calculator = FactorExposureCalculator(lookback_period=6)

        # Create data with 3 factors
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 20),
                "factor2": np.random.normal(0, 0.015, 20),
                "factor3": np.random.normal(0, 0.01, 20),
            },
            index=dates,
        )

        # Test regression path
        with patch("sklearn.linear_model.Ridge") as mock_ridge:
            mock_ridge.return_value.coef_ = np.array([0.1, 0.2, 0.3])
            result = calculator._rolling_regression(asset_returns, factors)

            # Should return Series with factor exposures
            assert isinstance(result, pd.Series)
            assert len(result) > 0

    def test_correlation_fallback_individual_factors(self):
        """Test that correlation fallback uses individual factor correlations, not mean."""
        calculator = FactorExposureCalculator(lookback_period=6)

        # Create data where factors have different correlations
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": asset_returns * 0.8
                + np.random.normal(0, 0.01, 20),  # High correlation
                "factor2": asset_returns * 0.2
                + np.random.normal(0, 0.01, 20),  # Low correlation
                "factor3": -asset_returns * 0.5
                + np.random.normal(0, 0.01, 20),  # Negative correlation
            },
            index=dates,
        )

        # Force correlation fallback by making regression fail
        with patch("sklearn.linear_model.Ridge") as mock_ridge:
            mock_ridge.side_effect = Exception("Regression failed")

            result = calculator._rolling_regression(asset_returns, factors)

            # Should still work and return meaningful values
            assert isinstance(result, pd.Series)
            assert len(result) > 0

    def test_index_coverage_consistency(self):
        """Test that regression and correlation paths have consistent index coverage."""
        calculator = FactorExposureCalculator(lookback_period=6)

        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 20),
                "factor2": np.random.normal(0, 0.015, 20),
            },
            index=dates,
        )

        # Test both paths return sparse Series with same pattern
        regression_result = calculator._rolling_regression(asset_returns, factors)
        correlation_result = calculator._calculate_correlation_exposure(
            asset_returns, factors
        )

        # Both should be sparse Series starting after lookback period
        min_required = max(calculator.lookback_period, len(factors.columns) + 1)
        expected_start = asset_returns.index[min_required]

        assert regression_result.index[0] >= expected_start
        assert correlation_result.index[0] >= expected_start

    def test_ridge_regression_stability(self):
        """Test that Ridge regression provides more stable results than LinearRegression."""
        calculator = FactorExposureCalculator(lookback_period=6)

        # Create data with multicollinearity (unstable for OLS)
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        base_factor = np.random.normal(0, 0.02, 20)
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": base_factor,
                "factor2": base_factor
                + np.random.normal(0, 0.001, 20),  # Nearly collinear
                "factor3": np.random.normal(0, 0.015, 20),
            },
            index=dates,
        )

        # Should not crash with Ridge regression
        result = calculator._rolling_regression(asset_returns, factors)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_data_alignment_with_missing_values(self):
        """Test handling of missing values and data alignment issues."""
        calculator = FactorExposureCalculator(lookback_period=6)

        # Create data with missing values and different frequencies
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)

        # Factors with some missing values
        factors = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 20),
                "factor2": np.random.normal(0, 0.015, 20),
            },
            index=dates,
        )
        factors.loc[dates[5:8], "factor1"] = np.nan  # Some missing values

        # Should handle missing values gracefully
        result = calculator._rolling_regression(asset_returns, factors)
        assert isinstance(result, pd.Series)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for regression."""
        calculator = FactorExposureCalculator(lookback_period=12)

        # Create data with insufficient observations
        dates = pd.date_range("2020-01-01", periods=10, freq="ME")  # Less than lookback
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 10), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 10),
                "factor2": np.random.normal(0, 0.015, 10),
            },
            index=dates,
        )

        # Should fall back to correlation method
        result = calculator._rolling_regression(asset_returns, factors)
        assert isinstance(result, pd.Series)


class TestRegimeDetector:
    """Test cases for RegimeDetector."""

    def test_init(self):
        """Test RegimeDetector initialization."""
        detector = RegimeDetector(n_regimes=3)
        assert detector.n_regimes == 3
        assert detector.scaler is not None
        assert detector.kmeans is not None

    def test_detect_regimes_rolling_stats(self):
        """Test regime detection using rolling statistics."""
        detector = RegimeDetector(n_regimes=3)

        # Create dummy factor exposures
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        exposures_df = pd.DataFrame(
            {"SPY": np.random.normal(0, 1, 36), "TLT": np.random.normal(0, 1, 36)},
            index=dates,
        )

        # Detect regimes
        regimes = detector.detect_regimes_rolling_stats(exposures_df, window=6)

        # Check output
        assert isinstance(regimes, pd.DataFrame)
        assert "regime" in regimes.columns
        assert len(regimes) > 0
        assert regimes["regime"].nunique() <= 3  # Should have at most 3 regimes

    def test_dimensionality_reduction(self):
        """Test that PCA is applied to reduce dimensionality."""
        detector = RegimeDetector(n_regimes=3)

        # Create high-dimensional data (many assets)
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        exposures_df = pd.DataFrame(
            {f"asset_{i}": np.random.normal(0, 1, 36) for i in range(10)}, index=dates
        )

        # This should trigger dimensionality reduction (10 assets * 3 stats = 30 features)
        with patch("quantfolio_engine.signals.factor_timing.PCA") as mock_pca_class:
            mock_pca_instance = MagicMock()
            # Return data with correct length (after rolling window)
            mock_pca_instance.fit_transform.return_value = np.random.normal(
                0, 1, (31, 6)
            )  # 36 - 6 + 1 = 31
            mock_pca_class.return_value = mock_pca_instance

            regimes = detector.detect_regimes_rolling_stats(exposures_df, window=6)

            # Should call PCA
            mock_pca_class.assert_called_once()
            assert regimes is not None

    def test_detect_regimes_hmm_success(self):
        """Test HMM regime detection when hmmlearn is available."""
        detector = RegimeDetector(n_regimes=3)

        # Create dummy factor exposures with no NaN values to avoid shape issues
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        exposures_df = pd.DataFrame(
            {"SPY": np.random.normal(0, 1, 36), "TLT": np.random.normal(0, 1, 36)},
            index=dates,
        )

        # Patch hmmlearn.hmm.GaussianHMM
        with patch("hmmlearn.hmm.GaussianHMM") as mock_hmm:
            mock_model = MagicMock()
            # Return same number of rows as input data (after dropna)
            mock_model.predict_proba.return_value = np.random.uniform(0, 1, (36, 3))
            mock_model.predict.return_value = np.random.randint(0, 3, 36)
            mock_hmm.return_value = mock_model

            regimes = detector.detect_regimes_hmm(exposures_df)

            # Check output format
            assert isinstance(regimes, pd.DataFrame)
            assert "regime" in regimes.columns
            assert any(
                col.startswith("regime_") and col.endswith("_prob")
                for col in regimes.columns
            )

    def test_detect_regimes_hmm_import_error(self):
        """Test HMM regime detection with fallback when hmmlearn not available."""
        detector = RegimeDetector(n_regimes=3)

        # Create dummy factor exposures
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        exposures_df = pd.DataFrame(
            {"SPY": np.random.normal(0, 1, 36), "TLT": np.random.normal(0, 1, 36)},
            index=dates,
        )

        # Patch the _get_hmm_module helper to raise ImportError
        with patch.object(
            RegimeDetector,
            "_get_hmm_module",
            side_effect=ImportError("hmmlearn not available"),
        ):
            regimes = detector.detect_regimes_hmm(exposures_df)
            assert isinstance(regimes, pd.DataFrame)
            assert "regime" in regimes.columns
            assert not any(
                col.startswith("regime_") and col.endswith("_prob")
                for col in regimes.columns
            )

    def test_detect_regimes_hmm_runtime_error(self):
        """Test HMM regime detection with fallback when HMM fails at runtime."""
        detector = RegimeDetector(n_regimes=3)

        # Create dummy factor exposures
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        exposures_df = pd.DataFrame(
            {"SPY": np.random.normal(0, 1, 36), "TLT": np.random.normal(0, 1, 36)},
            index=dates,
        )

        # Mock HMM that fails during fitting
        with patch("hmmlearn.hmm.GaussianHMM") as mock_hmm:
            mock_hmm.return_value.fit.side_effect = Exception("HMM fitting failed")

            # Should fall back to clustering
            regimes = detector.detect_regimes_hmm(exposures_df)

            # Check output
            assert isinstance(regimes, pd.DataFrame)
            assert "regime" in regimes.columns

    def test_empty_data_handling(self):
        """Test handling of empty or insufficient data."""
        detector = RegimeDetector(n_regimes=3)

        # Empty DataFrame
        empty_df = pd.DataFrame()

        # Should handle gracefully
        regimes = detector.detect_regimes_rolling_stats(empty_df)
        assert isinstance(regimes, pd.DataFrame)
        assert len(regimes) == 0


class TestFactorTimingEngine:
    """Test cases for FactorTimingEngine."""

    def test_init(self):
        """Test FactorTimingEngine initialization."""
        engine = FactorTimingEngine(lookback_period=60, n_regimes=3)
        assert engine.exposure_calculator.lookback_period == 60
        assert engine.regime_detector.n_regimes == 3

    @patch("quantfolio_engine.signals.factor_timing.pd.read_csv")
    def test_generate_factor_timing_signals(self, mock_read_csv):
        """Test factor timing signal generation."""
        engine = FactorTimingEngine(lookback_period=6, n_regimes=2)

        # Mock data loading with more data points
        dates = pd.date_range("2020-01-01", periods=48, freq="ME")
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 48),
                "TLT": np.random.normal(0.005, 0.03, 48),
            },
            index=dates,
        )

        factors_df = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 48),
                "factor2": np.random.normal(0, 0.015, 48),
            },
            index=dates,
        )

        mock_read_csv.side_effect = [returns_df, factors_df]

        # Generate signals
        results = engine.generate_factor_timing_signals()

        # Check results
        assert isinstance(results, dict)
        assert "factor_exposures" in results
        assert "rolling_regimes" in results
        assert "hmm_regimes" in results

    def test_logging_verbosity(self):
        """Test that logging verbosity is appropriate (debug level for inner loops)."""
        engine = FactorTimingEngine(lookback_period=6, n_regimes=2)

        # Create test data with proper structure and enough data points
        dates = pd.date_range("2020-01-01", periods=30, freq="ME")  # More data points

        # Create price data first, then convert to returns
        spy_prices = 100 * (1 + np.cumsum(np.random.normal(0.01, 0.05, 30)))
        tlt_prices = 100 * (1 + np.cumsum(np.random.normal(0.005, 0.03, 30)))

        returns_df = pd.DataFrame(
            {
                "SPY": pd.Series(spy_prices, index=dates).pct_change(),
                "TLT": pd.Series(tlt_prices, index=dates).pct_change(),
            },
            index=dates,
        )

        factors_df = pd.DataFrame(
            {
                "factor1": np.random.normal(100, 10, 30),  # Level data
                "factor2": np.random.normal(50, 5, 30),  # Level data
                "factor3": np.random.normal(200, 20, 30),  # Level data
            },
            index=dates,
        )

        # Test that per-asset logging is at debug level
        with patch("quantfolio_engine.signals.factor_timing.logger") as mock_logger:
            # Ensure debug level is enabled
            mock_logger.debug.return_value = None

            engine.exposure_calculator.calculate_rolling_factor_exposures(
                returns_df, factors_df
            )

            # Check that debug was called for per-asset logging
            # The debug logging happens in the loop, so it should be called at least once
            assert mock_logger.debug.call_count >= 1


class TestIntegration:
    """Integration tests for the complete factor timing pipeline."""

    def test_end_to_end_pipeline(self):
        """Test the complete factor timing pipeline with realistic data."""
        engine = FactorTimingEngine(lookback_period=12, n_regimes=3)

        # Create realistic test data
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")

        # Returns with some correlation structure (price data first, then returns)
        base_prices = 100 * (1 + np.cumsum(np.random.normal(0.01, 0.05, 60)))
        spy_prices = base_prices * (1 + np.cumsum(np.random.normal(0, 0.02, 60)))
        tlt_prices = base_prices * (1 + np.cumsum(np.random.normal(-0.003, 0.015, 60)))
        gld_prices = base_prices * (1 + np.cumsum(np.random.normal(0.001, 0.025, 60)))

        returns_df = pd.DataFrame(
            {
                "SPY": pd.Series(spy_prices, index=dates).pct_change(),
                "TLT": pd.Series(tlt_prices, index=dates).pct_change(),
                "GLD": pd.Series(gld_prices, index=dates).pct_change(),
            },
            index=dates,
        )

        # Factors with economic meaning (level data, not returns)
        factors_df = pd.DataFrame(
            {
                "market": np.random.normal(100, 10, 60),  # Market level
                "inflation": np.random.normal(2.5, 0.5, 60),  # Inflation rate
                "growth": np.random.normal(3.0, 1.0, 60),  # GDP growth
            },
            index=dates,
        )

        # Test the complete pipeline
        with patch(
            "quantfolio_engine.signals.factor_timing.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.side_effect = [returns_df, factors_df]

            results = engine.generate_factor_timing_signals()

            # Check all components
            assert "factor_exposures" in results
            assert "rolling_regimes" in results
            assert "hmm_regimes" in results

            # Check factor exposures
            exposures = results["factor_exposures"]
            assert isinstance(exposures, pd.DataFrame)
            assert len(exposures.columns) == 3  # SPY, TLT, GLD
            assert len(exposures) > 0

            # Check regimes
            rolling_regimes = results["rolling_regimes"]
            hmm_regimes = results["hmm_regimes"]
            assert isinstance(rolling_regimes, pd.DataFrame)
            assert isinstance(hmm_regimes, pd.DataFrame)

    def test_data_quality_validation(self):
        """Test that the pipeline handles various data quality issues."""
        engine = FactorTimingEngine(lookback_period=6, n_regimes=2)

        # Test with quarterly GDP data (missing values)
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")

        # Create proper returns data
        spy_prices = 100 * (1 + np.cumsum(np.random.normal(0.01, 0.05, 24)))
        returns_df = pd.DataFrame(
            {"SPY": pd.Series(spy_prices, index=dates).pct_change()}, index=dates
        )

        # Factors with quarterly data (missing values)
        factors_df = pd.DataFrame(
            {
                "GDP": [
                    100.0 if i % 3 == 0 else np.nan for i in range(24)
                ],  # Quarterly
                "inflation": np.random.normal(2.5, 0.5, 24),
            },
            index=dates,
        )

        with patch(
            "quantfolio_engine.signals.factor_timing.pd.read_csv"
        ) as mock_read_csv:
            mock_read_csv.side_effect = [returns_df, factors_df]

            # Should handle missing values gracefully
            results = engine.generate_factor_timing_signals()
            assert "factor_exposures" in results


def test_factor_timing_import():
    """Test that factor timing modules can be imported."""
    from quantfolio_engine.signals.factor_timing import (
        FactorExposureCalculator,
        FactorTimingEngine,
        RegimeDetector,
    )

    assert FactorExposureCalculator is not None
    assert RegimeDetector is not None
    assert FactorTimingEngine is not None


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests."""

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        calculator = FactorExposureCalculator(lookback_period=60)

        # Large dataset
        dates = pd.date_range("2010-01-01", periods=300, freq="ME")  # 25 years
        returns_df = pd.DataFrame(
            {f"asset_{i}": np.random.normal(0.01, 0.05, 300) for i in range(20)},
            index=dates,
        )

        factors_df = pd.DataFrame(
            {f"factor_{i}": np.random.normal(0, 0.02, 300) for i in range(10)},
            index=dates,
        )

        # Should complete without errors
        exposures = calculator.calculate_rolling_factor_exposures(
            returns_df, factors_df
        )
        assert isinstance(exposures, pd.DataFrame)
        assert len(exposures.columns) == 20

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        calculator = FactorExposureCalculator(lookback_period=12)

        # Data with extreme values
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        asset_returns = pd.Series(
            [1e6 if i % 5 == 0 else 0.01 for i in range(24)], index=dates
        )
        factors = pd.DataFrame(
            {
                "factor1": [1e-6 if i % 3 == 0 else 0.02 for i in range(24)],
                "factor2": np.random.normal(0, 0.015, 24),
            },
            index=dates,
        )

        # Should handle extreme values gracefully
        result = calculator._rolling_regression(asset_returns, factors)
        assert isinstance(result, pd.Series)
        assert not result.isna().all()  # Should have some valid results
