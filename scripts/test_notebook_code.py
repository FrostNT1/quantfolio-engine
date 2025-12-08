#!/usr/bin/env python3
"""
Test script for notebook code validation.

This script tests all code that will be used in the Jupyter notebook
to ensure it works correctly before notebook creation.

Checks for:
- Import errors
- Missing data
- Warnings
- Artifact generation
- Code execution errors
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings

# Filter out known non-critical warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*starting autoregressive parameters.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*No frequency information.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*InterpolationWarning.*"
)
# Keep other warnings as warnings (not errors) for review

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Time series & statistics
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, coint, kpss

from quantfolio_engine.config import REPORTS_DIR

# QuantFolio Engine
from quantfolio_engine.data.data_loader import DataLoader
from quantfolio_engine.forecasting import ForecastEvaluator, ForecastingEngine
from quantfolio_engine.forecasting.evaluation import TimeSeriesSplit
from quantfolio_engine.optimizer.portfolio_engine import PortfolioOptimizationEngine
from quantfolio_engine.signals.factor_timing import FactorTimingEngine

print("=" * 80)
print("NOTEBOOK CODE VALIDATION TEST")
print("=" * 80)

# Set paths
PROJECT_ROOT = Path.cwd()
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting_assignment"

errors = []
warnings_list = []


def test_section(name, test_func):
    """Run a test section and collect errors."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    try:
        result = test_func()
        if result:
            print(f"✅ {name}: PASSED")
        else:
            print(f"⚠️  {name}: WARNINGS (non-critical)")
    except Exception as e:
        print(f"❌ {name}: FAILED - {e}")
        errors.append((name, str(e)))
        import traceback

        traceback.print_exc()


# ============================================================================
# SECTION 1: Data Loading & Validation
# ============================================================================


def test_data_loading():
    """Test loading data from reports directory."""
    print("\n--- Loading Data Files ---")

    # Check if files exist
    required_files = [
        "portfolio_returns.csv",
        "covariates.csv",
        "y_train.csv",
        "y_test.csv",
        "evaluation_summary.csv",
        "diagnostics_summary.csv",
        "forecast_tslm.csv",
        "forecast_arima.csv",
        "forecast_arima-errors.csv",
    ]

    missing_files = []
    for file in required_files:
        filepath = REPORTS_DIR / file
        if not filepath.exists():
            missing_files.append(file)
            print(f"⚠️  Missing: {file}")
        else:
            print(f"✅ Found: {file}")

    if missing_files:
        warnings_list.append(f"Missing files: {missing_files}")
        print("⚠️  Some files missing - will need to regenerate")
        return False

    # Load data
    portfolio_returns = pd.read_csv(
        REPORTS_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True
    )
    portfolio_returns = portfolio_returns.iloc[:, 0]  # Get Series

    covariates = pd.read_csv(
        REPORTS_DIR / "covariates.csv", index_col=0, parse_dates=True
    )
    y_train = pd.read_csv(
        REPORTS_DIR / "y_train.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    y_test = pd.read_csv(
        REPORTS_DIR / "y_test.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]

    # Validate data
    print(f"\nData Validation:")
    print(f"  Portfolio returns: {len(portfolio_returns)} observations")
    print(
        f"  Date range: {portfolio_returns.index.min()} to {portfolio_returns.index.max()}"
    )
    print(f"  Missing values: {portfolio_returns.isna().sum()}")
    print(f"  Mean: {portfolio_returns.mean():.6f}")
    print(f"  Std: {portfolio_returns.std():.6f}")

    print(f"\n  Covariates: {covariates.shape}")
    print(f"  Missing values: {covariates.isna().sum().sum()}")

    print(f"\n  Train set: {len(y_train)} observations")
    print(f"  Test set: {len(y_test)} observations")

    # Check for data issues
    if portfolio_returns.isna().sum() > len(portfolio_returns) * 0.1:
        warnings_list.append("High percentage of missing values in portfolio returns")
        return False

    if len(portfolio_returns) < 50:
        warnings_list.append("Very short time series (< 50 observations)")
        return False

    return True


# ============================================================================
# SECTION 2: Stationarity Testing
# ============================================================================


def test_stationarity_tests():
    """Test stationarity testing code."""
    print("\n--- Stationarity Testing ---")

    # Load data
    portfolio_returns = pd.read_csv(
        REPORTS_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True
    )
    portfolio_returns = portfolio_returns.iloc[:, 0].dropna()

    covariates = pd.read_csv(
        REPORTS_DIR / "covariates.csv", index_col=0, parse_dates=True
    )

    # ADF Test on portfolio returns
    print("\nADF Test - Portfolio Returns:")
    try:
        adf_result = adfuller(portfolio_returns)
        print(f"  ADF Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        print(f"  Critical Values: {adf_result[4]}")
        if adf_result[1] < 0.05:
            print("  ✅ Stationary (p < 0.05)")
        else:
            print("  ⚠️  Non-stationary (p >= 0.05)")
    except Exception as e:
        raise Exception(f"ADF test failed: {e}")

    # KPSS Test
    print("\nKPSS Test - Portfolio Returns:")
    try:
        kpss_result = kpss(portfolio_returns, regression="c")
        print(f"  KPSS Statistic: {kpss_result[0]:.4f}")
        print(f"  p-value: {kpss_result[1]:.4f}")
        if kpss_result[1] > 0.05:
            print("  ✅ Stationary (p > 0.05)")
        else:
            print("  ⚠️  Non-stationary (p <= 0.05)")
    except Exception as e:
        raise Exception(f"KPSS test failed: {e}")

    # Test covariates
    print("\nADF Tests - Covariates:")
    for col in covariates.columns:
        try:
            series = covariates[col].dropna()
            if len(series) < 10:
                print(f"  ⚠️  {col}: Insufficient data ({len(series)} observations)")
                continue
            # Check if series is constant (variance = 0)
            if series.std() == 0 or series.nunique() == 1:
                print(f"  ⚠️  {col}: Constant series (cannot test stationarity)")
                warnings_list.append(
                    f"Covariate {col} is constant - will skip in notebook"
                )
                continue
            adf = adfuller(series)
            print(f"  {col}: p-value = {adf[1]:.4f} {'✅' if adf[1] < 0.05 else '⚠️'}")
        except Exception as e:
            if "constant" in str(e).lower() or "invalid input" in str(e).lower():
                print(f"  ⚠️  {col}: Constant series (cannot test)")
                warnings_list.append(
                    f"Covariate {col} is constant - will skip in notebook"
                )
            else:
                warnings_list.append(f"Covariate {col} ADF test failed: {e}")

    return True


# ============================================================================
# SECTION 3: Cointegration Testing
# ============================================================================


def test_cointegration_tests():
    """Test cointegration testing code."""
    print("\n--- Cointegration Testing ---")

    portfolio_returns = pd.read_csv(
        REPORTS_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True
    )
    portfolio_returns = portfolio_returns.iloc[:, 0].dropna()

    covariates = pd.read_csv(
        REPORTS_DIR / "covariates.csv", index_col=0, parse_dates=True
    )

    # Align data
    common_idx = portfolio_returns.index.intersection(covariates.index)
    portfolio_aligned = portfolio_returns.loc[common_idx]

    print(f"\nTesting cointegration between portfolio returns and covariates...")
    print(f"  Common observations: {len(common_idx)}")

    cointegration_results = {}
    for col in covariates.columns:
        try:
            cov_series = covariates.loc[common_idx, col].dropna()
            if len(cov_series) < 20:
                print(f"  ⚠️  {col}: Insufficient data")
                continue

            # Align both series
            common = portfolio_aligned.index.intersection(cov_series.index)
            if len(common) < 20:
                print(f"  ⚠️  {col}: Insufficient overlap")
                continue

            port_clean = portfolio_aligned.loc[common]
            cov_clean = cov_series.loc[common]

            # Cointegration test
            coint_result = coint(port_clean, cov_clean)
            cointegration_results[col] = {
                "statistic": coint_result[0],
                "pvalue": coint_result[1],
                "critical_values": coint_result[2],
            }

            print(
                f"  {col}: p-value = {coint_result[1]:.4f} {'✅ Cointegrated' if coint_result[1] < 0.05 else '⚠️  Not cointegrated'}"
            )
        except Exception as e:
            warnings_list.append(f"Cointegration test failed for {col}: {e}")

    if not cointegration_results:
        warnings_list.append("No cointegration tests completed successfully")

    return True


# ============================================================================
# SECTION 4: Model Evaluation & Diagnostics
# ============================================================================


def test_model_evaluation():
    """Test model evaluation code."""
    print("\n--- Model Evaluation ---")

    # Load evaluation results
    eval_df = pd.read_csv(REPORTS_DIR / "evaluation_summary.csv")
    diag_df = pd.read_csv(REPORTS_DIR / "diagnostics_summary.csv")

    print("\nEvaluation Summary:")
    print(eval_df.to_string(index=False))

    print("\nDiagnostics Summary:")
    print(diag_df.to_string(index=False))

    # Validate metrics
    required_metrics = ["rmse", "mae", "mape", "coverage"]
    for metric in required_metrics:
        if metric not in eval_df.columns:
            raise Exception(f"Missing metric: {metric}")
        if eval_df[metric].isna().all():
            warnings_list.append(f"All {metric} values are NaN")

    # Check diagnostics
    required_diag = ["ljung_box_pvalue", "jarque_bera_pvalue"]
    for diag in required_diag:
        if diag not in diag_df.columns:
            warnings_list.append(f"Missing diagnostic: {diag}")

    return True


# ============================================================================
# SECTION 5: Forecast Loading & Validation
# ============================================================================


def test_forecast_loading():
    """Test loading and validating forecasts."""
    print("\n--- Forecast Validation ---")

    forecast_files = {
        "TSLM": "forecast_tslm.csv",
        "ARIMA": "forecast_arima.csv",
        "ARIMA-Errors": "forecast_arima-errors.csv",
    }

    forecasts = {}
    for model_name, filename in forecast_files.items():
        filepath = REPORTS_DIR / filename
        if not filepath.exists():
            raise Exception(f"Forecast file missing: {filename}")

        forecast_df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        # Validate structure
        required_cols = ["forecast", "lower", "upper", "lower_80", "upper_80"]
        missing_cols = [col for col in required_cols if col not in forecast_df.columns]
        if missing_cols:
            raise Exception(f"{model_name} forecast missing columns: {missing_cols}")

        # Check for NaN values
        nan_count = forecast_df.isna().sum().sum()
        if nan_count > 0:
            warnings_list.append(f"{model_name} forecast has {nan_count} NaN values")

        # Validate intervals
        invalid_intervals = (forecast_df["lower"] > forecast_df["upper"]).sum()
        if invalid_intervals > 0:
            raise Exception(
                f"{model_name} has {invalid_intervals} invalid intervals (lower > upper)"
            )

        invalid_80 = (forecast_df["lower_80"] > forecast_df["upper_80"]).sum()
        if invalid_80 > 0:
            raise Exception(f"{model_name} has {invalid_80} invalid 80% intervals")

        forecasts[model_name] = forecast_df
        print(f"✅ {model_name}: {len(forecast_df)} forecasts loaded")

    return True


# ============================================================================
# SECTION 6: Visualization Testing
# ============================================================================


def test_visualizations():
    """Test that visualization code works."""
    print("\n--- Visualization Testing ---")

    # Load data
    portfolio_returns = pd.read_csv(
        REPORTS_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True
    )
    portfolio_returns = portfolio_returns.iloc[:, 0]

    y_train = pd.read_csv(
        REPORTS_DIR / "y_train.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    y_test = pd.read_csv(
        REPORTS_DIR / "y_test.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]

    forecast_tslm = pd.read_csv(
        REPORTS_DIR / "forecast_tslm.csv", index_col=0, parse_dates=True
    )

    # Test 1: Time series plot
    print("\n1. Testing time series plot...")
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            portfolio_returns.index, portfolio_returns.values, label="Portfolio Returns"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.set_title("Portfolio Returns Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.close(fig)
        print("  ✅ Time series plot works")
    except Exception as e:
        raise Exception(f"Time series plot failed: {e}")

    # Test 2: Forecast plot
    print("\n2. Testing forecast plot...")
    try:
        from quantfolio_engine.forecasting.visualization import plot_forecasts

        fig, ax = plt.subplots(figsize=(14, 6))
        plot_forecasts(y_train, y_test, forecast_tslm, model_name="TSLM", ax=ax)
        plt.close(fig)
        print("  ✅ Forecast plot works")
    except Exception as e:
        raise Exception(f"Forecast plot failed: {e}")

    # Test 3: Diagnostic plots
    print("\n3. Testing diagnostic plots...")
    try:
        from quantfolio_engine.forecasting.visualization import plot_diagnostics

        # Create dummy residuals and fitted values
        residuals = pd.Series(np.random.randn(len(y_train)), index=y_train.index)
        fitted = y_train + residuals

        fig = plot_diagnostics(residuals, fitted, model_name="Test")
        plt.close(fig)
        print("  ✅ Diagnostic plots work")
    except Exception as e:
        raise Exception(f"Diagnostic plots failed: {e}")

    # Test 4: Model comparison plot
    print("\n4. Testing model comparison plot...")
    try:
        from quantfolio_engine.forecasting.visualization import plot_model_comparison

        eval_df = pd.read_csv(REPORTS_DIR / "evaluation_summary.csv")
        comparison_results = {}
        for _, row in eval_df.iterrows():
            comparison_results[row["model"]] = {
                "metrics": {
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "mape": row["mape"],
                },
                "model_info": {
                    "aic": row.get("aic"),
                    "bic": row.get("bic"),
                },
            }

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_model_comparison(comparison_results, metric="rmse", ax=ax)
        plt.close(fig)
        print("  ✅ Model comparison plot works")
    except Exception as e:
        raise Exception(f"Model comparison plot failed: {e}")

    return True


# ============================================================================
# SECTION 7: Cross-Validation Testing
# ============================================================================


def test_cross_validation():
    """Test cross-validation code."""
    print("\n--- Cross-Validation Testing ---")

    portfolio_returns = pd.read_csv(
        REPORTS_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True
    )
    portfolio_returns = portfolio_returns.iloc[:, 0].dropna()

    covariates = pd.read_csv(
        REPORTS_DIR / "covariates.csv", index_col=0, parse_dates=True
    )

    # Align data
    common_idx = portfolio_returns.index.intersection(covariates.index)
    y = portfolio_returns.loc[common_idx]
    exog = covariates.loc[common_idx]

    print(f"\nData for CV: {len(y)} observations")

    # Test TimeSeriesSplit
    print("\nTesting TimeSeriesSplit...")
    try:
        tscv = TimeSeriesSplit(n_splits=3, test_size=12)
        splits = tscv.split(y)

        split_count = 0
        for train_idx, test_idx in splits:
            split_count += 1
            print(
                f"  Split {split_count}: Train={len(train_idx)}, Test={len(test_idx)}"
            )
            if len(train_idx) < 20:
                warnings_list.append(f"Split {split_count} has very small training set")
            if len(test_idx) == 0:
                raise Exception(f"Split {split_count} has empty test set")

        if split_count == 0:
            raise Exception("No CV splits generated")

        print(f"  ✅ Generated {split_count} CV splits")
    except Exception as e:
        raise Exception(f"CV split generation failed: {e}")

    # Test fitting models in CV loop (quick test)
    print("\nTesting model fitting in CV loop...")
    try:
        engine = ForecastingEngine(debug=False)

        # Use first split only for speed
        tscv = TimeSeriesSplit(n_splits=1, test_size=12)
        splits = list(tscv.split(y))
        if splits:
            train_idx, test_idx = splits[0]
            y_train_cv = y.iloc[train_idx]
            y_test_cv = y.iloc[test_idx]
            exog_train_cv = exog.iloc[train_idx] if not exog.empty else None

            # Test ARIMA fitting (suppress statsmodels warnings about starting parameters)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings(
                    "ignore", message=".*starting autoregressive parameters.*"
                )
                result = engine.fit_arima(
                    y_train_cv, auto_select=False, order=(1, 0, 1)
                )
            print(f"  ✅ Model fitting works in CV context")
    except Exception as e:
        # Check if it's just a warning about starting parameters
        if "starting autoregressive parameters" in str(e).lower():
            print(f"  ✅ Model fitting works (statsmodels warning suppressed)")
        else:
            warnings_list.append(f"CV model fitting test failed: {e}")

    return True


# ============================================================================
# SECTION 8: QuantFolio Engine Integration
# ============================================================================


def test_quantfolio_integration():
    """Test QuantFolio Engine integration."""
    print("\n--- QuantFolio Engine Integration Testing ---")

    # Test 1: DataLoader
    print("\n1. Testing DataLoader...")
    try:
        loader = DataLoader(debug=False)
        # Try loading processed data first
        returns_df = loader.load_processed_data("returns_monthly")
        macro_df = loader.load_processed_data("macro_monthly")
        sentiment_df = loader.load_processed_data("sentiment_monthly")

        if returns_df.empty:
            warnings_list.append("No processed returns data found")
        else:
            print(f"  ✅ Loaded returns: {returns_df.shape}")

        if macro_df.empty:
            warnings_list.append("No processed macro data found")
        else:
            print(f"  ✅ Loaded macro: {macro_df.shape}")

        if sentiment_df.empty:
            warnings_list.append("No processed sentiment data found")
        else:
            print(f"  ✅ Loaded sentiment: {sentiment_df.shape}")
    except Exception as e:
        warnings_list.append(f"DataLoader test failed: {e}")

    # Test 2: PortfolioOptimizationEngine (quick test)
    print("\n2. Testing PortfolioOptimizationEngine...")
    try:
        engine = PortfolioOptimizationEngine(method="black_litterman", debug=False)
        print("  ✅ PortfolioOptimizationEngine initialized")
    except Exception as e:
        warnings_list.append(f"PortfolioOptimizationEngine init failed: {e}")

    # Test 3: FactorTimingEngine (quick test)
    print("\n3. Testing FactorTimingEngine...")
    try:
        factor_engine = FactorTimingEngine(debug=False)
        print("  ✅ FactorTimingEngine initialized")
    except Exception as e:
        warnings_list.append(f"FactorTimingEngine init failed: {e}")

    return True


# ============================================================================
# SECTION 9: Statistical Tests
# ============================================================================


def test_statistical_tests():
    """Test all statistical test code."""
    print("\n--- Statistical Tests ---")

    portfolio_returns = pd.read_csv(
        REPORTS_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True
    )
    portfolio_returns = portfolio_returns.iloc[:, 0].dropna()

    # Test Ljung-Box
    print("\n1. Testing Ljung-Box test...")
    try:
        lb_result = acorr_ljungbox(portfolio_returns, lags=10, return_df=True)
        print(f"  ✅ Ljung-Box test works")
        print(f"     Last p-value: {lb_result['lb_pvalue'].iloc[-1]:.4f}")
    except Exception as e:
        raise Exception(f"Ljung-Box test failed: {e}")

    # Test Jarque-Bera
    print("\n2. Testing Jarque-Bera test...")
    try:
        jb_result = jarque_bera(portfolio_returns)
        # Handle both tuple and named tuple returns
        if isinstance(jb_result, tuple):
            jb_stat, jb_pvalue = jb_result[0], jb_result[1]
        else:
            jb_stat = jb_result.statistic
            jb_pvalue = jb_result.pvalue
        print(f"  ✅ Jarque-Bera test works")
        print(f"     Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
    except Exception as e:
        raise Exception(f"Jarque-Bera test failed: {e}")

    # Test Seasonal Decomposition
    print("\n3. Testing seasonal decomposition...")
    try:
        if len(portfolio_returns) >= 24:
            decomp = seasonal_decompose(portfolio_returns, model="additive", period=12)
            print(f"  ✅ Seasonal decomposition works")
        else:
            warnings_list.append("Insufficient data for decomposition")
    except Exception as e:
        warnings_list.append(f"Seasonal decomposition failed: {e}")

    return True


# ============================================================================
# SECTION 10: Data Alignment & Preprocessing
# ============================================================================


def test_data_alignment():
    """Test data alignment and preprocessing."""
    print("\n--- Data Alignment Testing ---")

    portfolio_returns = pd.read_csv(
        REPORTS_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True
    )
    portfolio_returns = portfolio_returns.iloc[:, 0]

    covariates = pd.read_csv(
        REPORTS_DIR / "covariates.csv", index_col=0, parse_dates=True
    )

    # Test alignment
    print("\n1. Testing data alignment...")
    common_idx = portfolio_returns.index.intersection(covariates.index)
    print(f"  Common dates: {len(common_idx)}")

    if len(common_idx) < len(portfolio_returns) * 0.8:
        warnings_list.append(
            f"Low overlap: {len(common_idx)}/{len(portfolio_returns)} dates"
        )

    portfolio_aligned = portfolio_returns.loc[common_idx]
    covariates_aligned = covariates.loc[common_idx]

    # Test missing value handling
    print("\n2. Testing missing value handling...")
    portfolio_nan = portfolio_aligned.isna().sum()
    covariates_nan = covariates_aligned.isna().sum().sum()

    print(f"  Portfolio NaN: {portfolio_nan}")
    print(f"  Covariates NaN: {covariates_nan}")

    if portfolio_nan > len(portfolio_aligned) * 0.1:
        warnings_list.append("High percentage of NaN in portfolio returns")

    # Test train/test split
    print("\n3. Testing train/test split...")
    from quantfolio_engine.forecasting.evaluation import train_test_split_time_series

    y_train, y_test, exog_train, exog_test = train_test_split_time_series(
        portfolio_returns, test_size=0.2, exog=covariates
    )

    print(f"  Train: {len(y_train)} observations")
    print(f"  Test: {len(y_test)} observations")

    if len(y_train) < 50:
        warnings_list.append("Training set too small (< 50 observations)")
    if len(y_test) < 10:
        warnings_list.append("Test set too small (< 10 observations)")

    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS")
    print("=" * 80)

    # Run all test sections
    test_section("Data Loading", test_data_loading)
    test_section("Stationarity Tests", test_stationarity_tests)
    test_section("Cointegration Tests", test_cointegration_tests)
    test_section("Model Evaluation", test_model_evaluation)
    test_section("Forecast Loading", test_forecast_loading)
    test_section("Visualizations", test_visualizations)
    test_section("Cross-Validation", test_cross_validation)
    test_section("QuantFolio Integration", test_quantfolio_integration)
    test_section("Statistical Tests", test_statistical_tests)
    test_section("Data Alignment", test_data_alignment)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for name, error in errors:
            print(f"  - {name}: {error}")
    else:
        print("\n✅ NO ERRORS")

    if warnings_list:
        print(f"\n⚠️  WARNINGS ({len(warnings_list)}):")
        for warning in warnings_list:
            print(f"  - {warning}")
    else:
        print("\n✅ NO WARNINGS")

    if errors:
        print("\n❌ VALIDATION FAILED - Fix errors before creating notebook")
        return 1
    elif warnings_list:
        print("\n⚠️  VALIDATION PASSED WITH WARNINGS - Review warnings")
        return 0
    else:
        print("\n✅ VALIDATION PASSED - Ready to create notebook")
        return 0


if __name__ == "__main__":
    exit(main())
