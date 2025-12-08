#!/usr/bin/env python3
"""
Main script for DS 5740 Forecasting Assignment.

This script:
1. Loads portfolio returns and covariates (macro + sentiment)
2. Fits TSLM, ARIMA, and ARIMA-with-errors models
3. Performs train/test split and cross-validation
4. Generates forecasts with prediction intervals
5. Creates diagnostic plots and comparison tables
6. Saves results for notebook analysis
"""

from pathlib import Path
import sys
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantfolio_engine.config import PROCESSED_DATA_DIR, REPORTS_DIR
from quantfolio_engine.data.data_loader import DataLoader
from quantfolio_engine.forecasting.evaluation import (
    ForecastEvaluator,
    TimeSeriesSplit,
    train_test_split_time_series,
)
from quantfolio_engine.forecasting.models import ForecastingEngine
from quantfolio_engine.forecasting.visualization import (
    plot_diagnostics,
    plot_forecasts,
    plot_model_comparison,
    plot_time_series_exploration,
)

# Configure logger
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO", colorize=True)


def construct_portfolio_returns(
    returns_df: pd.DataFrame, weights: Optional[dict] = None
) -> pd.Series:
    """
    Construct equal-weighted or custom-weighted portfolio returns.

    Args:
        returns_df: DataFrame with asset returns
        weights: Dictionary mapping asset names to weights (None for equal-weighted)

    Returns:
        Portfolio returns series
    """
    if weights is None:
        # Equal-weighted portfolio
        n_assets = len(returns_df.columns)
        weights = {asset: 1.0 / n_assets for asset in returns_df.columns}
        logger.info(f"Constructing equal-weighted portfolio with {n_assets} assets")

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted returns
    portfolio_returns = pd.Series(index=returns_df.index, dtype=float, data=0.0)
    for asset, weight in weights.items():
        if asset in returns_df.columns:
            portfolio_returns += weight * returns_df[asset].fillna(0)

    # Drop rows where all assets are NaN
    portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN with 0 (or forward fill if preferred)
    portfolio_returns = portfolio_returns.fillna(0)

    logger.info(
        f"Portfolio returns: mean={portfolio_returns.mean():.4f}, "
        f"std={portfolio_returns.std():.4f}, "
        f"length={len(portfolio_returns)}, "
        f"non-null={portfolio_returns.notna().sum()}"
    )

    return portfolio_returns


def select_covariates(
    macro_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    selected_macro: Optional[list] = None,
    selected_sentiment: Optional[list] = None,
) -> pd.DataFrame:
    """
    Select and combine covariates from macro and sentiment data.

    Args:
        macro_df: Macroeconomic indicators DataFrame
        sentiment_df: Sentiment scores DataFrame
        selected_macro: List of macro indicators to use (None for default)
        selected_sentiment: List of sentiment entities to use (None for default)

    Returns:
        Combined covariates DataFrame
    """
    covariates_list = []

    # Select macro indicators
    if selected_macro is None:
        # Default: Use key macro indicators
        selected_macro = ["CPIAUCSL", "FEDFUNDS", "^VIX", "UNRATE"]
        # Filter to available indicators
        selected_macro = [m for m in selected_macro if m in macro_df.columns]

    if selected_macro:
        macro_selected = macro_df[selected_macro].copy()
        # Convert to returns/percentage changes for macro indicators
        for col in macro_selected.columns:
            if col == "^VIX":
                # VIX is already a level, keep as is or use pct_change
                macro_selected[col] = macro_selected[col].pct_change().fillna(0)
            else:
                # Convert to percentage change
                macro_selected[col] = macro_selected[col].pct_change().fillna(0)

        covariates_list.append(macro_selected)
        logger.info(f"Selected macro covariates: {selected_macro}")

    # Select sentiment indicators
    if selected_sentiment is None:
        # Default: Use first available sentiment entity
        if not sentiment_df.empty:
            selected_sentiment = [sentiment_df.columns[0]]
        else:
            selected_sentiment = []

    if selected_sentiment:
        sentiment_selected = sentiment_df[selected_sentiment].copy()
        covariates_list.append(sentiment_selected)
        logger.info(f"Selected sentiment covariates: {selected_sentiment}")

    # Combine covariates
    if covariates_list:
        covariates = pd.concat(covariates_list, axis=1)
        # Fill missing values
        covariates = covariates.fillna(method="ffill").fillna(0)
        logger.info(f"Combined covariates shape: {covariates.shape}")
        return covariates
    else:
        logger.warning("No covariates selected")
        return pd.DataFrame()


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("DS 5740 Forecasting Assignment - Main Analysis Script")
    logger.info("=" * 80)

    # Create output directory
    output_dir = REPORTS_DIR / "forecasting_assignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Load data
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 80)

    loader = DataLoader(debug=False)
    returns_df, macro_df, sentiment_df = loader.load_all_data(
        start_date="2014-01-01",  # 10+ years of data
        end_date=None,
        save_raw=False,
    )

    if returns_df.empty:
        logger.error("No returns data loaded. Exiting.")
        return

    logger.info(f"Returns data: {returns_df.shape}")
    logger.info(f"Macro data: {macro_df.shape}")
    logger.info(f"Sentiment data: {sentiment_df.shape}")

    # Step 2: Construct portfolio returns
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Constructing Portfolio Returns")
    logger.info("=" * 80)

    # Use equal-weighted portfolio of key assets
    portfolio_assets = ["SPY", "TLT", "GLD"]
    available_assets = [a for a in portfolio_assets if a in returns_df.columns]

    if not available_assets:
        # Fallback to all available assets
        available_assets = returns_df.columns.tolist()[:5]  # Use first 5 assets
        logger.warning(f"Portfolio assets not found, using: {available_assets}")

    portfolio_returns = construct_portfolio_returns(
        returns_df[available_assets], weights=None
    )

    # Step 3: Select covariates
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Selecting Covariates")
    logger.info("=" * 80)

    covariates = select_covariates(macro_df, sentiment_df)

    if covariates.empty:
        logger.warning(
            "No covariates available. Models will use only time series features."
        )

    # Step 4: Train/test split
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Train/Test Split")
    logger.info("=" * 80)

    y_train, y_test, exog_train, exog_test = train_test_split_time_series(
        portfolio_returns, test_size=0.2, exog=covariates
    )

    logger.info(
        f"Training set: {len(y_train)} observations ({y_train.index[0]} to {y_train.index[-1]})"
    )
    logger.info(
        f"Test set: {len(y_test)} observations ({y_test.index[0]} to {y_test.index[-1]})"
    )

    # Step 5: Fit models
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Fitting Models")
    logger.info("=" * 80)

    engine = ForecastingEngine(debug=False)
    evaluator = ForecastEvaluator()
    model_results = {}

    # Fit TSLM
    logger.info("\n--- Fitting TSLM Model ---")
    try:
        tslm_result = engine.fit_tslm(
            y_train, exog=exog_train, trend="ct", seasonal=True, period=12
        )
        model_results["TSLM"] = tslm_result
        logger.success("TSLM fitted successfully")
    except Exception as e:
        logger.error(f"TSLM fitting failed: {e}")

    # Fit ARIMA
    logger.info("\n--- Fitting ARIMA Model ---")
    try:
        arima_result = engine.fit_arima(y_train, auto_select=True, ic="aic")
        model_results["ARIMA"] = arima_result
        logger.success("ARIMA fitted successfully")
    except Exception as e:
        logger.error(f"ARIMA fitting failed: {e}")

    # Fit ARIMA-with-errors
    logger.info("\n--- Fitting ARIMA-with-Errors Model ---")
    if not exog_train.empty:
        try:
            arima_errors_result = engine.fit_arima_errors(
                y_train, exog_train, auto_select=True, ic="aic"
            )
            model_results["ARIMA-Errors"] = arima_errors_result
            logger.success("ARIMA-with-errors fitted successfully")
        except Exception as e:
            logger.error(f"ARIMA-with-errors fitting failed: {e}")
    else:
        logger.warning("Skipping ARIMA-with-errors: No covariates available")

    # Step 6: Generate forecasts
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Generating Forecasts")
    logger.info("=" * 80)

    forecast_steps = len(y_test)
    forecasts = {}

    for model_name, result in model_results.items():
        model_key = model_name.lower().replace("-", "_")
        if model_key == "arima_errors":
            model_key = "arima_errors"
        elif model_key == "tslm":
            model_key = "tslm"
        else:
            model_key = "arima"

        logger.info(f"\n--- Forecasting with {model_name} ---")
        try:
            if model_key in ["tslm", "arima_errors"]:
                forecast_df = engine.forecast(
                    model_key, steps=forecast_steps, exog=exog_test, alpha=0.05
                )
            else:
                forecast_df = engine.forecast(
                    model_key, steps=forecast_steps, alpha=0.05
                )

            forecasts[model_name] = forecast_df
            logger.success(f"{model_name} forecast generated")

        except Exception as e:
            logger.error(f"{model_name} forecast failed: {e}")

    # Step 7: Evaluate models
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Evaluating Models")
    logger.info("=" * 80)

    evaluation_results = {}

    for model_name, forecast_df in forecasts.items():
        model_key = model_name.lower().replace("-", "_")
        if model_key == "arima_errors":
            model_key = "arima_errors"
        elif model_key == "tslm":
            model_key = "tslm"
        else:
            model_key = "arima"

        if model_key in engine.fitted_models:
            result = model_results[model_name]
            evaluation = evaluator.evaluate_model(result, y_test, forecast_df)
            evaluation_results[model_name] = evaluation

            logger.info(f"\n{model_name} Metrics:")
            for metric, value in evaluation["metrics"].items():
                logger.info(f"  {metric.upper()}: {value:.4f}")

    # Step 8: Generate plots
    logger.info("\n" + "=" * 80)
    logger.info("Step 8: Generating Plots")
    logger.info("=" * 80)

    # Time series exploration
    logger.info("Creating time series exploration plot...")
    fig_explore = plot_time_series_exploration(
        portfolio_returns, title="Portfolio Returns - Time Series Exploration"
    )
    fig_explore.savefig(
        output_dir / "01_time_series_exploration.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_explore)

    # Forecast plots for each model
    for model_name, forecast_df in forecasts.items():
        logger.info(f"Creating forecast plot for {model_name}...")
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_forecasts(
            y_train,
            y_test,
            forecast_df,
            model_name=model_name,
            show_intervals=True,
            ax=ax,
        )
        fig.savefig(
            output_dir / f"02_forecast_{model_name.lower().replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Diagnostic plots
    for model_name, result in model_results.items():
        logger.info(f"Creating diagnostic plot for {model_name}...")
        try:
            fig_diag = plot_diagnostics(
                result["residuals"], result["fitted_values"], model_name=model_name
            )
            fig_diag.savefig(
                output_dir
                / f"03_diagnostics_{model_name.lower().replace(' ', '_')}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_diag)
        except Exception as e:
            logger.warning(f"Diagnostic plot failed for {model_name}: {e}")

    # Model comparison
    logger.info("Creating model comparison plots...")
    for metric in ["rmse", "mae", "mape"]:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_model_comparison(evaluation_results, metric=metric, ax=ax)
            fig.savefig(
                output_dir / f"04_comparison_{metric}.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Comparison plot failed for {metric}: {e}")

    # Step 9: Save results
    logger.info("\n" + "=" * 80)
    logger.info("Step 9: Saving Results")
    logger.info("=" * 80)

    # Save forecasts
    for model_name, forecast_df in forecasts.items():
        forecast_df.to_csv(
            output_dir / f"forecast_{model_name.lower().replace(' ', '_')}.csv"
        )
        logger.info(f"Saved forecast for {model_name}")

    # Save evaluation results
    evaluation_summary = []
    for model_name, eval_result in evaluation_results.items():
        row = {"model": model_name}
        row.update(eval_result["metrics"])
        if "model_info" in eval_result:
            row.update(eval_result["model_info"])
        evaluation_summary.append(row)

    eval_df = pd.DataFrame(evaluation_summary)
    eval_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
    logger.info("Saved evaluation summary")

    # Save model diagnostics
    diagnostics_summary = []
    for model_name, result in model_results.items():
        row = {"model": model_name}
        if "diagnostics" in result:
            row.update(result["diagnostics"])
        diagnostics_summary.append(row)

    diag_df = pd.DataFrame(diagnostics_summary)
    diag_df.to_csv(output_dir / "diagnostics_summary.csv", index=False)
    logger.info("Saved diagnostics summary")

    # Save data for notebook
    portfolio_returns.to_csv(output_dir / "portfolio_returns.csv")
    if not covariates.empty:
        covariates.to_csv(output_dir / "covariates.csv")
    y_train.to_csv(output_dir / "y_train.csv")
    y_test.to_csv(output_dir / "y_test.csv")

    logger.info("\n" + "=" * 80)
    logger.success("Analysis Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
