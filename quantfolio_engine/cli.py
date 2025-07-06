"""
Command Line Interface for QuantFolio Engine.

This module provides CLI commands for data operations, model training, and portfolio optimization.
"""

from typing import Optional

from loguru import logger
import typer

from .config import DEFAULT_START_DATE
from .data.data_loader import DataLoader

app = typer.Typer(
    name="quantfolio",
    help="QuantFolio Engine - Smart Portfolio Construction Using Factor Timing",
    add_completion=False,
)


@app.command()
def fetch_data(
    start_date: str = typer.Option(
        DEFAULT_START_DATE,
        "--start-date",
        "-s",
        help="Start date for data fetch (YYYY-MM-DD)",
    ),
    end_date: str = typer.Option(
        None,
        "--end-date",
        "-e",
        help="End date for data fetch (YYYY-MM-DD), defaults to current date",
    ),
    save_raw: bool = typer.Option(
        True,
        "--save-raw/--no-save-raw",
        help="Whether to save raw data files",
    ),
    data_type: str = typer.Option(
        "all",
        "--type",
        "-t",
        help="Type of data to fetch: 'returns', 'macro', 'sentiment', or 'all'",
    ),
):
    """
    Fetch financial data from various sources.

    This command downloads asset returns, macroeconomic indicators, and sentiment data
    from Yahoo Finance, FRED, and News API respectively.
    """
    logger.info("Starting data fetch operation...")

    loader = DataLoader()

    if data_type == "all" or data_type == "returns":
        logger.info("Fetching asset returns...")
        returns_df = loader.fetch_asset_returns(start_date, end_date, save_raw)
        if not returns_df.empty:
            logger.success(
                f"Successfully fetched returns for {len(returns_df.columns)} assets"
            )
        else:
            logger.warning("No returns data fetched")

    if data_type == "all" or data_type == "macro":
        logger.info("Fetching macro indicators...")
        macro_df = loader.fetch_macro_indicators(start_date, end_date, save_raw)
        if not macro_df.empty:
            logger.success(
                f"Successfully fetched {len(macro_df.columns)} macro indicators"
            )
        else:
            logger.warning("No macro data fetched")

    if data_type == "all" or data_type == "sentiment":
        logger.info("Fetching sentiment data...")
        sentiment_df = loader.fetch_sentiment_data(start_date, end_date, save_raw)
        if not sentiment_df.empty:
            logger.success(
                f"Successfully fetched sentiment for {len(sentiment_df.columns)} entities/topics"
            )
        else:
            logger.warning("No sentiment data fetched")

    logger.success("Data fetch operation completed!")


@app.command()
def list_data():
    """List available data files."""
    from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR

    logger.info("Available data files:")

    # Raw data
    if RAW_DATA_DIR.exists():
        logger.info(f"\nRaw data ({RAW_DATA_DIR}):")
        for file in sorted(RAW_DATA_DIR.glob("*.csv")):
            size = file.stat().st_size / 1024  # KB
            logger.info(f"  {file.name} ({size:.1f} KB)")
    else:
        logger.info("\nNo raw data directory found")

    # Processed data
    if PROCESSED_DATA_DIR.exists():
        logger.info(f"\nProcessed data ({PROCESSED_DATA_DIR}):")
        for file in sorted(PROCESSED_DATA_DIR.glob("*.csv")):
            size = file.stat().st_size / 1024  # KB
            logger.info(f"  {file.name} ({size:.1f} KB)")
    else:
        logger.info("\nNo processed data directory found")


@app.command()
def clean_data():
    """Validate and clean existing data files."""
    import pandas as pd

    from .config import PROCESSED_DATA_DIR

    logger.info("Validating and cleaning data...")

    issues_found = []
    files_processed = 0

    # Check processed data files
    if PROCESSED_DATA_DIR.exists():
        for file in PROCESSED_DATA_DIR.glob("*.csv"):
            try:
                logger.info(f"Checking {file.name}...")
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                files_processed += 1

                # Check for common issues
                file_issues = []

                # Check for missing values
                missing_count = df.isnull().sum().sum()
                if missing_count > 0:
                    file_issues.append(f"{missing_count} missing values")

                # Check for infinite values
                inf_count = df.isin([float("inf"), float("-inf")]).sum().sum()
                if inf_count > 0:
                    file_issues.append(f"{inf_count} infinite values")

                # Check for duplicate dates
                if df.index.duplicated().any():
                    file_issues.append("duplicate dates in index")

                # Check for reasonable data ranges
                if "returns" in file.name.lower():
                    # Returns should typically be between -1 and 1 (or reasonable bounds)
                    extreme_returns = ((df > 0.5) | (df < -0.5)).sum().sum()
                    if extreme_returns > 0:
                        file_issues.append(
                            f"{extreme_returns} extreme return values (>50% or <-50%)"
                        )

                if file_issues:
                    issues_found.append(f"{file.name}: {', '.join(file_issues)}")
                else:
                    logger.success(f"✓ {file.name} - No issues found")

            except Exception as e:
                issues_found.append(f"{file.name}: Error reading file - {str(e)}")

    # Summary
    if issues_found:
        logger.warning(f"Found {len(issues_found)} issues in {files_processed} files:")
        for issue in issues_found:
            logger.warning(f"  - {issue}")

        logger.info("\nTo fix these issues, you can:")
        logger.info("  1. Re-fetch the data: quantfolio fetch-data")
        logger.info("  2. Check your API keys: quantfolio status")
        logger.info("  3. Review the raw data files in data/raw/")
    else:
        logger.success(f"✓ All {files_processed} data files are clean!")

    if files_processed == 0:
        logger.warning(
            "No data files found. Run 'quantfolio fetch-data' to download data first."
        )


@app.command()
def status():
    """Show system status and configuration."""
    from .config import (
        ASSET_UNIVERSE,
        FRED_API_KEY,
        MACRO_INDICATORS,
        NEWS_API_KEY,
        SENTIMENT_ENTITIES,
        SENTIMENT_TOPICS,
    )

    logger.info("QuantFolio Engine Status:")
    logger.info(f"  Asset Universe: {len(ASSET_UNIVERSE)} assets")
    logger.info(f"  Macro Indicators: {len(MACRO_INDICATORS)} indicators")
    logger.info(f"  Sentiment Entities: {len(SENTIMENT_ENTITIES)} entities")
    logger.info(f"  Sentiment Topics: {len(SENTIMENT_TOPICS)} topics")
    logger.info(f"  FRED API Key: {'✓' if FRED_API_KEY else '✗'}")
    logger.info(f"  News API Key: {'✓' if NEWS_API_KEY else '✗'}")


@app.command()
def validate_data():
    """Validate data quality without cleaning."""
    import pandas as pd

    from .config import PROCESSED_DATA_DIR

    logger.info("Validating data quality...")

    if not PROCESSED_DATA_DIR.exists():
        logger.error(
            "No processed data directory found. Run 'quantfolio fetch-data' first."
        )
        return

    data_files = list(PROCESSED_DATA_DIR.glob("*.csv"))
    if not data_files:
        logger.warning(
            "No data files found. Run 'quantfolio fetch-data' to download data first."
        )
        return

    logger.info(f"Found {len(data_files)} data files to validate...")

    validation_results = {}

    for file in data_files:
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)

            # Basic statistics
            stats = {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": df.isnull().sum().sum(),
                "duplicate_dates": df.index.duplicated().sum(),
                "date_range": f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            }

            # File-specific checks
            if "returns" in file.name.lower():
                stats["extreme_returns"] = ((df > 0.5) | (df < -0.5)).sum().sum()
                stats["mean_return"] = df.mean().mean()
                stats["std_return"] = df.std().mean()

            validation_results[file.name] = stats

        except Exception as e:
            validation_results[file.name] = {"error": str(e)}

    # Display results
    logger.info("\nData Validation Results:")
    logger.info("=" * 50)

    for filename, stats in validation_results.items():
        logger.info(f"\n{filename}:")
        if "error" in stats:
            logger.error(f"  Error: {stats['error']}")
        else:
            logger.info(f"  Rows: {stats['rows']}, Columns: {stats['columns']}")
            logger.info(f"  Date Range: {stats['date_range']}")
            logger.info(f"  Missing Values: {stats['missing_values']}")
            logger.info(f"  Duplicate Dates: {stats['duplicate_dates']}")

            if "extreme_returns" in stats:
                logger.info(f"  Extreme Returns: {stats['extreme_returns']}")
                logger.info(f"  Mean Return: {stats['mean_return']:.4f}")
                logger.info(f"  Std Return: {stats['std_return']:.4f}")


@app.command()
def normalize_data():
    """Normalize processed returns, macro, and sentiment data and save as *_normalized.csv."""
    import pandas as pd

    from .config import PROCESSED_DATA_DIR

    loader = DataLoader()

    # Normalize returns
    returns_file = PROCESSED_DATA_DIR / "returns_monthly.csv"
    if returns_file.exists():
        returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        norm_returns = loader.normalize_returns(returns_df)
        out_file = PROCESSED_DATA_DIR / "returns_monthly_normalized.csv"
        norm_returns.to_csv(out_file)
        logger.success(f"Saved normalized returns to {out_file}")
    else:
        logger.warning("returns_monthly.csv not found")

    # Normalize macro
    macro_file = PROCESSED_DATA_DIR / "macro_monthly.csv"
    if macro_file.exists():
        macro_df = pd.read_csv(macro_file, index_col=0, parse_dates=True)
        norm_macro = loader.normalize_macro(macro_df)
        out_file = PROCESSED_DATA_DIR / "macro_monthly_normalized.csv"
        norm_macro.to_csv(out_file)
        logger.success(f"Saved normalized macro data to {out_file}")
    else:
        logger.warning("macro_monthly.csv not found")

    # Normalize sentiment
    sentiment_file = PROCESSED_DATA_DIR / "sentiment_monthly.csv"
    if sentiment_file.exists():
        sentiment_df = pd.read_csv(sentiment_file, index_col=0, parse_dates=True)
        norm_sentiment = loader.normalize_sentiment(sentiment_df)
        out_file = PROCESSED_DATA_DIR / "sentiment_monthly_normalized.csv"
        norm_sentiment.to_csv(out_file)
        logger.success(f"Saved normalized sentiment data to {out_file}")
    else:
        logger.warning("sentiment_monthly.csv not found")


@app.command()
def generate_signals(
    lookback_period: int = typer.Option(
        60,
        "--lookback",
        "-l",
        help="Lookback period for rolling factor exposure calculation (months)",
    ),
    n_regimes: int = typer.Option(
        3,
        "--regimes",
        "-r",
        help="Number of regimes to detect",
    ),
    factor_method: str = typer.Option(
        "macro",
        "--factor-method",
        "-f",
        help="Factor generation method: 'macro', 'fama_french', or 'simple'",
    ),
    returns_file: Optional[str] = typer.Option(
        None,
        "--returns",
        help="Path to returns CSV file (default: data/processed/returns_monthly.csv)",
    ),
    factors_file: Optional[str] = typer.Option(
        None,
        "--factors",
        help="Path to factors CSV file (default: data/processed/macro_monthly.csv)",
    ),
):
    """
    Generate factor timing signals from processed data.

    This command calculates factor exposures using rolling regression and detects
    factor regimes using clustering and HMM methods.
    """
    from .signals.factor_timing import FactorTimingEngine

    logger.info("Starting factor timing signal generation...")

    # Initialize factor timing engine
    engine = FactorTimingEngine(
        lookback_period=lookback_period,
        n_regimes=n_regimes,
        factor_method=factor_method,
    )

    # Generate signals
    results = engine.generate_factor_timing_signals(
        returns_file=returns_file, factors_file=factors_file
    )

    if results:
        logger.success("Factor timing signals generated successfully!")

        # Log summary
        if "factor_exposures" in results and not results["factor_exposures"].empty:
            exposures = results["factor_exposures"]
            logger.info(
                f"Generated factor exposures for {len(exposures.columns)} assets"
            )
            logger.info(
                f"Date range: {exposures.index.min()} to {exposures.index.max()}"
            )

        if "rolling_regimes" in results and not results["rolling_regimes"].empty:
            logger.info("Rolling statistics regime detection completed")

        if "hmm_regimes" in results and not results["hmm_regimes"].empty:
            logger.info("HMM regime detection completed")
    else:
        logger.error("Failed to generate factor timing signals")


@app.command()
def optimize_portfolio(
    method: str = typer.Option(
        "combined",
        "--method",
        "-m",
        help="Optimization method: 'black_litterman', 'monte_carlo', or 'combined'",
    ),
    target_return: Optional[float] = typer.Option(
        None,
        "--target-return",
        "-r",
        help="Target annual return (e.g., 0.08 for 8%)",
    ),
    max_volatility: Optional[float] = typer.Option(
        None,
        "--max-volatility",
        "-v",
        help="Maximum annual volatility (e.g., 0.15 for 15%)",
    ),
    max_weight: Optional[float] = typer.Option(
        None,
        "--max-weight",
        help="Maximum weight per asset (e.g., 0.3 for 30%)",
    ),
    min_weight: Optional[float] = typer.Option(
        None,
        "--min-weight",
        help="Minimum weight per asset (e.g., 0.05 for 5%)",
    ),
    sector_limits: Optional[str] = typer.Option(
        None,
        "--sector-limits",
        help='Sector limits as JSON string (e.g., \'{"Tech": 0.4, "Finance": 0.3}\')',
    ),
    generate_frontier: bool = typer.Option(
        False,
        "--frontier",
        "-f",
        help="Generate efficient frontier",
    ),
    frontier_points: int = typer.Option(
        20,
        "--frontier-points",
        help="Number of points for efficient frontier",
    ),
    save_results: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save optimization results to file",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results (default: reports/)",
    ),
    risk_free_rate: float = typer.Option(
        0.02,
        "--risk-free-rate",
        help="Risk-free rate for Sharpe ratio calculation",
    ),
    max_drawdown: float = typer.Option(
        0.15,
        "--max-drawdown",
        help="Maximum drawdown constraint",
    ),
    confidence_level: float = typer.Option(
        0.95,
        "--confidence-level",
        help="Confidence level for risk metrics",
    ),
    random_state: Optional[int] = typer.Option(
        None,
        "--random-state",
        help="Random state for reproducibility",
    ),
    # Black-Litterman specific parameters
    bl_lambda: Optional[str] = typer.Option(
        "auto",
        "--bl-lambda",
        help="Black-Litterman λ: 'auto' for calibration, or float value",
    ),
    bl_lambda_range: Optional[str] = typer.Option(
        None,
        "--bl-lambda-range",
        help="λ calibration range as 'min,max' (e.g., '0.5,5.0')",
    ),
    bl_gamma: float = typer.Option(
        0.3,
        "--bl-gamma",
        help="Grand view blend parameter γ (0.0 = pure π, 1.0 = pure μ̄)",
    ),
    bl_view_strength: float = typer.Option(
        1.5,
        "--bl-view-strength",
        help="Black-Litterman view strength multiplier",
    ),
    bl_auto: bool = typer.Option(
        False,
        "--bl-auto",
        help="Enable auto-calibration for Black-Litterman (λ + γ)",
    ),
):
    """
    Optimize portfolio using various methods.

    This command runs portfolio optimization using Black-Litterman, Monte Carlo,
    or combined methods with factor timing integration.
    """
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from .optimizer.portfolio_engine import PortfolioOptimizationEngine

    logger.info(f"Starting portfolio optimization using {method} method...")

    # Parse sector limits if provided
    parsed_sector_limits = None
    if sector_limits:
        try:
            parsed_sector_limits = json.loads(sector_limits)
            logger.info(f"Using sector limits: {parsed_sector_limits}")
        except json.JSONDecodeError:
            logger.error("Invalid sector limits JSON format")
            return

    # Build constraints dictionary
    constraints = {}
    if max_weight is not None:
        constraints["max_weight"] = max_weight
    if min_weight is not None:
        constraints["min_weight"] = min_weight

    # Initialize optimization engine
    engine = PortfolioOptimizationEngine(
        method=method,
        risk_free_rate=risk_free_rate,
        max_drawdown=max_drawdown,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    # Set Black-Litterman parameters if using BL method
    if method == "black_litterman":
        # Handle auto-calibration
        if bl_auto:
            bl_lambda = "auto"
            bl_gamma = 0.4  # Stronger grand view blend for auto mode

        # Set BL parameters in the engine
        engine.set_bl_parameters(
            lambda_param=bl_lambda,
            gamma=bl_gamma,
            view_strength=bl_view_strength,
            lambda_range=bl_lambda_range,
        )

    # Load data
    logger.info("Loading data for optimization...")
    data = engine.load_data()

    if not data:
        logger.error(
            "Failed to load required data. Please run 'quantfolio fetch-data' and 'quantfolio generate-signals' first."
        )
        return

    # Run optimization
    try:
        if generate_frontier:
            logger.info(
                f"Generating efficient frontier with {frontier_points} points..."
            )
            frontier = engine.generate_efficient_frontier(
                data=data,
                n_points=frontier_points,
                constraints=constraints,
                sector_limits=parsed_sector_limits,
            )

            logger.success("Efficient frontier generated successfully!")
            logger.info(f"Frontier contains {len(frontier['returns'])} points")
            logger.info(
                f"Return range: {frontier['returns'].min():.3f} to {frontier['returns'].max():.3f}"
            )
            logger.info(
                f"Volatility range: {frontier['volatilities'].min():.3f} to {frontier['volatilities'].max():.3f}"
            )

            # Save frontier results
            if save_results:
                output_path = Path(output_dir) if output_dir else Path("reports")
                output_path.mkdir(exist_ok=True)

                frontier_file = output_path / "efficient_frontier.csv"
                frontier_df = pd.DataFrame(
                    {
                        "return": frontier["returns"],
                        "volatility": frontier["volatilities"],
                    }
                )
                frontier_df.to_csv(frontier_file)
                logger.success(f"Saved efficient frontier to {frontier_file}")

                # Save weights for each point
                weights_file = output_path / "frontier_weights.csv"
                weights_df = pd.DataFrame(frontier["weights"])
                weights_df.to_csv(weights_file)
                logger.success(f"Saved frontier weights to {weights_file}")

            results = {
                "method": method,
                "frontier": frontier,
                "constraints": constraints,
                "sector_limits": parsed_sector_limits,
            }

        else:
            logger.info("Running portfolio optimization...")
            results = engine.optimize_portfolio(
                data=data,
                constraints=constraints,
                target_return=target_return,
                max_volatility=max_volatility,
                sector_limits=parsed_sector_limits,
            )

            logger.success("Portfolio optimization completed successfully!")

            # Display results
            logger.info("\nPortfolio Optimization Results:")
            logger.info("=" * 50)
            logger.info(f"Method: {results['method']}")
            logger.info(f"Expected Return: {results['expected_return']:.3f}")
            logger.info(f"Volatility: {results['volatility']:.3f}")
            logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")

            if "max_drawdown" in results:
                logger.info(f"Max Drawdown: {results['max_drawdown']:.3f}")
            if "var_95" in results:
                logger.info(f"VaR (95%): {results['var_95']:.3f}")

            # Display top weights
            weights = results["weights"]
            logger.info("\nTop 5 Asset Weights:")

            # Handle both pandas Series and numpy array weights
            if isinstance(weights, pd.Series):
                top_weights = weights.nlargest(5)
                for asset, weight in top_weights.items():
                    logger.info(f"  {asset}: {weight:.3f}")
            elif isinstance(weights, np.ndarray):
                # For numpy arrays, we need asset names and weights
                asset_names = list(data["returns"].columns)
                if len(asset_names) == len(weights):
                    # Create a Series for easier handling
                    weights_series = pd.Series(weights, index=asset_names)
                    top_weights = weights_series.nlargest(5)
                    for asset, weight in top_weights.items():
                        logger.info(f"  {asset}: {weight:.3f}")
                else:
                    logger.warning("Asset names and weights have different lengths")
            else:
                logger.warning(f"Unexpected weights type: {type(weights)}")

            # Save results
            if save_results:
                output_path = Path(output_dir) if output_dir else Path("reports")
                output_path.mkdir(exist_ok=True)

                # Save weights
                weights_file = output_path / "optimal_weights.csv"
                if isinstance(weights, pd.Series):
                    weights.to_csv(weights_file)
                elif isinstance(weights, np.ndarray):
                    # For numpy arrays, create a DataFrame with asset names
                    asset_names = list(data["returns"].columns)
                    if len(asset_names) == len(weights):
                        weights_df = pd.DataFrame(
                            {"asset": asset_names, "weight": weights}
                        )
                        weights_df.to_csv(weights_file, index=False)
                    else:
                        # Fallback: save as numpy array
                        np.save(weights_file.with_suffix(".npy"), weights)
                        logger.warning(
                            f"Saved weights as numpy array to {weights_file.with_suffix('.npy')}"
                        )
                else:
                    logger.warning(f"Cannot save weights of type {type(weights)}")
                    return
                logger.success(f"Saved optimal weights to {weights_file}")

                # Save summary
                summary_file = output_path / "optimization_summary.txt"
                with open(summary_file, "w") as f:
                    f.write("Portfolio Optimization Summary\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Method: {results['method']}\n")
                    f.write(f"Expected Return: {results['expected_return']:.3f}\n")
                    f.write(f"Volatility: {results['volatility']:.3f}\n")
                    f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}\n")
                    if "max_drawdown" in results:
                        f.write(f"Max Drawdown: {results['max_drawdown']:.3f}\n")
                    if "var_95" in results:
                        f.write(f"VaR (95%): {results['var_95']:.3f}\n")
                    f.write(f"\nConstraints: {constraints}\n")
                    if parsed_sector_limits:
                        f.write(f"Sector Limits: {parsed_sector_limits}\n")

                logger.success(f"Saved optimization summary to {summary_file}")

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return

    logger.success("Portfolio optimization completed!")


if __name__ == "__main__":
    app()
