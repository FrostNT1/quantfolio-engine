"""
Command Line Interface for QuantFolio Engine.

This module provides CLI commands for data operations, model training, and portfolio optimization.
"""

from typing import Optional

from loguru import logger
import pandas as pd
import typer

from .backtesting import WalkForwardBacktester
from .config import DEFAULT_START_DATE
from .data.data_loader import DataLoader
from .plots import (
    plot_aggregate_metrics,
    plot_backtest_results,
    plot_performance_comparison,
    plot_weight_evolution,
)

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
    # Add CLI option for transaction cost multiplier
    transaction_costs: Optional[str] = typer.Option(
        None,
        "--transaction-costs",
        help='JSON string mapping asset types to transaction costs (e.g., \'{"ETF":0.0005,"Large_Cap":0.001}\')',
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


@app.command()
def run_backtest(
    method: str = typer.Option(
        "combined",
        "--method",
        "-m",
        help="Optimization method: 'black_litterman', 'monte_carlo', or 'combined'",
    ),
    train_years: int = typer.Option(
        8,
        "--train-years",
        help="Years of data to use for training",
    ),
    test_years: int = typer.Option(
        2,
        "--test-years",
        help="Years of data to use for testing",
    ),
    rebalance_frequency: str = typer.Option(
        "monthly",
        "--rebalance",
        "-r",
        help="Rebalance frequency: 'monthly', 'quarterly', or 'annual'",
    ),
    max_weight: float = typer.Option(
        0.3,
        "--max-weight",
        help="Maximum weight per asset (e.g., 0.3 for 30%)",
    ),
    min_weight: float = typer.Option(
        0.05,
        "--min-weight",
        help="Minimum weight per asset (e.g., 0.05 for 5%)",
    ),
    max_volatility: float = typer.Option(
        0.15,
        "--max-volatility",
        "-v",
        help="Maximum annual volatility (e.g., 0.15 for 15%)",
    ),
    risk_free_rate: float = typer.Option(
        0.045,
        "--risk-free-rate",
        help="Risk-free rate for Sharpe ratio calculation",
    ),
    save_results: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save backtest results to file",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results (default: reports/)",
    ),
    random_state: Optional[int] = typer.Option(
        None,
        "--random-state",
        help="Random state for reproducibility",
    ),
    # Add CLI option for transaction cost multiplier
    transaction_costs: Optional[str] = typer.Option(
        None,
        "--transaction-costs",
        help='JSON string mapping asset types to transaction costs (e.g., \'{"ETF":0.0005,"Large_Cap":0.001}\')',
    ),
):
    """
    Run walk-forward backtesting.

    This command performs walk-forward backtesting with configurable train/test windows
    and rebalance frequencies. It validates data quality, runs portfolio optimization
    on training windows, and evaluates performance on out-of-sample test periods.
    """
    import json
    from pathlib import Path

    import pandas as pd

    logger.info("Starting walk-forward backtesting...")

    # Initialize backtester
    backtester = WalkForwardBacktester(
        train_years=train_years,
        test_years=test_years,
        rebalance_frequency=rebalance_frequency,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        min_weight=min_weight,
        max_volatility=max_volatility,
        random_state=random_state,
        transaction_costs=json.loads(transaction_costs) if transaction_costs else None,
    )

    # Load data
    logger.info("Loading data for backtesting...")
    try:
        # Load returns data
        returns_file = Path("data/processed/returns_monthly.csv")
        if not returns_file.exists():
            logger.error(
                "Returns data not found. Please run 'quantfolio fetch-data' first."
            )
            return

        returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        logger.info(
            f"Loaded returns data: {returns_df.shape[0]} periods, {returns_df.shape[1]} assets"
        )

        # Load factor data if available
        factor_exposures = None
        factor_regimes = None
        sentiment_scores = None
        macro_data = None

        # Try to load factor exposures
        factor_file = Path("data/processed/factor_exposures.csv")
        if factor_file.exists():
            factor_exposures = pd.read_csv(factor_file)
            logger.info(f"Loaded factor exposures: {factor_exposures.shape}")

        # Try to load factor regimes
        regimes_file = Path("data/processed/factor_regimes_hmm.csv")
        if regimes_file.exists():
            factor_regimes = pd.read_csv(regimes_file)
            logger.info(f"Loaded factor regimes: {factor_regimes.shape}")

        # Try to load sentiment data
        sentiment_file = Path("data/processed/sentiment_monthly.csv")
        if sentiment_file.exists():
            sentiment_scores = pd.read_csv(
                sentiment_file, index_col=0, parse_dates=True
            )
            logger.info(f"Loaded sentiment data: {sentiment_scores.shape}")

        # Try to load macro data
        macro_file = Path("data/processed/macro_monthly.csv")
        if macro_file.exists():
            macro_data = pd.read_csv(macro_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded macro data: {macro_data.shape}")

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return

    # Run backtest
    try:
        results = backtester.run_backtest(
            returns_df=returns_df,
            factor_exposures=factor_exposures,
            factor_regimes=factor_regimes,
            sentiment_scores=sentiment_scores,
            macro_data=macro_data,
            method=method,
        )

        if "error" in results:
            logger.error(f"Backtest failed: {results['error']}")
            return

        # Display results
        logger.info("\nBacktest Results:")
        logger.info("=" * 50)

        performance_df = results["performance_history"]
        aggregate_metrics = results["aggregate_metrics"]

        logger.info(f"Total periods: {aggregate_metrics['total_periods']}")
        logger.info(
            f"Average total return: {aggregate_metrics['avg_total_return']:.3f}"
        )
        logger.info(
            f"Average Sharpe ratio: {aggregate_metrics['avg_sharpe_ratio']:.3f}"
        )
        logger.info(
            f"Average Sortino ratio: {aggregate_metrics['avg_sortino_ratio']:.3f}"
        )
        logger.info(
            f"Average Calmar ratio: {aggregate_metrics['avg_calmar_ratio']:.3f}"
        )
        logger.info(
            f"Worst max drawdown: {aggregate_metrics['worst_max_drawdown']:.3f}"
        )
        logger.info(f"Average volatility: {aggregate_metrics['avg_volatility']:.3f}")
        logger.info(f"Hit ratio: {aggregate_metrics['hit_ratio']:.3f}")
        logger.info(
            f"Excess return vs benchmark: {aggregate_metrics['excess_return']:.3f}"
        )
        logger.info(
            f"Excess Sharpe vs benchmark: {aggregate_metrics['excess_sharpe']:.3f}"
        )
        # Add transaction cost reporting
        logger.info(
            f"Total transaction costs: {aggregate_metrics['total_transaction_costs']:.4f}"
        )
        logger.info(
            f"Average transaction cost per period: {aggregate_metrics['avg_transaction_cost']:.4f}"
        )
        logger.info(
            f"Total portfolio turnover: {aggregate_metrics['total_turnover']:.3f}"
        )
        logger.info(
            f"Average turnover per period: {aggregate_metrics['avg_turnover']:.3f}"
        )
        logger.info(
            f"Net total return (after costs): {aggregate_metrics['net_total_return']:.3f}"
        )

        # Save results
        if save_results:
            output_path = Path(output_dir) if output_dir else Path("reports")
            output_path.mkdir(exist_ok=True)

            # Save performance history
            performance_file = output_path / "backtest_performance.csv"
            if isinstance(performance_df, pd.DataFrame):
                performance_df.to_csv(performance_file)
            else:
                pd.DataFrame(performance_df).to_csv(performance_file)
            logger.success(f"Saved performance history to {performance_file}")

            # Save weight history
            weight_df = pd.DataFrame(results["weight_history"])
            weight_file = output_path / "backtest_weights.csv"
            weight_df.to_csv(weight_file)
            logger.success(f"Saved weight history to {weight_file}")

            # Save aggregate metrics
            metrics_file = output_path / "backtest_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(aggregate_metrics, f, indent=2, default=str)
            logger.success(f"Saved aggregate metrics to {metrics_file}")

            # Save summary report
            summary_file = output_path / "backtest_summary.txt"
            with open(summary_file, "w") as f:
                f.write("Walk-Forward Backtest Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Method: {method}\n")
                f.write(f"Train years: {train_years}\n")
                f.write(f"Test years: {test_years}\n")
                f.write(f"Rebalance frequency: {rebalance_frequency}\n")
                f.write(f"Total periods: {aggregate_metrics['total_periods']}\n")
                f.write(
                    f"Average total return: {aggregate_metrics['avg_total_return']:.3f}\n"
                )
                f.write(
                    f"Average Sharpe ratio: {aggregate_metrics['avg_sharpe_ratio']:.3f}\n"
                )
                f.write(
                    f"Average Sortino ratio: {aggregate_metrics['avg_sortino_ratio']:.3f}\n"
                )
                f.write(
                    f"Average Calmar ratio: {aggregate_metrics['avg_calmar_ratio']:.3f}\n"
                )
                f.write(
                    f"Worst max drawdown: {aggregate_metrics['worst_max_drawdown']:.3f}\n"
                )
                f.write(
                    f"Average volatility: {aggregate_metrics['avg_volatility']:.3f}\n"
                )
                f.write(f"Hit ratio: {aggregate_metrics['hit_ratio']:.3f}\n")
                f.write(
                    f"Excess return vs benchmark: {aggregate_metrics['excess_return']:.3f}\n"
                )
                f.write(
                    f"Excess Sharpe vs benchmark: {aggregate_metrics['excess_sharpe']:.3f}\n"
                )
                f.write(
                    f"Total transaction costs: {aggregate_metrics['total_transaction_costs']:.4f}\n"
                )
                f.write(
                    f"Average transaction cost per period: {aggregate_metrics['avg_transaction_cost']:.4f}\n"
                )
                f.write(
                    f"Total portfolio turnover: {aggregate_metrics['total_turnover']:.3f}\n"
                )
                f.write(
                    f"Average turnover per period: {aggregate_metrics['avg_turnover']:.3f}\n"
                )
                f.write(
                    f"Net total return (after costs): {aggregate_metrics['net_total_return']:.3f}\n"
                )

            logger.success(f"Saved backtest summary to {summary_file}")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        return

    logger.success("Walk-forward backtesting completed!")


@app.command()
def plot_backtest(
    performance_file: str = typer.Option(
        "reports/backtest_performance.csv",
        "--performance-file",
        "-p",
        help="Path to backtest performance CSV file",
    ),
    weights_file: Optional[str] = typer.Option(
        None,
        "--weights-file",
        "-w",
        help="Path to backtest weights CSV file",
    ),
    metrics_file: Optional[str] = typer.Option(
        None,
        "--metrics-file",
        "-m",
        help="Path to backtest metrics JSON file",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: reports/)",
    ),
    plot_type: str = typer.Option(
        "all",
        "--type",
        "-t",
        help="Type of plot: 'backtest', 'comparison', 'weights', 'metrics', or 'all'",
    ),
):
    """
    Generate plots from backtest results.

    This command creates various visualizations of backtest performance,
    including cumulative returns, risk metrics, and portfolio evolution.
    """
    import json
    from pathlib import Path

    logger.info("Generating backtest plots...")

    # Set output directory
    output_path = Path(output_dir) if output_dir else Path("reports")
    output_path.mkdir(exist_ok=True)

    # Load performance data
    try:
        performance_df = pd.read_csv(performance_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded performance data: {performance_df.shape}")
    except Exception as e:
        logger.error(f"Error loading performance data: {str(e)}")
        return

    # Generate plots based on type
    if plot_type in ["backtest", "all"]:
        logger.info("Generating backtest results plot...")
        plot_backtest_results(
            performance_df=performance_df,
            save_path=str(output_path / "backtest_results.png"),
        )
        # Add return distribution plot
        from .plots import plot_return_distribution

        logger.info("Generating return distribution histogram...")
        plot_return_distribution(
            performance_df=performance_df,
            save_path=str(output_path / "return_distribution.png"),
        )

    if plot_type in ["comparison", "all"]:
        logger.info("Generating performance comparison plot...")
        plot_performance_comparison(
            performance_df=performance_df,
            save_path=str(output_path / "performance_comparison.png"),
        )

    if plot_type in ["weights", "all"] and weights_file:
        try:
            weights_df = pd.read_csv(weights_file)
            logger.info("Generating weight evolution plot...")
            plot_weight_evolution(
                weight_df=weights_df,
                save_path=str(output_path / "weight_evolution.png"),
            )
        except Exception as e:
            logger.warning(f"Could not generate weight plot: {str(e)}")

    if plot_type in ["metrics", "all"] and metrics_file:
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            logger.info("Generating aggregate metrics plot...")
            plot_aggregate_metrics(
                metrics=metrics, save_path=str(output_path / "aggregate_metrics.png")
            )
        except Exception as e:
            logger.warning(f"Could not generate metrics plot: {str(e)}")

    logger.success(f"Plots saved to {output_path}")


if __name__ == "__main__":
    app()
