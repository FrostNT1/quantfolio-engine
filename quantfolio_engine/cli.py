"""
Command Line Interface for QuantFolio Engine.

This module provides CLI commands for data operations, model training, and portfolio optimization.
"""

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


if __name__ == "__main__":
    app()
