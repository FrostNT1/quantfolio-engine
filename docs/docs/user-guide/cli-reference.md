# CLI Reference

This guide provides a complete reference for all QuantFolio Engine command-line interface (CLI) commands.

## Overview

The QuantFolio Engine CLI provides a unified interface for all operations:

```bash
quantfolio [COMMAND] [OPTIONS]
```

## Data Management Commands

### `fetch-data`

Fetch financial data from various sources.

```bash
quantfolio fetch-data [OPTIONS]
```

**Options:**
- `--start-date TEXT`: Start date for data fetch (YYYY-MM-DD) [default: 2010-01-01]
- `--end-date TEXT`: End date for data fetch (YYYY-MM-DD) [default: current date]
- `--save-raw / --no-save-raw`: Whether to save raw data files [default: True]
- `--type TEXT`: Type of data to fetch: 'returns', 'macro', 'sentiment', or 'all' [default: all]

**Examples:**
```bash
# Fetch all data from 2010
quantfolio fetch-data --start-date 2010-01-01

# Fetch only returns data for 2023
quantfolio fetch-data --start-date 2023-01-01 --end-date 2023-12-31 --type returns

# Fetch macro data only
quantfolio fetch-data --type macro --start-date 2020-01-01
```

### `list-data`

List available data files.

```bash
quantfolio list-data
```

**Output:**
- Shows raw and processed data files
- File sizes and modification dates
- Data quality indicators

### `clean-data`

Validate and clean existing data files.

```bash
quantfolio clean-data
```

**Features:**
- Checks for missing values
- Identifies extreme outliers
- Validates data continuity
- Reports data quality issues

### `validate-data`

Validate data quality and completeness.

```bash
quantfolio validate-data
```

**Checks:**
- Data completeness
- Date alignment
- Missing values
- Extreme values
- Data format consistency

### `normalize-data`

Normalize data for analysis.

```bash
quantfolio normalize-data
```

**Normalization Types:**
- Returns: Z-score normalization
- Macro: Rolling standardization
- Sentiment: Bounds clipping to [-1, 1]

### `status`

Show system status and configuration.

```bash
quantfolio status
```

**Information Displayed:**
- API key status
- Data file availability
- System configuration
- Environment information

## Signal Generation Commands

### `generate-signals`

Generate factor timing signals and regime detection.

```bash
quantfolio generate-signals [OPTIONS]
```

**Options:**
- `--lookback INTEGER`: Lookback period for rolling factor exposure calculation (months) [default: 60]
- `--regimes INTEGER`: Number of regimes to detect [default: 3]
- `--factor-method TEXT`: Factor generation method: 'macro', 'fama_french', or 'simple' [default: macro]
- `--returns TEXT`: Path to returns CSV file
- `--factors TEXT`: Path to factors CSV file

**Examples:**
```bash
# Generate signals with default parameters
quantfolio generate-signals

# Use custom lookback and regimes
quantfolio generate-signals --lookback 48 --regimes 4

# Use Fama-French factors
quantfolio generate-signals --factor-method fama_french

# Use custom data files
quantfolio generate-signals --returns data/custom_returns.csv --factors data/custom_factors.csv
```

## Portfolio Optimization Commands

### `optimize-portfolio`

Run portfolio optimization using various methods.

```bash
quantfolio optimize-portfolio [OPTIONS]
```

**Options:**

**Basic Parameters:**
- `--method TEXT`: Optimization method: 'black_litterman', 'monte_carlo', or 'combined' [default: combined]
- `--target-return FLOAT`: Target annual return (e.g., 0.08 for 8%)
- `--max-volatility FLOAT`: Maximum annual volatility (e.g., 0.15 for 15%)
- `--max-weight FLOAT`: Maximum weight per asset (e.g., 0.3 for 30%)
- `--min-weight FLOAT`: Minimum weight per asset (e.g., 0.05 for 5%)
- `--risk-free-rate FLOAT`: Risk-free rate for Sharpe ratio calculation [default: 0.02]
- `--max-drawdown FLOAT`: Maximum drawdown constraint [default: 0.15]
- `--confidence-level FLOAT`: Confidence level for risk metrics [default: 0.95]

**Black-Litterman Specific:**
- `--bl-lambda TEXT`: Black-Litterman λ: 'auto' for calibration, or float value [default: auto]
- `--bl-lambda-range TEXT`: λ calibration range as 'min,max' (e.g., '0.5,5.0')
- `--bl-gamma FLOAT`: Grand view blend parameter γ (0.0 = pure π, 1.0 = pure μ̄) [default: 0.3]
- `--bl-view-strength FLOAT`: Black-Litterman view strength multiplier [default: 1.5]
- `--bl-auto`: Enable auto-calibration for Black-Litterman (λ + γ)

**Output Options:**
- `--save / --no-save`: Save optimization results to file [default: True]
- `--output-dir TEXT`: Output directory for results (default: reports/)
- `--generate-frontier`: Generate efficient frontier
- `--frontier-points INTEGER`: Number of points for efficient frontier [default: 20]

**Examples:**
```bash
# Basic optimization
quantfolio optimize-portfolio --method combined --max-weight 0.3 --min-weight 0.05

# Black-Litterman with custom parameters
quantfolio optimize-portfolio --method black_litterman --bl-auto --bl-view-strength 2.0

# Monte Carlo with constraints
quantfolio optimize-portfolio --method monte_carlo --max-volatility 0.12 --target-return 0.08

# Generate efficient frontier
quantfolio optimize-portfolio --method combined --generate-frontier --frontier-points 30
```

## Backtesting Commands

### `run-backtest`

Run walk-forward backtesting with comprehensive validation.

```bash
quantfolio run-backtest [OPTIONS]
```

**Options:**

**Backtest Parameters:**
- `--method TEXT`: Optimization method: 'black_litterman', 'monte_carlo', or 'combined' [default: combined]
- `--train-years INTEGER`: Years of data to use for training [default: 8]
- `--test-years INTEGER`: Years of data to use for testing [default: 2]
- `--rebalance TEXT`: Rebalance frequency: 'monthly', 'quarterly', or 'annual' [default: monthly]

**Constraints:**
- `--max-weight FLOAT`: Maximum weight per asset (e.g., 0.3 for 30%) [default: 0.3]
- `--min-weight FLOAT`: Minimum weight per asset (e.g., 0.05 for 5%) [default: 0.05]
- `--max-volatility FLOAT`: Maximum annual volatility (e.g., 0.15 for 15%) [default: 0.15]
- `--risk-free-rate FLOAT`: Risk-free rate for Sharpe ratio calculation [default: 0.045]

**Transaction Costs:**
- `--transaction-costs TEXT`: JSON string mapping asset types to transaction costs (e.g., '{"ETF":0.0005,"Large_Cap":0.001}')

**Output Options:**
- `--save / --no-save`: Save backtest results to file [default: True]
- `--output-dir TEXT`: Output directory for results (default: reports/)
- `--random-state INTEGER`: Random state for reproducibility

**Examples:**
```bash
# Basic backtest
quantfolio run-backtest --method combined --train-years 8 --test-years 2

# Custom parameters
quantfolio run-backtest --method black_litterman --train-years 6 --test-years 1 --rebalance quarterly

# With transaction costs
quantfolio run-backtest --transaction-costs '{"ETF":0.0005,"Large_Cap":0.001}' --max-weight 0.25

# Reproducible results
quantfolio run-backtest --random-state 42 --method combined
```

## Visualization Commands

### `plot-backtest`

Generate visualization plots from backtest results.

```bash
quantfolio plot-backtest [OPTIONS]
```

**Options:**
- `--performance-file TEXT`: Path to backtest performance CSV file [default: reports/backtest_performance.csv]
- `--weights-file TEXT`: Path to backtest weights CSV file
- `--metrics-file TEXT`: Path to backtest metrics JSON file
- `--output-dir TEXT`: Output directory for plots (default: reports/)
- `--type TEXT`: Type of plot: 'backtest', 'comparison', 'weights', 'metrics', or 'all' [default: all]

**Plot Types:**
- **`backtest`**: Backtest results plots (cumulative returns, Sharpe ratio, drawdown, volatility, turnover, transaction costs)
- **`comparison`**: Performance comparison charts (portfolio vs benchmark)
- **`weights`**: Weight evolution tracking over time
- **`metrics`**: Aggregate metrics visualization
- **`all`**: Generate all visualization types

**Examples:**
```bash
# Generate all plots
quantfolio plot-backtest --type all

# Generate only backtest results
quantfolio plot-backtest --type backtest

# Custom file paths
quantfolio plot-backtest --performance-file custom_performance.csv --weights-file custom_weights.csv

# Custom output directory
quantfolio plot-backtest --output-dir custom_plots/
```

## Command Output

### Data Files

All commands generate structured output files:

**Data Files:**
- `data/raw/`: Raw data from external sources
- `data/processed/`: Cleaned and processed data
- `notebooks/data/processed/`: Processed data for analysis

**Results Files:**
- `reports/`: Backtest results and optimization outputs
- `reports/figures/`: Generated plots and visualizations

**Common File Formats:**
- `.csv`: Tabular data
- `.json`: Configuration and metrics
- `.png`: Generated plots
- `.txt`: Summary reports

### Logging

All commands provide detailed logging:

```bash
# Verbose output
quantfolio fetch-data --start-date 2020-01-01

# Check logs for errors
quantfolio run-backtest 2>&1 | grep -i error
```

## Environment Variables

The CLI respects the following environment variables:

```bash
# API Keys
FRED_API_KEY=your_fred_api_key
NEWS_API_KEY=your_news_api_key

# Data Paths
DATA_DIR=data/
REPORTS_DIR=reports/

# Configuration
LOG_LEVEL=INFO
RANDOM_STATE=42
```

## Help and Support

### Getting Help

```bash
# General help
quantfolio --help

# Command-specific help
quantfolio fetch-data --help
quantfolio optimize-portfolio --help
quantfolio run-backtest --help
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
quantfolio fetch-data

# Check system status
quantfolio status

# Validate data
quantfolio validate-data
```

### Common Issues

1. **API Key Errors**: Check your `.env` file and API key validity
2. **Data Not Found**: Run `quantfolio fetch-data` first
3. **Memory Issues**: Reduce data size or increase system memory
4. **Permission Errors**: Check file permissions in data/ and reports/ directories

---

*For more detailed examples, see the [Tutorials](../tutorials/) section.*
