# Getting Started with QuantFolio Engine

## Overview

QuantFolio Engine is a quantitative portfolio optimization system designed for institutional asset management. It combines macroeconomic context, factor-timing models, and LLM-driven sentiment signals into a dynamic portfolio optimizer and risk explainer.

## Prerequisites

- Python 3.11 or higher
- Git
- Conda or virtual environment manager

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/quantfolio-engine.git
cd quantfolio-engine
```

### 2. Create Environment

```bash
# Using conda
make create_environment
conda activate quantfolio-engine

# Or using venv
python -m venv quantfolio-engine
source quantfolio-engine/bin/activate  # On Windows: quantfolio-engine\Scripts\activate
```

### 3. Install Dependencies

```bash
make requirements
```

### 4. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Federal Reserve Economic Data (FRED) API
FRED_API_KEY=your_actual_fred_api_key

# News API for sentiment analysis
NEWS_API_KEY=your_actual_news_api_key

# Alpha Vantage API (optional)
ALPHA_VANTAGE_KEY=your_actual_alpha_vantage_key
```

**API Key Sources:**
- **FRED API**: Free key from [Federal Reserve Economic Data](https://fred.stlouisfed.org/docs/api/api_key.html)
- **News API**: Free tier available at [NewsAPI.org](https://newsapi.org/)
- **Alpha Vantage**: Free tier available at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

## Quick Start

### 1. Load Data

```bash
quantfolio fetch-data --start-date 2010-01-01
```

This will download:
- Asset returns from YFinance
- Macroeconomic indicators from FRED
- Generate synthetic sentiment signals

### 2. Generate Factor Timing Signals

```bash
quantfolio generate-signals
```

This calculates:
- Factor exposures (Value, Growth, Momentum, Quality, Size)
- Macroeconomic regime classification
- Factor timing signals

### 3. Run Portfolio Optimization

```bash
quantfolio optimize-portfolio --method combined
```

This performs:
- Black-Litterman optimization
- Factor and sentiment view integration
- Constraint optimization

### 4. Run Backtesting

```bash
quantfolio run-backtest --method combined --train-years 8 --test-years 2
```

This performs:
- Walk-forward backtesting with transaction costs
- Performance analysis and risk metrics
- Benchmark comparison

### 5. Generate Plots and Reports

```bash
quantfolio plot-backtest --type all
```

This creates:
- Performance charts and risk analysis
- Factor attribution plots
- Transaction cost analysis

## Configuration

### Asset Universe

The default asset universe includes:

- **Equity ETFs**: SPY, QQQ, IWM, EFA, EEM, VTI, VEA, VWO, VTV, VUG
- **Sector ETFs**: XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE
- **Fixed Income**: TLT, IEF, SHY, LQD, HYG, EMB, BND, AGG

### Model Parameters

Key parameters can be adjusted in `quantfolio_engine/config.py`:

- **Factor Timing**: Lookback period, regime clusters, rebalancing frequency
- **Black-Litterman**: Risk aversion, confidence levels, prior uncertainty
- **Optimization**: Weight constraints, target volatility, tracking error limits

### Data Sources

The system integrates multiple data sources:

1. **Asset Data**: YFinance for real-time market data
2. **Macro Data**: FRED for economic indicators
3. **Sentiment**: News API or synthetic signals for sentiment analysis

## Usage Examples

### Basic Portfolio Optimization

```python
from quantfolio_engine.data.data_loader import DataLoader
from quantfolio_engine.optimizer.black_litterman import BlackLittermanOptimizer

# Load data
loader = DataLoader()
data = loader.load_all_data()

# Optimize portfolio
optimizer = BlackLittermanOptimizer()
equilibrium_returns = optimizer.calculate_equilibrium_returns(data['asset_returns'])
optimal_weights, stats = optimizer.optimize_portfolio()

print(f"Expected Return: {stats['expected_return']:.2%}")
print(f"Volatility: {stats['volatility']:.2%}")
```

### Factor Timing Analysis

```python
from quantfolio_engine.signals.factor_timing import FactorTiming

# Calculate factor exposures
factor_timing = FactorTiming()
exposures = factor_timing.calculate_factor_exposures(data['asset_returns'])

# Classify regimes
regime_labels, model = factor_timing.classify_regimes(data['macro_indicators'])

# Generate signals
signals = factor_timing.generate_factor_timing_signals(exposures, regime_labels, data['macro_indicators'])
```

### Risk Attribution

```python
from quantfolio_engine.attribution.risk_attribution import RiskAttribution

# Perform attribution
attribution = RiskAttribution()
results = attribution.comprehensive_attribution(
    portfolio_returns=data['asset_returns'].mean(),
    benchmark_returns=data['asset_returns'].mean(),
    portfolio_weights=optimal_weights,
    benchmark_weights=benchmark_weights
)
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in `.env`
2. **Data Loading Failures**: Check internet connection and API rate limits
3. **Optimization Failures**: Adjust constraint parameters in config
4. **Memory Issues**: Reduce asset universe size or data history

### Getting Help

- Check the logs in `logs/` directory
- Run `quantfolio --help` to see available commands
- Review error messages in the CLI output

## Next Steps

- Customize asset universe for your needs
- Integrate additional data sources
- Implement custom factor models
- Add backtesting capabilities
- Deploy to production environment

## Contributing

See [Contributing](development/contributing.md) for development guidelines.
