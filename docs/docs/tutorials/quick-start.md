# Quick Start Tutorial

This tutorial will guide you through your first complete analysis with QuantFolio Engine, from data fetching to backtesting and visualization.

## Prerequisites

Before starting, ensure you have:

1. **QuantFolio Engine installed** (see [Installation Guide](../user-guide/installation.md))
2. **API keys configured** (FRED API and News API)
3. **Conda environment activated**: `conda activate quantfolio-engine`

## Step 1: Fetch Data

Start by fetching historical financial data:

```bash
# Fetch data from 2010 onwards
quantfolio fetch-data --start-date 2010-01-01

# Check what data was fetched
quantfolio list-data
```

**Expected Output:**
```
✅ Successfully fetched returns for 12 assets
✅ Successfully fetched 8 macro indicators
✅ Successfully fetched sentiment for 5 entities/topics
```

## Step 2: Generate Factor Timing Signals

Create factor timing signals and regime detection:

```bash
# Generate signals with default parameters
quantfolio generate-signals

# Or customize the parameters
quantfolio generate-signals --lookback 60 --regimes 3 --factor-method macro
```

**What This Does:**
- Calculates rolling factor exposures
- Detects market regimes using clustering
- Creates factor timing views for optimization

## Step 3: Run Portfolio Optimization

Optimize a portfolio using the combined method:

```bash
# Basic optimization
quantfolio optimize-portfolio --method combined --max-weight 0.3 --min-weight 0.05

# With more constraints
quantfolio optimize-portfolio \
  --method combined \
  --max-weight 0.25 \
  --min-weight 0.05 \
  --max-volatility 0.15 \
  --risk-free-rate 0.045
```

**Expected Output:**
```
Portfolio Optimization Results:
- Optimal Weights: [0.25, 0.20, 0.15, ...]
- Expected Return: 8.5%
- Expected Volatility: 12.3%
- Sharpe Ratio: 0.69
- Max Drawdown: -8.2%
```

## Step 4: Run Walk-Forward Backtesting

Validate your strategy with comprehensive backtesting:

```bash
# Basic backtest
quantfolio run-backtest --method combined --train-years 8 --test-years 2

# With transaction costs
quantfolio run-backtest \
  --method combined \
  --train-years 8 \
  --test-years 2 \
  --rebalance monthly \
  --transaction-costs '{"ETF":0.0005,"Large_Cap":0.001}' \
  --max-weight 0.3 \
  --min-weight 0.05 \
  --random-state 42
```

**Expected Output:**
```
Backtest Results:
- Total periods: 90
- Average total return: 1.9%
- Average Sharpe ratio: 0.850
- Average Sortino ratio: 0.349
- Hit ratio: 60%
- Total transaction costs: 0.35%
- Net total return (after costs): 1.9%
```

## Step 5: Generate Visualizations

Create comprehensive plots of your results:

```bash
# Generate all visualization types
quantfolio plot-backtest --type all

# Or generate specific plots
quantfolio plot-backtest --type backtest
quantfolio plot-backtest --type comparison
```

**Generated Files:**
- `reports/backtest_results.png`: Comprehensive backtest plots
- `reports/performance_comparison.png`: Portfolio vs benchmark
- `reports/return_distribution.png`: Return distribution histogram
- `reports/weight_evolution.png`: Weight changes over time
- `reports/aggregate_metrics.png`: Summary metrics

## Step 6: Analyze Results

Check the generated files in the `reports/` directory:

```bash
# List generated files
ls -la reports/

# View summary report
cat reports/backtest_summary.txt
```

## Complete Example

Here's a complete workflow you can run:

```bash
#!/bin/bash

echo "🚀 Starting QuantFolio Engine Analysis..."

# Step 1: Fetch data
echo "📊 Fetching data..."
quantfolio fetch-data --start-date 2010-01-01

# Step 2: Generate signals
echo "🎯 Generating factor timing signals..."
quantfolio generate-signals --lookback 60 --regimes 3

# Step 3: Run backtest
echo "📈 Running walk-forward backtest..."
quantfolio run-backtest \
  --method combined \
  --train-years 8 \
  --test-years 2 \
  --rebalance monthly \
  --max-weight 0.3 \
  --min-weight 0.05 \
  --max-volatility 0.15 \
  --risk-free-rate 0.045 \
  --random-state 42

# Step 4: Generate plots
echo "📊 Creating visualizations..."
quantfolio plot-backtest --type all

echo "✅ Analysis complete! Check reports/ directory for results."
```

## Understanding the Results

### Key Metrics to Focus On:

1. **Sharpe Ratio**: Risk-adjusted returns (higher is better)
2. **Sortino Ratio**: Downside risk-adjusted returns
3. **Hit Ratio**: Percentage of periods with positive returns
4. **Max Drawdown**: Worst peak-to-trough decline
5. **Transaction Costs**: Impact of trading on performance
6. **Excess Return**: Performance vs benchmark

### What the Plots Show:

- **Cumulative Returns**: Overall performance over time
- **Rolling Sharpe**: Risk-adjusted performance trends
- **Drawdown**: Risk periods and recovery
- **Weight Evolution**: How allocations changed
- **Return Distribution**: Statistical properties of returns

## Next Steps

After completing this tutorial:

1. **Explore Advanced Features**:
   - Try different optimization methods (`black_litterman`, `monte_carlo`)
   - Experiment with different constraints
   - Test various rebalancing frequencies

2. **Customize Your Analysis**:
   - Add custom transaction costs
   - Modify factor timing parameters
   - Include additional assets

3. **Deep Dive into Documentation**:
   - [Backtesting Tutorial](backtesting.md) for detailed validation
   - [Factor Timing](../advanced/factor-timing.md) for advanced strategies

## Troubleshooting

### Common Issues:

**"No data found" error:**
```bash
# Check if data exists
quantfolio list-data

# Re-fetch data if needed
quantfolio fetch-data --start-date 2010-01-01
```

**"API key not found" error:**
```bash
# Check configuration
quantfolio status

# Verify .env file exists and has valid keys
cat .env
```

**"Insufficient data" error:**
```bash
# Fetch more historical data
quantfolio fetch-data --start-date 2005-01-01

# Or reduce training window
quantfolio run-backtest --train-years 5 --test-years 1
```

### Getting Help:

```bash
# Check command help
quantfolio --help
quantfolio run-backtest --help

# Validate data quality
quantfolio validate-data

# Check system status
quantfolio status
```

---

*Ready for more? Explore the [Advanced Topics](../advanced/) for deep dives into specific features!*
