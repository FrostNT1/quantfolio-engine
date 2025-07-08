# Backtesting Tutorial

This tutorial covers walk-forward backtesting in QuantFolio Engine, including data validation, performance metrics, and best practices.

## Overview

Walk-forward backtesting is a robust validation framework that:

- **Simulates real-world conditions** with rolling training windows
- **Accounts for transaction costs** and turnover
- **Provides comprehensive metrics** for strategy evaluation
- **Compares against benchmarks** for relative performance
- **Validates data quality** before testing

## Walk-Forward Methodology

### How It Works

1. **Training Window**: Use historical data to train the model
2. **Testing Window**: Apply the model to out-of-sample data
3. **Roll Forward**: Move both windows forward in time
4. **Aggregate Results**: Combine all testing periods for final metrics

### Example Timeline

```
Training: 2010-2017 → Test: 2018-2019
Training: 2012-2019 → Test: 2020-2021
Training: 2014-2021 → Test: 2022-2023
```

## Step-by-Step Backtesting

### Step 1: Data Preparation

Ensure you have sufficient data:

```bash
# Fetch comprehensive historical data
quantfolio fetch-data --start-date 2010-01-01

# Validate data quality
quantfolio validate-data

# Generate factor timing signals
quantfolio generate-signals --lookback 60 --regimes 3
```

### Step 2: Basic Backtest

Run a simple backtest with default parameters:

```bash
quantfolio run-backtest \
  --method combined \
  --train-years 8 \
  --test-years 2 \
  --rebalance monthly
```

**Parameters Explained:**
- `--method combined`: Uses both Black-Litterman and Monte Carlo
- `--train-years 8`: 8 years of training data
- `--test-years 2`: 2 years of out-of-sample testing
- `--rebalance monthly`: Rebalance portfolio monthly

### Step 3: Advanced Backtest

Add transaction costs and custom constraints:

```bash
quantfolio run-backtest \
  --method combined \
  --train-years 8 \
  --test-years 2 \
  --rebalance monthly \
  --max-weight 0.25 \
  --min-weight 0.05 \
  --max-volatility 0.15 \
  --risk-free-rate 0.045 \
  --transaction-costs '{"ETF":0.0005,"Large_Cap":0.001,"Small_Cap":0.002}' \
  --random-state 42
```

### Step 4: Compare Methods

Test different optimization approaches:

```bash
# Black-Litterman only
quantfolio run-backtest --method black_litterman --train-years 8 --test-years 2

# Monte Carlo only
quantfolio run-backtest --method monte_carlo --train-years 8 --test-years 2

# Combined approach
quantfolio run-backtest --method combined --train-years 8 --test-years 2
```

## Understanding the Results

### Key Performance Metrics

**Return Metrics:**
- **Total Return**: Cumulative performance over the testing period
- **Annualized Return**: Average annual performance
- **Excess Return**: Performance vs benchmark

**Risk Metrics:**
- **Volatility**: Standard deviation of returns
- **Max Drawdown**: Worst peak-to-trough decline
- **VaR/CVaR**: Value at Risk and Conditional VaR

**Risk-Adjusted Metrics:**
- **Sharpe Ratio**: Return per unit of risk
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: Return per unit of max drawdown

**Trading Metrics:**
- **Hit Ratio**: Percentage of positive periods
- **Turnover**: Portfolio rebalancing activity
- **Transaction Costs**: Impact of trading on performance

### Example Output

```
Backtest Results Summary:
┌─────────────────────┬─────────────┬─────────────┐
│ Metric              │ Portfolio   │ Benchmark   │
├─────────────────────┼─────────────┼─────────────┤
│ Total Return        │ 45.2%       │ 38.1%       │
│ Annualized Return   │ 8.9%        │ 7.2%        │
│ Volatility          │ 12.3%       │ 14.1%       │
│ Sharpe Ratio        │ 0.72        │ 0.51        │
│ Max Drawdown        │ -8.2%       │ -12.5%      │
│ Hit Ratio           │ 62%          │ 58%         │
│ Turnover            │ 15.3%        │ 5.2%        │
│ Transaction Costs   │ 0.35%        │ 0.12%       │
└─────────────────────┴─────────────┴─────────────┘
```

## Advanced Backtesting Scenarios

### Scenario 1: Different Rebalancing Frequencies

```bash
# Monthly rebalancing
quantfolio run-backtest --rebalance monthly --train-years 8 --test-years 2

# Quarterly rebalancing
quantfolio run-backtest --rebalance quarterly --train-years 8 --test-years 2

# Annual rebalancing
quantfolio run-backtest --rebalance annual --train-years 8 --test-years 2
```

### Scenario 2: Various Training Windows

```bash
# Short training window
quantfolio run-backtest --train-years 5 --test-years 1

# Medium training window
quantfolio run-backtest --train-years 8 --test-years 2

# Long training window
quantfolio run-backtest --train-years 10 --test-years 3
```

### Scenario 3: Custom Transaction Costs

```bash
# Conservative costs
quantfolio run-backtest \
  --transaction-costs '{"ETF":0.0003,"Large_Cap":0.0008,"Small_Cap":0.0015}'

# Realistic costs
quantfolio run-backtest \
  --transaction-costs '{"ETF":0.0005,"Large_Cap":0.001,"Small_Cap":0.002}'

# High costs
quantfolio run-backtest \
  --transaction-costs '{"ETF":0.001,"Large_Cap":0.002,"Small_Cap":0.005}'
```

### Scenario 4: Different Constraints

```bash
# Conservative constraints
quantfolio run-backtest \
  --max-weight 0.2 \
  --min-weight 0.05 \
  --max-volatility 0.12

# Aggressive constraints
quantfolio run-backtest \
  --max-weight 0.4 \
  --min-weight 0.02 \
  --max-volatility 0.18
```

## Data Validation

### Pre-Backtest Validation

The system automatically validates data before backtesting:

```bash
# Manual validation
quantfolio validate-data
```

**Validation Checks:**
- Data completeness and continuity
- Missing value thresholds
- Extreme value detection
- Date alignment across datasets
- Factor data format consistency

### Common Validation Issues

**Insufficient Data:**
```bash
# Error: "Insufficient data for backtesting"
# Solution: Fetch more historical data
quantfolio fetch-data --start-date 2005-01-01
```

**Data Quality Issues:**
```bash
# Error: "Data validation failed"
# Solution: Check data quality
quantfolio validate-data
quantfolio clean-data
```

## Performance Analysis

### Generate Visualizations

```bash
# Create comprehensive plots
quantfolio plot-backtest --type all

# Specific plot types
quantfolio plot-backtest --type backtest
quantfolio plot-backtest --type comparison
quantfolio plot-backtest --type weights
quantfolio plot-backtest --type metrics
```

### Analyze Results Files

Check the generated files in `reports/`:

```bash
# Performance data
head reports/backtest_performance.csv

# Weight evolution
head reports/backtest_weights.csv

# Summary metrics
cat reports/backtest_metrics.json
```

## Best Practices

### 1. Data Quality

- **Use sufficient history**: At least 10+ years for robust testing
- **Validate data**: Check for missing values and outliers
- **Ensure consistency**: Align dates across all datasets

### 2. Testing Parameters

- **Realistic constraints**: Use practical weight limits
- **Appropriate costs**: Include realistic transaction costs
- **Multiple scenarios**: Test various parameter combinations

### 3. Interpretation

- **Focus on risk-adjusted returns**: Sharpe/Sortino ratios
- **Consider transaction costs**: Net performance matters
- **Compare to benchmarks**: Relative performance is key
- **Check robustness**: Test across different time periods

### 4. Common Pitfalls

**Overfitting:**
- Avoid excessive parameter tuning
- Use out-of-sample testing
- Test across different market regimes

**Data Mining:**
- Don't cherry-pick time periods
- Use consistent methodologies
- Document all assumptions

**Transaction Costs:**
- Include realistic costs
- Consider market impact
- Account for bid-ask spreads

## Troubleshooting

### Common Errors

**"Insufficient data for training":**
```bash
# Increase training window or fetch more data
quantfolio fetch-data --start-date 2005-01-01
quantfolio run-backtest --train-years 6 --test-years 1
```

**"Optimization failed":**
```bash
# Relax constraints
quantfolio run-backtest --max-weight 0.4 --min-weight 0.02

# Check data quality
quantfolio validate-data
```

**"Data validation failed":**
```bash
# Clean and validate data
quantfolio clean-data
quantfolio validate-data

# Regenerate signals
quantfolio generate-signals
```

### Debug Mode

Enable detailed logging:

```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Run backtest with full output
quantfolio run-backtest --method combined 2>&1 | tee backtest.log
```

## Next Steps

After mastering backtesting:

1. **Explore Advanced Features**:
   - [Factor Timing](../advanced/factor-timing.md) for regime detection
   - Black-Litterman for Bayesian optimization
   - Risk Attribution for risk decomposition

2. **Customize Your Analysis**:
   - Add custom benchmarks
   - Implement custom constraints
   - Create custom performance metrics

3. **Production Deployment**:
   - Set up automated backtesting
   - Implement monitoring and alerts
   - Create reporting dashboards

---

*Ready to dive deeper? Check out the [Advanced Topics](../advanced/) for sophisticated strategies!*
