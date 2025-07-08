# Factor Timing

Factor timing is a sophisticated approach to dynamic portfolio management that adjusts asset allocations based on macroeconomic regimes and factor performance cycles.

## Overview

Factor timing combines:

- **Regime Detection**: Identifying market states using clustering algorithms
- **Factor Exposure**: Calculating rolling factor exposures for assets
- **Dynamic Views**: Creating time-varying Black-Litterman views
- **Signal Integration**: Combining multiple data sources for robust signals

## Theoretical Foundation

### Factor Models

The engine uses a multi-factor model:

$$R_i = \alpha_i + \sum_{j=1}^{k} \beta_{ij} F_j + \epsilon_i$$

Where:
- $R_i$ = Return of asset $i$
- $\alpha_i$ = Alpha (excess return)
- $\beta_{ij}$ = Factor exposure of asset $i$ to factor $j$
- $F_j$ = Factor return $j$
- $\epsilon_i$ = Idiosyncratic return

### Regime Detection

Market regimes are identified using Hidden Markov Models (HMM) or K-means clustering:

1. **Feature Engineering**: Create regime features from macro data
2. **Clustering**: Group similar market conditions
3. **Regime Assignment**: Assign each period to a regime
4. **Transition Matrix**: Model regime transitions

### Factor Timing Views

Dynamic views are created based on:

1. **Regime-Based Views**: Different views per market regime
2. **Factor Momentum**: Recent factor performance trends
3. **Macro Conditioning**: Economic indicator signals
4. **Sentiment Integration**: News-based sentiment signals

## Implementation in QuantFolio Engine

### Signal Generation Process

```python
# 1. Calculate rolling factor exposures
factor_exposures = calculate_rolling_exposures(returns, factors, lookback=60)

# 2. Detect market regimes
regimes = detect_regimes(macro_data, n_regimes=3)

# 3. Create factor timing views
views = create_factor_timing_views(factor_exposures, regimes, macro_data)

# 4. Generate Black-Litterman views
bl_views = generate_black_litterman_views(views, confidence=0.8)
```

### Factor Exposure Calculation

**Rolling Regression Method:**

```python
def calculate_rolling_exposures(returns, factors, lookback=60):
    """
    Calculate rolling factor exposures using rolling regression.

    Parameters:
    - returns: Asset returns (T x N)
    - factors: Factor returns (T x K)
    - lookback: Rolling window size in months

    Returns:
    - factor_exposures: Rolling exposures (T x N x K)
    """
    exposures = []

    for t in range(lookback, len(returns)):
        # Rolling window data
        window_returns = returns.iloc[t-lookback:t]
        window_factors = factors.iloc[t-lookback:t]

        # Calculate exposures for each asset
        asset_exposures = []
        for asset in returns.columns:
            # Linear regression: asset_return ~ factors
            model = LinearRegression()
            model.fit(window_factors, window_returns[asset])
            asset_exposures.append(model.coef_)

        exposures.append(asset_exposures)

    return np.array(exposures)
```

### Regime Detection

**K-means Clustering Method:**

```python
def detect_regimes(macro_data, n_regimes=3):
    """
    Detect market regimes using K-means clustering.

    Parameters:
    - macro_data: Macroeconomic indicators (T x M)
    - n_regimes: Number of regimes to detect

    Returns:
    - regimes: Regime labels for each period
    """
    # Standardize macro data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(macro_data)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    regimes = kmeans.fit_predict(scaled_data)

    return regimes
```

### Factor Timing Views

**Regime-Based View Generation:**

```python
def create_factor_timing_views(factor_exposures, regimes, macro_data):
    """
    Create factor timing views based on regime and macro conditions.

    Parameters:
    - factor_exposures: Rolling factor exposures
    - regimes: Market regime labels
    - macro_data: Macroeconomic indicators

    Returns:
    - views: Factor timing views for each period
    """
    views = []

    for t in range(len(factor_exposures)):
        current_regime = regimes[t]
        current_exposures = factor_exposures[t]
        current_macro = macro_data.iloc[t]

        # Regime-specific view adjustments
        regime_multiplier = get_regime_multiplier(current_regime)

        # Macro conditioning
        macro_signal = calculate_macro_signal(current_macro)

        # Combine signals
        view = current_exposures * regime_multiplier * macro_signal
        views.append(view)

    return np.array(views)
```

## Advanced Factor Timing Strategies

### 1. Multi-Factor Momentum

Combine multiple factor signals:

```python
def multi_factor_momentum(factor_returns, lookback=12):
    """
    Calculate multi-factor momentum signals.

    Parameters:
    - factor_returns: Historical factor returns
    - lookback: Momentum lookback period

    Returns:
    - momentum_signal: Combined momentum signal
    """
    # Calculate momentum for each factor
    momentum_signals = []

    for factor in factor_returns.columns:
        # Rolling momentum
        momentum = factor_returns[factor].rolling(lookback).mean()
        momentum_signals.append(momentum)

    # Combine signals (equal weight or custom weights)
    combined_momentum = np.mean(momentum_signals, axis=0)

    return combined_momentum
```

### 2. Regime-Dependent Factor Weights

Adjust factor importance based on regime:

```python
def regime_dependent_weights(regimes, factor_importance):
    """
    Calculate regime-dependent factor weights.

    Parameters:
    - regimes: Market regime labels
    - factor_importance: Base factor importance weights

    Returns:
    - adjusted_weights: Regime-adjusted factor weights
    """
    regime_multipliers = {
        0: [1.2, 0.8, 1.0],  # Growth regime
        1: [0.8, 1.2, 1.0],  # Value regime
        2: [1.0, 1.0, 1.2]   # Defensive regime
    }

    adjusted_weights = []

    for regime in regimes:
        multiplier = regime_multipliers[regime]
        adjusted_weight = factor_importance * multiplier
        adjusted_weights.append(adjusted_weight)

    return np.array(adjusted_weights)
```

### 3. Sentiment-Enhanced Factor Timing

Integrate news sentiment with factor signals:

```python
def sentiment_enhanced_views(factor_views, sentiment_data):
    """
    Enhance factor views with sentiment signals.

    Parameters:
    - factor_views: Base factor timing views
    - sentiment_data: News sentiment scores

    Returns:
    - enhanced_views: Sentiment-enhanced views
    """
    # Calculate sentiment signal
    sentiment_signal = calculate_sentiment_signal(sentiment_data)

    # Combine with factor views
    enhanced_views = factor_views * (1 + sentiment_signal * 0.1)

    return enhanced_views
```

## Practical Implementation

### CLI Usage

```bash
# Generate factor timing signals
quantfolio generate-signals --lookback 60 --regimes 3 --factor-method macro

# Use in optimization
quantfolio optimize-portfolio --method black_litterman --bl-auto

# Backtest with factor timing
quantfolio run-backtest --method combined --train-years 8 --test-years 2
```

### Configuration Options

**Factor Timing Parameters:**

```python
# In config.py
FACTOR_TIMING_CONFIG = {
    'lookback_period': 60,  # months
    'n_regimes': 3,
    'factor_method': 'macro',  # 'macro', 'fama_french', 'simple'
    'regime_detection': 'kmeans',  # 'kmeans', 'hmm'
    'view_strength': 1.5,
    'sentiment_weight': 0.1,
    'momentum_lookback': 12
}
```

### Custom Factor Timing

**Adding Custom Factors:**

```python
def custom_factor_timing(returns, macro_data, custom_factors):
    """
    Custom factor timing implementation.

    Parameters:
    - returns: Asset returns
    - macro_data: Macroeconomic data
    - custom_factors: Custom factor definitions

    Returns:
    - custom_views: Custom factor timing views
    """
    # Calculate custom factor exposures
    custom_exposures = calculate_custom_exposures(returns, custom_factors)

    # Detect regimes
    regimes = detect_regimes(macro_data)

    # Create custom views
    custom_views = create_custom_views(custom_exposures, regimes)

    return custom_views
```

## Performance Analysis

### Factor Timing Effectiveness

**Metrics to Monitor:**

1. **Factor Timing Alpha**: Excess return from factor timing
2. **Regime Hit Rate**: Accuracy of regime predictions
3. **View Consistency**: Stability of factor views over time
4. **Signal-to-Noise**: Quality of factor timing signals

### Backtesting Factor Timing

```bash
# Compare with and without factor timing
quantfolio run-backtest --method black_litterman --bl-auto
quantfolio run-backtest --method black_litterman --no-factor-timing

# Analyze factor timing contribution
quantfolio analyze-factor-timing --performance-file reports/backtest_performance.csv
```

## Best Practices

### 1. Data Quality

- **Use sufficient history**: At least 10+ years for regime detection
- **Factor stability**: Ensure factors are economically meaningful
- **Regime persistence**: Check regime stability over time

### 2. Model Robustness

- **Out-of-sample testing**: Validate on unseen data
- **Parameter stability**: Test sensitivity to parameters
- **Regime validation**: Verify regime economic interpretation

### 3. Implementation

- **Transaction costs**: Account for rebalancing costs
- **Implementation lag**: Consider signal delay
- **Risk management**: Monitor factor timing risk

### 4. Common Pitfalls

**Overfitting:**
- Avoid excessive parameter tuning
- Use cross-validation for regime detection
- Test across different market cycles

**Regime Instability:**
- Ensure sufficient data for regime detection
- Use robust clustering algorithms
- Validate regime economic meaning

**Signal Decay:**
- Monitor factor timing effectiveness over time
- Update factor models periodically
- Consider regime transitions

## Advanced Topics

### 1. Dynamic Factor Models

```python
def dynamic_factor_model(returns, factors, regime_data):
    """
    Dynamic factor model with regime-dependent parameters.
    """
    # Regime-dependent factor loadings
    regime_loadings = estimate_regime_loadings(returns, factors, regime_data)

    # Dynamic factor timing
    dynamic_views = create_dynamic_views(regime_loadings)

    return dynamic_views
```

### 2. Machine Learning Enhancement

```python
def ml_enhanced_factor_timing(returns, factors, macro_data):
    """
    Machine learning enhanced factor timing.
    """
    # Feature engineering
    features = create_factor_features(returns, factors, macro_data)

    # ML model for factor timing
    model = RandomForestRegressor()
    model.fit(features, future_returns)

    # Generate ML-enhanced views
    ml_views = model.predict(features)

    return ml_views
```

### 3. Real-Time Factor Timing

```python
def real_time_factor_timing(live_data, factor_model):
    """
    Real-time factor timing implementation.
    """
    # Update factor exposures
    current_exposures = update_exposures(live_data)

    # Detect current regime
    current_regime = detect_current_regime(live_data)

    # Generate real-time views
    real_time_views = generate_real_time_views(current_exposures, current_regime)

    return real_time_views
```

## Next Steps

After understanding factor timing:

1. **Explore Related Topics**:
   - Black-Litterman for Bayesian optimization
   - Monte Carlo for scenario analysis
   - Risk Attribution for risk decomposition

2. **Advanced Applications**:
   - Multi-asset factor timing
   - Currency factor timing
   - Alternative data integration

3. **Production Implementation**:
   - Real-time factor timing systems
   - Automated regime detection
   - Factor timing monitoring

---

*Ready to explore more advanced topics? Check out the other documentation sections for sophisticated strategies!*
