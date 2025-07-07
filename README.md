# QuantFolio Engine

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Smart Portfolio Construction Using Factor Timing and Multi-Source Signal Integration**

A quantitative portfolio optimization engine designed for institutional asset management, combining macroeconomic context, factor-timing models, and LLM-driven sentiment signals into a dynamic portfolio optimizer and risk explainer.

## 🎯 Target Audience
- Institutional quantitative teams
- Hedge fund portfolio managers
- Smart-beta product managers
- Asset management firms (AllianceBernstein, BlackRock, etc.)

## 🚀 Development Roadmap

### 🔹 Phase 1: Data Ingestion & Preprocessing
- [x] Fetch historical prices for 30–100 assets (e.g. S&P 500 subset, ETFs)
- [x] Retrieve 5–10 macro indicators from FRED or equivalent
- [x] Load static or dynamic sentiment scores for corresponding timeframes
    - ⚠️ Note: Sentiment score is currently placeholder/random. Replace with live data from a real API (e.g., News API, RavenPack) in this phase.
- [x] Normalize all data (e.g., Z-scores, rolling %change)
    - Normalization implemented for returns, macro, and sentiment data.

### 🔹 Phase 2: Factor Timing Signal Builder
- [x] Calculate factor exposures using regression (Fama-French 3 or 5 factor model)
- [x] Detect factor regimes using:
  - [x] Rolling means / volatilities
  - [x] Clustering methods (e.g. k-means, DBSCAN)
  - [x] HMMs for regime probabilities

### 🔹 Phase 3: Portfolio Optimization Engine
- [x] Implement Black-Litterman:
  - [x] Use empirical covariance matrix
  - [x] Encode views via factor-timing outputs (e.g. bullish on momentum)
  - [x] Adjust priors based on sentiment scores
- [x] Monte Carlo alternative:
  - [x] Simulate 1000+ future paths under different macro regimes
  - [x] Constrain for max drawdown, volatility, sector allocation
  - [ ] **Future Enhancement: Macro Conditioning**
    - [ ] Implement copula-based macro conditioning for Monte Carlo scenarios
    - [ ] Add regime-specific volatility adjustments based on macro indicators
    - [ ] Incorporate macro factor loading adjustments in scenario generation
    - [ ] Develop dynamic correlation structure based on economic conditions

### 🔹 Phase 3.5: Walk-Forward Back-Testing & Validation
- [ ] **Phase 1: Foundation**
  - [ ] Data validator for sufficient history and quality
  - [ ] Basic walk-forward framework with configurable train/test windows
  - [ ] Simple transaction cost model (fixed bps by asset type)
  - [ ] Core performance metrics (Sharpe, Sortino, MaxDD, Calmar)
- [ ] **Phase 2: Benchmarks**
  - [ ] Static 60/40 portfolio benchmark
  - [ ] Rebalanced 60/40 (monthly/quarterly) for realistic competition
  - [ ] Equal-weighted portfolio baseline
  - [ ] Information ratio calculation vs benchmarks
- [ ] **Phase 3: Advanced Features**
  - [ ] Parameter sensitivity analysis (lookback_window, num_regimes, tau, view_confidence, rebalance_frequency)
  - [ ] Advanced metrics (Omega ratio, skewness, hit ratio, VaR/CVaR)
  - [ ] Factor-tilted benchmarks for factor timing validation
  - [ ] Turnover analysis and optimization
- [ ] **Data Strategy**
  - [ ] 8 years training (2015-2023) + 2 years testing (2023-2025)
  - [ ] Data quality validation before backtesting
  - [ ] Synthetic data padding if needed (<8 years real data)
  - [ ] Validate data continuity for rolling calculations
- [ ] **Transaction Cost Model**
  - [ ] Tiered costs: ETFs (5 bps), Large-Cap (10 bps), Small-Cap (20 bps), International (25 bps)
  - [ ] Include benchmark transaction costs to prevent false alpha
  - [ ] Turnover × cost per unit for each rebalance
- [ ] **Performance Metrics**
  - [ ] Core: Sharpe, Sortino, Max Drawdown, Calmar ratios
  - [ ] Factor timing: Information ratio, hit ratio, skewness
  - [ ] Risk: VaR/CVaR, turnover ratio, regime-specific performance

### 🔹 Phase 4: Risk Attribution Framework
- [ ] Use marginal contribution to risk (MCR) or Brinson model
- [ ] Breakdown:
  - [ ] Asset-level risk
  - [ ] Factor contributions
  - [ ] Macro-linked variance (e.g., via PCA loadings)
- [ ] **Performance Analytics**
  - [ ] Use equity curve & weight history from Phase 3.5
  - [ ] Factor attribution analysis
  - [ ] Regime-specific performance breakdown

### 🔹 Phase 5: UI & Deployment
- [ ] Deploy via Streamlit:
  - [ ] Input views, rebalancing frequency
  - [ ] Output recommended portfolio, expected return, CVaR
  - [ ] Plot attribution summaries

## 🚀 MVP Features

### Data Pipeline
- **Asset-level historical returns** (stocks, ETFs, sectors)
- **Macroeconomic indicators** (inflation, unemployment, GDP growth)
- **LLM-based sentiment signals** (FinNews-LLM scores)

### Factor Timing Layer
- Rolling factor exposures (Value, Growth, Momentum)
- Factor regime classification (k-means clustering/HMM)

### Optimization Engine
- **Black-Litterman model** with customizable priors from:
  - User-defined views
  - Factor-timing signals
  - Sentiment-based scores
- **Monte Carlo simulation** with constraints and risk targeting

### Risk Attribution Module
- **Brinson attribution** or **Shapley value decomposition**
- Return contribution by asset and factor
- Risk contribution (volatility, beta) by macro and style drivers

### Dashboard
- **Streamlit/Plotly Dash** interface for:
  - Factor regime visualization
  - Optimal weights and rebalancing suggestions
  - Return decomposition and risk attribution

## 🛠️ Technology Stack

### Core Libraries
- **Python 3.11+**
- **Data Science**: pandas, numpy, scipy, statsmodels, scikit-learn
- **Optimization**: cvxpy
- **Data Sources**: yfinance, fredapi, newsapi
- **Visualization**: matplotlib, plotly, seaborn, streamlit
- **Optional**: quantlib, alphalens, pyfolio

### Development Tools
- **Environment**: conda/venv with .env configuration
- **Testing**: pytest
- **Code Quality**: black, flake8, isort
- **Documentation**: MkDocs

## 📁 Project Organization

```
quantfolio-engine/
├── LICENSE                 <- Open-source license
├── Makefile               <- Convenience commands
├── README.md              <- Project overview
├── pyproject.toml         <- Package configuration
├── setup.cfg              <- Development tools config
├── data/                  <- Data storage
│   ├── external/          <- Third-party data sources
│   ├── interim/           <- Intermediate transformations
│   ├── processed/         <- Final canonical datasets
│   └── raw/               <- Original immutable data
├── docs/                  <- Documentation
├── models/                <- Trained models and outputs
├── notebooks/             <- Jupyter notebooks for EDA
├── quantfolio_engine/     <- Source code
│   ├── __init__.py
│   ├── config.py          <- Configuration management
│   ├── data/              <- Data pipeline modules
│   ├── signals/           <- Factor timing & sentiment signals
│   ├── optimizer/         <- Black-Litterman & MC simulation
│   ├── attribution/       <- Risk & return attribution
│   ├── dashboard/         <- Streamlit interface
│   └── utils/             <- Utility functions
├── reports/               <- Generated analysis
│   └── figures/           <- Visualizations
├── tests/                 <- Unit tests
└── logs/                  <- Model run history
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/quantfolio-engine.git
cd quantfolio-engine

# Create environment
make create_environment
conda activate quantfolio-engine

# Install dependencies
make requirements
```

### Configuration
1. Copy `.env.example` to `.env`
2. Add your API keys for data sources:
   - FRED API (Federal Reserve Economic Data)
   - News API or FinNews-LLM outputs
   - YFinance (optional, for real-time data)

### Basic Usage
```bash
# Process data pipeline
make data

# Generate features and signals
python quantfolio_engine/signals/main.py

# Run portfolio optimization
python quantfolio_engine/optimizer/main.py

# Launch dashboard
streamlit run quantfolio_engine/dashboard/app.py
```

## 📊 Key Concepts

### Factor Timing
Dynamic adjustment of portfolio weights based on macroeconomic regimes and factor performance cycles.

### Black-Litterman Model
Bayesian approach combining market equilibrium with investor views and quantitative signals.

### Risk Attribution
Decomposition of portfolio risk and return into attributable components for transparency and client communication.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔮 Future Implementation Plans

#### Planned Macro Conditioning Features

##### 1. Copula-Based Macro Conditioning
```python
# Future implementation in monte_carlo.py
def _apply_macro_copula_conditioning(
    self,
    scenarios: np.ndarray,
    macro_data: pd.DataFrame,
    current_macro_state: pd.Series
) -> np.ndarray:
    """
    Apply copula-based macro conditioning to Monte Carlo scenarios.
    - Use copula models to capture macro-asset return dependencies
    - Shift scenario distributions based on current macro environment
    - Adjust tail risk based on macro stress indicators
    """
```

##### 2. Dynamic Volatility Adjustment
```python
def _calculate_macro_volatility_adjustment(
    self,
    macro_indicators: pd.Series,
    base_volatility: float
) -> float:
    """
    Adjust volatility based on macro environment.
    - High inflation → Increase volatility
    - Low unemployment → Decrease volatility
    - High VIX → Increase volatility
    - Economic stress → Increase volatility
    """
```

##### 3. Macro Factor Loading Adjustments
```python
def _adjust_factor_loadings_for_macro(
    self,
    factor_exposures: pd.DataFrame,
    macro_environment: pd.Series
) -> pd.DataFrame:
    """
    Adjust factor loadings based on macro environment.
    - Recession: Increase defensive factor weights
    - Expansion: Increase growth factor weights
    - High rates: Increase value factor weights
    - Low rates: Increase momentum factor weights
    """
```

##### 4. Dynamic Correlation Structure
```python
def _calculate_macro_conditioned_correlation(
    self,
    base_correlation: np.ndarray,
    macro_stress_level: float,
    economic_regime: str
) -> np.ndarray:
    """
    Adjust correlation matrix based on macro conditions.
    - Crisis periods: Increase correlations (flight to quality)
    - Expansion periods: Decrease correlations (diversification)
    - High stress: Increase defensive asset correlations
    """
```

#### Implementation Priority
1. **Phase 1**: Implement basic macro volatility adjustments
2. **Phase 2**: Add copula-based scenario conditioning
3. **Phase 3**: Develop dynamic factor loading adjustments
4. **Phase 4**: Implement full dynamic correlation structure

#### Expected Benefits
- **Enhanced Risk Modeling**: More accurate tail risk estimation
- **Better Regime Transitions**: Smoother transitions between market states
- **Improved Diversification**: Dynamic correlation adjustments
- **Macro-Aware Optimization**: Portfolio weights that respond to economic conditions

#### Technical Considerations
- **Performance**: Copula calculations can be computationally intensive
- **Data Requirements**: Need high-quality macro data with sufficient history
- **Model Complexity**: Balance between sophistication and interpretability
- **Backtesting**: Extensive validation required for macro conditioning features

---

## 📄 License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template
- Inspired by institutional quantitative workflows
- Leverages modern Python data science ecosystem

## Phase 3: Black-Litterman Troubleshooting & Tuning

| Symptom                                                                   | Why it matters                                                                                      | Quick checks & tweaks                                                                                                                                  |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Sharpe only ≈ 0.18** after views                                        | That's still thin excess return for 10 % vol.                                                       | ① Bump `lambda_range` to, say, `(1, 10)` and finer grid. ② Try lower `γ` (grand-view blend) — it drags π toward the mean and can dilute richer assets. |
| **View strength fixed at 3× multiplier**                                  | May be too timid once λ rises.                                                                      | Pass `view_strength=self.bl_view_strength` from engine into `create_factor_timing_views`. Try 2 – 4 and compare.                                      |
| **Regime = 0 every time**                                                 | Your HMM may be stuck in one state or the dates aren't lining up, so multiplier 2.0 is always used. | Inspect the last row of `factor_regimes` — if it never changes you're not getting regime diversification.                                              |
| **Weights hit the hard cap 20 %** (TLT, GLD, WMT)                         | Caps are binding. Portfolio could want >20 % in other assets but can't.                             | Decide if 20 % is a design choice or temporary.                                                                                                        |
| **VaR (95 %) –3.6 % monthly**                                             | Reasonable, but check if that's dominated by one asset class (GLD? TLT?).                           | Run `analyze_portfolio_risk()` to see asset-level contributions.                                                                                       |
| **Warnings about "no common dates between factor exposures and regimes"** | Means views are built without regime context when those indices diverge.                            | Align dates earlier or fill forward regime labels.                                                                                                     |
