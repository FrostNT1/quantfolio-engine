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
- [ ] Calculate factor exposures using regression (Fama-French 3 or 5 factor model)
- [ ] Detect factor regimes using:
  - [ ] Rolling means / volatilities
  - [ ] Clustering methods (e.g. k-means, DBSCAN)
  - [ ] HMMs for regime probabilities

### 🔹 Phase 3: Portfolio Optimization Engine
- [ ] Implement Black-Litterman:
  - [ ] Use empirical covariance matrix
  - [ ] Encode views via factor-timing outputs (e.g. bullish on momentum)
  - [ ] Adjust priors based on sentiment scores
- [ ] Monte Carlo alternative:
  - [ ] Simulate 1000+ future paths under different macro regimes
  - [ ] Constrain for max drawdown, volatility, sector allocation

### 🔹 Phase 4: Risk Attribution Framework
- [ ] Use marginal contribution to risk (MCR) or Brinson model
- [ ] Breakdown:
  - [ ] Asset-level risk
  - [ ] Factor contributions
  - [ ] Macro-linked variance (e.g., via PCA loadings)

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

## 📄 License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template
- Inspired by institutional quantitative workflows
- Leverages modern Python data science ecosystem
