# QuantFolio Engine Documentation

Welcome to the QuantFolio Engine documentation! This comprehensive guide will help you understand and use our quantitative portfolio optimization engine.

## 🎯 What is QuantFolio Engine?

QuantFolio Engine is a sophisticated quantitative portfolio optimization platform designed for institutional asset management. It combines macroeconomic context, factor-timing models, and multi-source signal integration to create dynamic, risk-aware portfolio strategies.

### Key Features

- **📊 Multi-Source Data Integration**: Asset returns, macroeconomic indicators, and sentiment signals
- **🎯 Factor Timing**: Dynamic regime detection and factor exposure optimization
- **⚖️ Advanced Optimization**: Black-Litterman and Monte Carlo simulation methods
- **📈 Walk-Forward Backtesting**: Comprehensive validation framework with transaction costs
- **📊 Visualization**: Rich plotting and analysis tools
- **🔧 CLI Interface**: Easy-to-use command-line tools

## 🚀 Quick Start

```bash
# Install and setup
make create_environment
conda activate quantfolio-engine
make requirements

# Fetch data and run analysis
quantfolio fetch-data --start-date 2010-01-01
quantfolio generate-signals
quantfolio run-backtest --method combined --train-years 8 --test-years 2
quantfolio plot-backtest --type all
```

## 📚 Documentation Structure

### User Guide
- **[Installation](user-guide/installation.md)**: Setup and environment configuration
- **[Configuration](user-guide/configuration.md)**: API keys and system settings
- **[CLI Reference](user-guide/cli-reference.md)**: Complete command-line interface guide

### Tutorials
- **[Quick Start](tutorials/quick-start.md)**: Get up and running in minutes
- **[Backtesting](tutorials/backtesting.md)**: Walk-forward validation workflows

### Results
- **[Performance Results](results.md)**: Actual backtesting results and analysis

### Advanced Topics
- **[Factor Timing](advanced/factor-timing.md)**: Regime detection and factor exposure

### Development
- **[Contributing](development/contributing.md)**: How to contribute to the project

## 🎯 Target Audience

This documentation is designed for:

- **Institutional quantitative teams** looking for robust portfolio optimization tools
- **Hedge fund portfolio managers** seeking factor-timing capabilities
- **Smart-beta product managers** developing systematic strategies
- **Asset management firms** requiring institutional-grade tools
- **Quantitative researchers** exploring advanced portfolio optimization

## 🔧 Technology Stack

- **Python 3.11+**: Core programming language
- **Data Science**: pandas, numpy, scipy, statsmodels, scikit-learn
- **Optimization**: cvxpy for convex optimization
- **Data Sources**: yfinance, fredapi, newsapi
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest for comprehensive testing

## 📊 Key Concepts

### Factor Timing
Dynamic adjustment of portfolio weights based on macroeconomic regimes and factor performance cycles.

*Layman's explanation:* Think of factor timing like adjusting your wardrobe for the weather. Just as you wear lighter clothes in summer and heavier ones in winter, the system shifts investments based on the current "economic weather"—choosing the best mix of assets for each environment.

### Black-Litterman Model
Bayesian approach combining market equilibrium with investor views and quantitative signals.

*Layman's explanation:* Imagine you're making a group decision about where to eat. The Black-Litterman model starts with the "default" choice (what the market suggests), but then blends in everyone's opinions (your own research and signals) to reach a smarter, more balanced decision.

### Walk-Forward Backtesting
Robust validation framework with transaction costs, turnover analysis, and comprehensive performance metrics.

*Layman's explanation:* This is like practicing for a race by running on different tracks and weather conditions, not just the same one over and over. The system tests its strategies on new, unseen data to make sure they work in real life, not just in theory.

### Risk Attribution
Decomposition of portfolio risk and return into attributable components for transparency and client communication.

*Layman's explanation:* Risk attribution is like figuring out which ingredients in a recipe make it taste good or bad. It breaks down your investment results to show which parts helped or hurt, so you know what’s working and what needs changing.

## 🤝 Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and tutorials
- **Examples**: Working code examples in tutorials
- **CLI Help**: Use `quantfolio --help` for command-line assistance

## 📈 Roadmap

Our development roadmap includes:

- **Phase 4**: Risk attribution framework
- **Phase 5**: Web-based UI and deployment
- **Advanced Features**: Macro conditioning, dynamic correlations, enhanced risk modeling

---

*Built with ❤️ for the quantitative finance community*
