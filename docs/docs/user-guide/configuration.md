# Configuration Guide

This guide covers all configuration options for QuantFolio Engine, including environment variables, API keys, and system settings.

## Environment Setup

### Environment Variables

Create a `.env` file in your project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your settings
nano .env
```

### Required API Keys

```env
# Data Source API Keys
FRED_API_KEY=your_fred_api_key_here
NEWS_API_KEY=your_news_api_key_here

# Optional: AWS S3 for data storage
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name
```

### System Configuration

```env
# Data and Output Directories
DATA_DIR=data/
REPORTS_DIR=reports/
NOTEBOOKS_DIR=notebooks/

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/quantfolio.log

# Performance Settings
RANDOM_STATE=42
MAX_WORKERS=4
CHUNK_SIZE=1000

# Cache Settings
CACHE_DIR=.cache/
CACHE_TTL=3600
```

## API Key Setup

### FRED API Key

The Federal Reserve Economic Data (FRED) API provides macroeconomic indicators.

1. **Get API Key**:
   - Visit [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
   - Create a free account
   - Request an API key

2. **Test Your Key**:
```bash
# Test FRED API access
curl "https://api.stlouisfed.org/fred/series?series_id=GDP&api_key=YOUR_API_KEY&file_type=json"
```

3. **Add to Environment**:
```env
FRED_API_KEY=your_fred_api_key_here
```

### News API Key

The News API provides sentiment data for factor timing.

1. **Get API Key**:
   - Visit [News API](https://newsapi.org/)
   - Sign up for a free account
   - Get your API key

2. **Test Your Key**:
```bash
# Test News API access
curl "https://newsapi.org/v2/everything?q=finance&apiKey=YOUR_API_KEY"
```

3. **Add to Environment**:
```env
NEWS_API_KEY=your_news_api_key_here
```

### AWS S3 (Optional)

For cloud data storage and backup:

1. **Create AWS Account**:
   - Sign up at [AWS Console](https://aws.amazon.com/)
   - Create an IAM user with S3 permissions

2. **Configure AWS CLI**:
```bash
aws configure
# Enter your Access Key ID, Secret Access Key, and region
```

3. **Add to Environment**:
```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
```

## Data Configuration

### Data Sources

Configure which data sources to use:

```python
# In config.py
DATA_SOURCES = {
    'returns': {
        'enabled': True,
        'symbols': ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM'],
        'start_date': '2010-01-01',
        'frequency': 'monthly'
    },
    'macro': {
        'enabled': True,
        'indicators': ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS'],
        'frequency': 'monthly'
    },
    'sentiment': {
        'enabled': True,
        'entities': ['SPY', 'TLT', 'GLD'],
        'topics': ['finance', 'economy', 'markets']
    }
}
```

### Data Validation Settings

```python
DATA_VALIDATION = {
    'max_missing_pct': 0.1,  # Maximum 10% missing data
    'max_gap_days': 90,      # Maximum gap in days
    'extreme_value_threshold': 5.0,  # Z-score threshold
    'min_data_points': 100    # Minimum data points required
}
```

## Optimization Configuration

### Black-Litterman Settings

```python
BLACK_LITTERMAN_CONFIG = {
    'lambda_auto': True,      # Auto-calibrate λ
    'lambda_range': [0.5, 5.0],
    'gamma': 0.3,            # Grand view blend parameter
    'view_strength': 1.5,    # View strength multiplier
    'confidence_level': 0.95  # Confidence level for views
}
```

### Monte Carlo Settings

```python
MONTE_CARLO_CONFIG = {
    'n_simulations': 10000,   # Number of simulations
    'confidence_level': 0.95, # Confidence level for CVaR
    'risk_aversion': 3.0,     # Risk aversion parameter
    'seed': 42               # Random seed for reproducibility
}
```

### Portfolio Constraints

```python
PORTFOLIO_CONSTRAINTS = {
    'max_weight': 0.3,        # Maximum weight per asset
    'min_weight': 0.05,       # Minimum weight per asset
    'max_volatility': 0.15,   # Maximum annual volatility
    'target_return': 0.08,    # Target annual return
    'risk_free_rate': 0.02    # Risk-free rate
}
```

## Backtesting Configuration

### Walk-Forward Settings

```python
BACKTEST_CONFIG = {
    'train_years': 8,         # Training window in years
    'test_years': 2,          # Testing window in years
    'rebalance_frequency': 'monthly',  # Rebalancing frequency
    'transaction_costs': {     # Transaction cost structure
        'ETF': 0.0005,
        'Large_Cap': 0.001,
        'Small_Cap': 0.002
    }
}
```

### Performance Metrics

```python
PERFORMANCE_METRICS = {
    'risk_free_rate': 0.045,  # Risk-free rate for Sharpe ratio
    'confidence_level': 0.95,  # Confidence level for VaR/CVaR
    'benchmark': '60_40',      # Benchmark strategy
    'include_transaction_costs': True
}
```

## Factor Timing Configuration

### Regime Detection

```python
REGIME_CONFIG = {
    'n_regimes': 3,           # Number of regimes to detect
    'detection_method': 'kmeans',  # 'kmeans' or 'hmm'
    'features': ['GDP', 'UNRATE', 'CPIAUCSL'],  # Regime features
    'lookback_period': 60     # Lookback period in months
}
```

### Factor Exposure

```python
FACTOR_CONFIG = {
    'lookback_period': 60,    # Rolling window size
    'factor_method': 'macro',  # 'macro', 'fama_french', 'simple'
    'regime_dependent': True,  # Use regime-dependent weights
    'sentiment_weight': 0.1    # Weight for sentiment signals
}
```

## Logging Configuration

### Log Levels

```python
LOGGING_CONFIG = {
    'level': 'INFO',          # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/quantfolio.log',
    'max_size': '10MB',
    'backup_count': 5
}
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Run with debug output
quantfolio fetch-data --start-date 2020-01-01
```

## Performance Configuration

### Memory Settings

```python
PERFORMANCE_CONFIG = {
    'max_memory_gb': 8,       # Maximum memory usage
    'chunk_size': 1000,       # Data processing chunk size
    'max_workers': 4,         # Number of parallel workers
    'cache_enabled': True,     # Enable caching
    'cache_ttl': 3600         # Cache time-to-live in seconds
}
```

### Caching

Configure caching for expensive computations:

```python
CACHE_CONFIG = {
    'enabled': True,
    'directory': '.cache/',
    'ttl': 3600,              # Time-to-live in seconds
    'max_size': '1GB'         # Maximum cache size
}
```

## Security Configuration

### API Key Validation

```python
SECURITY_CONFIG = {
    'validate_api_keys': True, # Validate API keys on startup
    'key_format_check': True,  # Check API key format
    'rate_limit_check': True,  # Check API rate limits
    'secure_headers': True     # Use secure headers for requests
}
```

### Data Validation

```python
DATA_SECURITY = {
    'validate_input': True,    # Validate all input data
    'sanitize_output': True,   # Sanitize output data
    'log_sensitive': False,    # Don't log sensitive data
    'encrypt_cache': False     # Encrypt cached data
}
```

## Environment-Specific Configuration

### Development Environment

```env
# Development settings
LOG_LEVEL=DEBUG
CACHE_ENABLED=false
MAX_WORKERS=2
RANDOM_STATE=42
```

### Production Environment

```env
# Production settings
LOG_LEVEL=INFO
CACHE_ENABLED=true
MAX_WORKERS=8
RANDOM_STATE=42
```

### Testing Environment

```env
# Testing settings
LOG_LEVEL=WARNING
CACHE_ENABLED=false
MAX_WORKERS=1
RANDOM_STATE=42
```

## Configuration Validation

### Check Configuration

```bash
# Validate configuration
quantfolio status

# Check API keys
quantfolio validate-config

# Test data sources
quantfolio test-connections
```

### Configuration Errors

Common configuration issues and solutions:

**"API key not found"**:
```bash
# Check environment variables
echo $FRED_API_KEY
echo $NEWS_API_KEY

# Verify .env file
cat .env
```

**"Invalid API key format"**:
```bash
# Test API key format
python -c "import os; print(len(os.getenv('FRED_API_KEY', '')))"
```

**"Permission denied"**:
```bash
# Check file permissions
ls -la data/
ls -la reports/

# Fix permissions
chmod 755 data/ reports/
```

## Advanced Configuration

### Custom Data Sources

Add custom data sources:

```python
# In config.py
CUSTOM_DATA_SOURCES = {
    'custom_returns': {
        'type': 'csv',
        'path': 'data/custom_returns.csv',
        'date_column': 'date',
        'return_columns': ['asset1', 'asset2', 'asset3']
    },
    'custom_factors': {
        'type': 'csv',
        'path': 'data/custom_factors.csv',
        'date_column': 'date',
        'factor_columns': ['factor1', 'factor2']
    }
}
```

### Custom Constraints

Define custom portfolio constraints:

```python
CUSTOM_CONSTRAINTS = {
    'sector_limits': {
        'technology': 0.4,
        'financial': 0.3,
        'healthcare': 0.2
    },
    'country_limits': {
        'US': 0.7,
        'international': 0.3
    },
    'style_limits': {
        'growth': 0.6,
        'value': 0.4
    }
}
```

### Custom Metrics

Add custom performance metrics:

```python
CUSTOM_METRICS = {
    'custom_ratio': {
        'function': 'custom_ratio_function',
        'description': 'Custom risk-adjusted ratio'
    },
    'regime_alpha': {
        'function': 'regime_alpha_function',
        'description': 'Regime-adjusted alpha'
    }
}
```

## Configuration Best Practices

### 1. Security

- **Never commit API keys** to version control
- **Use environment variables** for sensitive data
- **Validate all input** data
- **Log security events** appropriately

### 2. Performance

- **Cache expensive computations** when possible
- **Use appropriate chunk sizes** for large datasets
- **Monitor memory usage** during processing
- **Optimize for your hardware** configuration

### 3. Maintainability

- **Document all configuration** options
- **Use consistent naming** conventions
- **Version control** configuration changes
- **Test configuration** changes thoroughly

### 4. Scalability

- **Design for growth** in data size
- **Plan for additional** data sources
- **Consider cloud deployment** options
- **Monitor resource usage** patterns

## Troubleshooting

### Common Issues

**Configuration not loaded**:
```bash
# Check environment file
ls -la .env

# Verify variable loading
python -c "import os; print(os.getenv('FRED_API_KEY'))"
```

**API rate limits**:
```bash
# Check API usage
quantfolio status

# Implement rate limiting
export API_RATE_LIMIT=100
```

**Memory issues**:
```bash
# Reduce memory usage
export MAX_MEMORY_GB=4
export CHUNK_SIZE=500
```

### Getting Help

```bash
# Check system status
quantfolio status

# Validate configuration
quantfolio validate-config

# Test connections
quantfolio test-connections

# View logs
tail -f logs/quantfolio.log
```

---

*For more advanced configuration options, see the [Advanced Topics](../advanced/) section.*
