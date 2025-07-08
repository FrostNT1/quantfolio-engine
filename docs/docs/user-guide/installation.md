# Installation Guide

This guide will walk you through installing and setting up QuantFolio Engine on your system.

## Prerequisites

### System Requirements

- **Operating System**: macOS, Linux, or Windows (WSL recommended for Windows)
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space for data and models
- **Internet**: Required for data fetching

### Required Software

- **Conda** or **Miniconda**: For environment management
- **Git**: For cloning the repository
- **Make**: For build automation (optional but recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/FrostNT1/quantfolio-engine.git
cd quantfolio-engine
```

### 2. Create Conda Environment

```bash
# Create the environment
make create_environment

# Activate the environment
conda activate quantfolio-engine
```

### 3. Install Dependencies

```bash
# Install all required packages
make requirements
```

### 4. Verify Installation

```bash
# Check if quantfolio CLI is available
quantfolio --help

# Run tests to ensure everything works
make test
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:

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

### 2. API Keys Setup

#### FRED API Key
1. Visit [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a free account
3. Request an API key
4. Add to your `.env` file

#### News API Key
1. Visit [News API](https://newsapi.org/)
2. Sign up for a free account
3. Get your API key
4. Add to your `.env` file

### 3. Verify Configuration

```bash
# Check system status
quantfolio status
```

## Quick Test

Run a quick test to ensure everything is working:

```bash
# Fetch some sample data
quantfolio fetch-data --start-date 2023-01-01 --end-date 2023-12-31

# Generate signals
quantfolio generate-signals

# Run a quick backtest
quantfolio run-backtest --method combined --train-years 2 --test-years 1
```

## Troubleshooting

### Common Issues

#### 1. Conda Environment Issues

```bash
# If conda create fails, try:
conda create -n quantfolio-engine python=3.11

# If activation fails:
conda init zsh  # or conda init bash
source ~/.zshrc  # or source ~/.bashrc
```

#### 2. Package Installation Issues

```bash
# Update conda
conda update conda

# Clean conda cache
conda clean --all

# Reinstall packages
make requirements
```

#### 3. API Key Issues

```bash
# Check if API keys are loaded
quantfolio status

# Test individual data sources
quantfolio fetch-data --type returns
quantfolio fetch-data --type macro
```

#### 4. Permission Issues

```bash
# Make sure you have write permissions
chmod -R 755 data/
chmod -R 755 reports/
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the terminal output
2. **Verify API keys**: Ensure your API keys are valid and have proper permissions
3. **Check dependencies**: Run `make test` to verify all packages are installed correctly
4. **GitHub Issues**: Report bugs on the GitHub repository

## Development Setup

For developers who want to contribute:

```bash
# Install development dependencies
make requirements-dev

# Install pre-commit hooks
make install-hooks

# Run linting
make lint

# Run type checking
make type-check
```

## Docker Installation (Alternative)

If you prefer Docker:

```bash
# Build the Docker image
docker build -t quantfolio-engine .

# Run with Docker
docker run -it --rm quantfolio-engine quantfolio --help
```

## Next Steps

After successful installation:

1. **Read the [Configuration Guide](configuration.md)** to understand system settings
2. **Follow the [Quick Start Tutorial](../tutorials/quick-start.md)** to run your first analysis
3. **Explore the [CLI Reference](cli-reference.md)** for all available commands
4. **Check out the [Advanced Topics](../advanced/)** for deep dives into specific features

---

*Need help? Check our [GitHub Issues](https://github.com/FrostNT1/quantfolio-engine/issues) or create a new one!*
