# Contributing to QuantFolio Engine

Thank you for your interest in contributing to QuantFolio Engine! This guide will help you get started with development and contributing to the project.

## Development Setup

### Prerequisites

1. **Python 3.11+**: Required for development
2. **Git**: For version control
3. **Conda**: For environment management
4. **Make**: For build automation (optional but recommended)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/FrostNT1/quantfolio-engine.git
cd quantfolio-engine

# Create development environment
make create_environment
conda activate quantfolio-engine

# Install development dependencies
make requirements-dev

# Install pre-commit hooks
make install-hooks
```

### Verify Setup

```bash
# Run tests to ensure everything works
make test

# Run linting
make lint

# Run type checking
make type-check
```

## Development Workflow

### 1. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-description
```

### 2. Make Your Changes

Follow the coding standards:

- **Python**: Follow PEP 8 style guide
- **Type Hints**: Use type annotations for all functions
- **Documentation**: Add docstrings for new functions
- **Tests**: Write tests for new functionality

### 3. Test Your Changes

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_your_module.py

# Run with coverage
pytest --cov=quantfolio_engine tests/

# Run linting
make lint

# Run type checking
make type-check
```

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new factor timing algorithm" -m "- Implement regime-dependent factor weights" -m "- Add tests for new functionality" -m "- Update documentation"
```

### 5. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Code Standards

### Python Style Guide

Follow PEP 8 with these additional rules:

```python
# Good: Clear variable names
factor_exposures = calculate_rolling_exposures(returns, factors)

# Bad: Unclear names
fe = calc_re(returns, factors)

# Good: Type hints
def calculate_rolling_exposures(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    lookback: int = 60
) -> np.ndarray:
    """Calculate rolling factor exposures."""
    pass

# Good: Comprehensive docstrings
def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "combined",
    constraints: Optional[Dict] = None
) -> PortfolioResult:
    """
    Optimize portfolio using specified method.

    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns with datetime index and asset columns
    method : str, default="combined"
        Optimization method: "black_litterman", "monte_carlo", or "combined"
    constraints : Optional[Dict], default=None
        Portfolio constraints dictionary

    Returns:
    --------
    PortfolioResult
        Optimization results including weights and metrics

    Raises:
    -------
    ValueError
        If method is not supported or constraints are invalid
    """
    pass
```

### File Organization

```
quantfolio_engine/
├── __init__.py
├── cli.py                 # Command-line interface
├── config.py              # Configuration settings
├── data/                  # Data handling modules
│   ├── __init__.py
│   └── data_loader.py
├── modeling/              # Machine learning models
│   ├── __init__.py
│   ├── predict.py
│   └── train.py
├── optimizer/             # Portfolio optimization
│   ├── __init__.py
│   ├── black_litterman.py
│   ├── monte_carlo.py
│   └── portfolio_engine.py
├── signals/               # Signal generation
│   ├── __init__.py
│   └── factor_timing.py
├── utils/                 # Utility functions
│   └── __init__.py
└── plots.py               # Visualization functions
```

### Testing Standards

Write comprehensive tests for all new functionality:

```python
# tests/test_new_feature.py
import pytest
import pandas as pd
import numpy as np
from quantfolio_engine.new_module import new_function


class TestNewFeature:
    """Test suite for new feature."""

    def setup_method(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'asset1': [0.01, 0.02, -0.01],
            'asset2': [0.015, 0.025, -0.005]
        })

    def test_new_function_basic(self):
        """Test basic functionality."""
        result = new_function(self.sample_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_data)

    def test_new_function_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty data
        with pytest.raises(ValueError):
            new_function(pd.DataFrame())

        # Test with invalid parameters
        with pytest.raises(ValueError):
            new_function(self.sample_data, invalid_param=True)

    def test_new_function_regression(self):
        """Test regression against known results."""
        expected_result = pd.DataFrame({
            'result': [0.5, 0.6, 0.4]
        })
        result = new_function(self.sample_data)
        pd.testing.assert_frame_equal(result, expected_result, atol=1e-6)
```

## Adding New Features

### 1. Data Sources

To add a new data source:

```python
# quantfolio_engine/data/new_source.py
from typing import Dict, Optional
import pandas as pd
from .base_loader import BaseDataLoader


class NewSourceLoader(BaseDataLoader):
    """Loader for new data source."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or self._get_api_key('NEW_SOURCE_API_KEY')

    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Fetch data from new source.

        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        symbols : Optional[list]
            List of symbols to fetch

        Returns:
        --------
        pd.DataFrame
            Data with datetime index and symbol columns
        """
        # Implementation here
        pass
```

### 2. Optimization Methods

To add a new optimization method:

```python
# quantfolio_engine/optimizer/new_method.py
from typing import Dict, Optional
import pandas as pd
import numpy as np
from .base_optimizer import BaseOptimizer


class NewMethodOptimizer(BaseOptimizer):
    """New optimization method implementation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.parameters = kwargs

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Optimize portfolio using new method.

        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        constraints : Optional[Dict]
            Portfolio constraints

        Returns:
        --------
        Dict
            Optimization results with weights and metrics
        """
        # Implementation here
        pass
```

### 3. CLI Commands

To add a new CLI command:

```python
# In quantfolio_engine/cli.py
@click.command()
@click.option('--input-file', type=str, required=True, help='Input file path')
@click.option('--output-file', type=str, help='Output file path')
@click.option('--param', type=float, default=1.0, help='Parameter value')
def new_command(input_file: str, output_file: Optional[str], param: float):
    """Execute new command."""
    try:
        # Load data
        data = load_data(input_file)

        # Process data
        result = process_data(data, param)

        # Save results
        if output_file:
            save_results(result, output_file)
        else:
            click.echo(result)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
```

## Documentation

### Code Documentation

All functions should have comprehensive docstrings:

```python
def complex_function(
    data: pd.DataFrame,
    param1: float,
    param2: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Brief description of what the function does.

    Longer description explaining the algorithm, methodology, or purpose.
    Include mathematical formulas if relevant.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data with specific format requirements
    param1 : float
        Description of parameter and its range/constraints
    param2 : Optional[str], default=None
        Optional parameter description

    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        - DataFrame: Processed results
        - Dict: Additional metadata and statistics

    Raises:
    -------
    ValueError
        When data format is invalid
    RuntimeError
        When processing fails

    Examples:
    --------
    >>> data = pd.DataFrame({'A': [1, 2, 3]})
    >>> result, metadata = complex_function(data, 0.5)
    >>> print(result)
    """
    pass
```

### User Documentation

Update user documentation when adding features:

1. **CLI Reference**: Add new commands to `docs/docs/user-guide/cli-reference.md`
2. **Tutorials**: Create tutorials for new features
3. **API Reference**: Document new modules and functions
4. **Examples**: Provide working examples

## Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test module interactions
3. **Regression Tests**: Test against known results
4. **Performance Tests**: Test computational efficiency

### Test Naming

```python
# Good test names
def test_calculate_rolling_exposures_with_valid_data():
    """Test rolling exposure calculation with valid input."""

def test_calculate_rolling_exposures_with_missing_values():
    """Test rolling exposure calculation with missing data."""

def test_calculate_rolling_exposures_edge_cases():
    """Test rolling exposure calculation edge cases."""
```

### Test Data

Use realistic test data:

```python
# Create realistic test data
@pytest.fixture
def sample_returns():
    """Create sample return data for testing."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    np.random.seed(42)

    return pd.DataFrame({
        'SPY': np.random.normal(0.01, 0.05, len(dates)),
        'TLT': np.random.normal(0.005, 0.03, len(dates)),
        'GLD': np.random.normal(0.008, 0.04, len(dates))
    }, index=dates)
```

## Performance Considerations

### Optimization Guidelines

1. **Vectorization**: Use NumPy/Pandas operations when possible
2. **Memory Efficiency**: Avoid unnecessary data copies
3. **Caching**: Cache expensive computations
4. **Parallelization**: Use multiprocessing for independent operations

### Profiling

```bash
# Profile code performance
python -m cProfile -o profile.stats your_script.py

# Analyze profile results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## Security Guidelines

### API Keys

- Never commit API keys to version control
- Use environment variables for sensitive data
- Validate API key format and permissions

### Data Validation

```python
def validate_input_data(data: pd.DataFrame) -> None:
    """Validate input data for security and correctness."""
    if data.empty:
        raise ValueError("Data cannot be empty")

    if data.isnull().all().all():
        raise ValueError("Data contains only null values")

    # Check for suspicious patterns
    if (data > 1.0).any().any():
        raise ValueError("Returns exceed 100% - check data format")
```

## Release Process

### Version Management

Follow semantic versioning:

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. **Update Version**: Update version in `pyproject.toml`
2. **Update Changelog**: Document changes in `CHANGELOG.md`
3. **Run Tests**: Ensure all tests pass
4. **Update Documentation**: Update docs for new features
5. **Create Release**: Tag and release on GitHub

### Pre-Release Testing

```bash
# Run full test suite
make test

# Run linting and type checking
make lint
make type-check

# Test CLI commands
quantfolio --help
quantfolio fetch-data --help

# Test with sample data
quantfolio fetch-data --start-date 2023-01-01 --end-date 2023-12-31
```

## Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and discussions
- **Pull Requests**: For code contributions

### Issue Templates

When creating issues, use the provided templates:

- **Bug Report**: Include steps to reproduce
- **Feature Request**: Describe the feature and use case
- **Documentation**: Specify what needs improvement

### Code Review Process

1. **Automated Checks**: All PRs must pass CI checks
2. **Code Review**: At least one maintainer must approve
3. **Testing**: New code must have adequate test coverage
4. **Documentation**: New features must be documented

## Recognition

Contributors will be recognized in:

- **README.md**: List of contributors
- **CHANGELOG.md**: Credit for significant contributions
- **GitHub**: Contributor profile and statistics

---

*Thank you for contributing to QuantFolio Engine! Your contributions help make quantitative finance more accessible and robust.*
