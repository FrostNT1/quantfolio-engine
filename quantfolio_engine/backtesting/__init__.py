"""
Backtesting module for QuantFolio Engine.

This module provides walk-forward backtesting capabilities including:
- Data validation for backtesting
- Walk-forward framework with configurable windows
- Transaction cost modeling
- Performance metrics calculation
- Benchmark comparison
"""

from .data_validator import DataValidator
from .walk_forward import WalkForwardBacktester

__all__ = ["DataValidator", "WalkForwardBacktester"]
