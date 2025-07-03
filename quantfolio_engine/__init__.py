"""
QuantFolio Engine - Smart Portfolio Construction Using Factor Timing and Multi-Source Signal Integration

A quantitative portfolio optimization engine designed for institutional asset management,
combining macroeconomic context, factor-timing models, and LLM-driven sentiment signals
into a dynamic portfolio optimizer and risk explainer.
"""

__version__ = "0.1.0"
__author__ = "Shivam Tyagi"

from quantfolio_engine import config  # noqa: F401

# Import main modules for easy access
# TODO: Uncomment imports when modules are implemented
# from quantfolio_engine.attribution import risk_attribution  # noqa: F401
from quantfolio_engine.data import data_loader  # noqa: F401

# from quantfolio_engine.optimizer import black_litterman  # noqa: F401
# from quantfolio_engine.signals import factor_timing  # noqa: F401
