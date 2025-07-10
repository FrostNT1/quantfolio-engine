from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data Configuration
ASSET_UNIVERSE = {
    "SPY": {"type": "Broad Index", "description": "S&P 500 ETF"},
    "TLT": {"type": "Bonds", "description": "Long-Term Treasury ETF"},
    "GLD": {"type": "Gold", "description": "Gold ETF"},
    "AAPL": {"type": "Tech", "description": "Apple Inc."},
    "MSFT": {"type": "Tech", "description": "Microsoft Corp."},
    "JPM": {"type": "Financials", "description": "JPMorgan Chase & Co."},
    "UNH": {"type": "Healthcare", "description": "UnitedHealth Group"},
    "WMT": {"type": "Consumer", "description": "Walmart Inc."},
    "XLE": {"type": "Energy", "description": "Energy Select Sector ETF"},
    "BA": {"type": "Industrials", "description": "Boeing Co."},
    "IWM": {"type": "Small-Cap", "description": "Russell 2000 ETF"},
    "EFA": {"type": "Global", "description": "Developed Markets ex-US ETF"},
}

MACRO_INDICATORS = {
    "CPIAUCSL": {"name": "U.S. CPI (inflation)", "source": "FRED"},
    "UNRATE": {"name": "Unemployment Rate", "source": "FRED"},
    "FEDFUNDS": {"name": "Fed Funds Rate", "source": "FRED"},
    "INDPRO": {"name": "Industrial Production", "source": "FRED"},
    "GDPC1": {"name": "Real GDP (Quarterly)", "source": "FRED"},
    "UMCSENT": {"name": "Consumer Confidence", "source": "FRED"},
    "GS10": {"name": "10-Year Treasury Yield", "source": "FRED"},
    "M2SL": {"name": "M2 Money Supply", "source": "FRED"},
    "DCOILWTICO": {"name": "Oil Prices (WTI Crude)", "source": "FRED"},
    "^VIX": {"name": "VIX Index", "source": "YAHOO"},
}

SENTIMENT_ENTITIES = ["AAPL", "MSFT", "JPM", "WMT", "SPY", "XLE"]
SENTIMENT_TOPICS = [
    "interest rates",
    "inflation",
    "recession",
    "growth",
    "earnings",
    "Fed",
    "oil prices",
    "supply chain",
]

# Data frequency and date ranges
DATA_FREQUENCY = "M"  # Monthly
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = None  # Will use current date

# API Configuration
FRED_API_KEY = None  # Will be loaded from environment
NEWS_API_KEY = None  # Will be loaded from environment

FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not FRED_API_KEY:
    logger.warning("FRED_API_KEY not found in environment variables")
if not NEWS_API_KEY:
    logger.warning("NEWS_API_KEY not found in environment variables")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


@dataclass
class DataConfig:
    """Configuration for data loading operations."""

    # Asset universe and data sources
    asset_universe: Dict[str, Dict]
    macro_indicators: Dict[str, Dict]
    sentiment_entities: list
    sentiment_topics: list

    # Date ranges
    start_date: str
    end_date: Optional[str]
    data_frequency: str = "M"

    # API keys
    fred_api_key: Optional[str] = None
    news_api_key: Optional[str] = None

    # Data directories
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR

    # Processing options
    save_raw: bool = True
    use_batch_download: bool = True
    max_workers: int = 4

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.asset_universe:
            raise ValueError("Asset universe cannot be empty")

        if not self.macro_indicators:
            raise ValueError("Macro indicators cannot be empty")

        # Ensure directories exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


def get_default_data_config() -> DataConfig:
    """Get default data configuration."""
    return DataConfig(
        asset_universe=ASSET_UNIVERSE,
        macro_indicators=MACRO_INDICATORS,
        sentiment_entities=SENTIMENT_ENTITIES,
        sentiment_topics=SENTIMENT_TOPICS,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        data_frequency=DATA_FREQUENCY,
        fred_api_key=FRED_API_KEY,
        news_api_key=NEWS_API_KEY,
    )


# Black-Litterman Configuration
@dataclass
class BlackLittermanConfig:
    """Configuration for Black-Litterman optimization parameters."""

    # Market risk aversion calibration
    lambda_auto: bool = True
    lambda_range: tuple = (0.5, 10.0)  # Much wider range to ensure π > rf
    lambda_points: int = 30

    # Grand view blend
    grand_view_gamma: float = 0.3  # Blend parameter for grand view

    # View strength and regime multipliers
    view_strength: float = 1.5
    regime_multipliers: dict[int, float] = field(default_factory=dict)

    # Prior uncertainty
    tau: float = 0.05

    # View uncertainty scaling
    view_uncertainty_scale: float = 0.5

    def __post_init__(self):
        """Set default regime multipliers if not provided."""
        if not self.regime_multipliers:
            self.regime_multipliers = {
                0: 2.0,  # Bull market - much stronger views
                1: 1.2,  # Neutral market - moderate views
                2: 1.5,  # Bear market - stronger views
            }

    def get_adjusted_view_strength(self, regime: Optional[int] = None) -> float:
        """Get view strength adjusted for current regime."""
        if regime is None:
            return self.view_strength

        regime_multiplier = self.regime_multipliers.get(regime, 1.0)
        return self.view_strength * regime_multiplier


# Default configuration
DEFAULT_BL_CONFIG = BlackLittermanConfig()
