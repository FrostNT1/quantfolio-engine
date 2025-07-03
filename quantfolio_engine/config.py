import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

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
DEFAULT_START_DATE = "2015-01-01"
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
