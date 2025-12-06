"""
Archive utility for backing up sentiment data before migration.
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
import pandas as pd

from quantfolio_engine.config import DATA_DIR, PROCESSED_DATA_DIR

ARCHIVE_DIR = DATA_DIR / "archive"


def archive_sentiment_data() -> str:
    """
    Archive existing sentiment data files to archive directory.

    Returns:
        Archive directory path with timestamp
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"sentiment_vader_{timestamp}"
    archive_path.mkdir(parents=True, exist_ok=True)

    sentiment_files = [
        "sentiment_monthly.csv",
        "sentiment_monthly_normalized.csv",
        "sentiment_monthly.parquet",
    ]

    archived_files = {}
    checksums = {}

    for filename in sentiment_files:
        source_file = PROCESSED_DATA_DIR / filename
        if source_file.exists():
            dest_file = archive_path / filename
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if filename.endswith(".csv"):
                df = pd.read_csv(source_file, index_col=0, parse_dates=True)
                df.to_csv(dest_file)
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(source_file)
                df.to_parquet(dest_file)

            # Calculate checksum
            with open(dest_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            archived_files[filename] = str(dest_file)
            checksums[filename] = file_hash

            logger.info(f"Archived {filename} to {dest_file}")
        else:
            logger.warning(f"Sentiment file not found: {source_file}")

    # Create metadata file
    metadata = {
        "archive_date": timestamp,
        "archive_path": str(archive_path),
        "provider": "VADER",
        "files_archived": archived_files,
        "checksums": checksums,
        "data_range": _get_sentiment_data_range(),
    }

    metadata_file = archive_path / "ARCHIVE_METADATA.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Sentiment data archived to {archive_path}")
    logger.info(f"Metadata saved to {metadata_file}")

    return str(archive_path)


def _get_sentiment_data_range() -> Dict[str, Optional[str]]:
    """Get date range from sentiment data if available."""
    sentiment_file = PROCESSED_DATA_DIR / "sentiment_monthly.csv"
    if sentiment_file.exists():
        try:
            df = pd.read_csv(sentiment_file, index_col=0, parse_dates=True)
            return {
                "start_date": str(df.index.min()) if len(df) > 0 else None,
                "end_date": str(df.index.max()) if len(df) > 0 else None,
                "num_months": len(df),
            }
        except Exception as e:
            logger.warning(f"Could not determine data range: {e}")
    return {"start_date": None, "end_date": None, "num_months": 0}


def list_archives() -> list:
    """List all sentiment archives."""
    if not ARCHIVE_DIR.exists():
        return []

    archives = []
    for archive_dir in ARCHIVE_DIR.iterdir():
        if archive_dir.is_dir() and archive_dir.name.startswith("sentiment_"):
            metadata_file = archive_dir / "ARCHIVE_METADATA.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                archives.append(metadata)

    return sorted(archives, key=lambda x: x["archive_date"], reverse=True)


def restore_archive(archive_path: Optional[str] = None) -> bool:
    """
    Restore sentiment data from archive.

    Args:
        archive_path: Path to archive directory. If None, uses most recent.

    Returns:
        True if successful
    """
    if archive_path is None:
        archives = list_archives()
        if not archives:
            logger.error("No archives found")
            return False
        archive_path = archives[0]["archive_path"]

    archive_dir = Path(archive_path)
    if not archive_dir.exists():
        logger.error(f"Archive directory not found: {archive_dir}")
        return False

    metadata_file = archive_dir / "ARCHIVE_METADATA.json"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return False

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Restore files
    for filename, archived_path in metadata["files_archived"].items():
        source_file = Path(archived_path)
        dest_file = PROCESSED_DATA_DIR / filename

        if source_file.exists():
            if filename.endswith(".csv"):
                df = pd.read_csv(source_file, index_col=0, parse_dates=True)
                df.to_csv(dest_file)
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(source_file)
                df.to_parquet(dest_file)

            logger.info(f"Restored {filename} from archive")
        else:
            logger.warning(f"Archived file not found: {source_file}")

    logger.info(f"✅ Sentiment data restored from {archive_path}")
    return True
