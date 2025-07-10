"""
Data validation module for backtesting.

This module validates data quality and sufficiency for walk-forward backtesting,
ensuring we have enough historical data and proper data quality.
"""

from datetime import timedelta
from typing import Dict, Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd


class DataValidator:
    """
    Validates data quality and sufficiency for backtesting.

    Ensures:
    - Sufficient historical data for training and testing
    - Data quality (no large gaps, reasonable values)
    - Proper alignment between datasets
    - Continuity for rolling calculations
    """

    def __init__(
        self,
        min_training_years: int = 8,
        min_testing_years: int = 2,
        max_gap_days: int = 30,
        min_data_completeness: float = 0.95,
        rebalance_frequency: str = "monthly",
    ):
        """
        Initialize data validator.

        Args:
            min_training_years: Minimum years required for training
            min_testing_years: Minimum years required for testing
            max_gap_days: Maximum allowed gap in days
            min_data_completeness: Minimum data completeness ratio
        """
        self.min_training_years = min_training_years
        self.min_testing_years = min_testing_years
        self.max_gap_days = max_gap_days
        self.min_data_completeness = min_data_completeness
        self.rebalance_frequency = rebalance_frequency

        # Define gap tolerance based on rebalancing frequency
        self.gap_tolerance_days = {
            "monthly": 35,  # ~1 month
            "quarterly": 120,  # ~4 months
            "annual": 400,  # ~13 months
        }.get(
            rebalance_frequency, 35
        )  # Default to monthly if unknown

    def validate_data_for_backtesting(
        self,
        returns_df: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame] = None,
        factor_regimes: Optional[pd.DataFrame] = None,
        sentiment_scores: Optional[pd.DataFrame] = None,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Validate all datasets for backtesting.

        Args:
            returns_df: Asset returns DataFrame
            factor_exposures: Factor exposures DataFrame
            factor_regimes: Factor regimes DataFrame
            sentiment_scores: Sentiment scores DataFrame
            macro_data: Macroeconomic data DataFrame

        Returns:
            Tuple of (is_valid, validation_messages)
        """
        logger.info("Starting data validation for backtesting...")

        validation_messages = {}
        all_valid = True

        # 1. Validate returns data
        returns_valid, returns_msg = self._validate_returns_data(returns_df)
        validation_messages["returns"] = returns_msg
        all_valid = all_valid and returns_valid

        # 2. Validate factor exposures if provided
        if factor_exposures is not None:
            exposures_valid, exposures_msg = self._validate_factor_data(
                factor_exposures, "exposures"
            )
            validation_messages["factor_exposures"] = exposures_msg
            all_valid = all_valid and exposures_valid

        # 3. Validate factor regimes if provided
        if factor_regimes is not None:
            regimes_valid, regimes_msg = self._validate_factor_data(
                factor_regimes, "regimes"
            )
            validation_messages["factor_regimes"] = regimes_msg
            all_valid = all_valid and regimes_valid

        # 4. Validate sentiment data if provided
        if sentiment_scores is not None:
            sentiment_valid, sentiment_msg = self._validate_sentiment_data(
                sentiment_scores
            )
            validation_messages["sentiment_scores"] = sentiment_msg
            all_valid = all_valid and sentiment_valid

        # 5. Validate macro data if provided
        if macro_data is not None:
            macro_valid, macro_msg = self._validate_macro_data(macro_data)
            validation_messages["macro_data"] = macro_msg
            all_valid = all_valid and macro_valid

        # 6. Validate data alignment
        alignment_valid, alignment_msg = self._validate_data_alignment(
            returns_df, factor_exposures, factor_regimes, sentiment_scores, macro_data
        )
        validation_messages["alignment"] = alignment_msg
        all_valid = all_valid and alignment_valid

        # 7. Validate sufficient history
        history_valid, history_msg = self._validate_sufficient_history(returns_df)
        validation_messages["history"] = history_msg
        all_valid = all_valid and history_valid

        if all_valid:
            logger.success("✅ All data validation checks passed!")
        else:
            logger.warning("⚠️ Some data validation checks failed")
            for dataset, message in validation_messages.items():
                if "FAILED" in message:
                    logger.error(f"{dataset}: {message}")

        return all_valid, validation_messages

    def _validate_returns_data(self, returns_df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate returns data quality."""
        try:
            # Check basic structure
            if returns_df.empty:
                return False, "FAILED: Returns DataFrame is empty"

            if returns_df.shape[1] < 2:
                return (
                    False,
                    "FAILED: Need at least 2 assets for portfolio optimization",
                )

            # Check for reasonable return values
            if (returns_df > 1.0).any().any():
                return (
                    False,
                    "FAILED: Returns contain values > 100% (likely price data instead of returns)",
                )

            if (returns_df < -0.5).any().any():
                return (
                    False,
                    "FAILED: Returns contain values < -50% (suspiciously large losses)",
                )

            # Check data completeness
            completeness = 1 - returns_df.isnull().sum().sum() / (
                returns_df.shape[0] * returns_df.shape[1]
            )
            if completeness < self.min_data_completeness:
                return (
                    False,
                    f"FAILED: Data completeness {completeness:.2%} < {self.min_data_completeness:.2%}",
                )

            # Check for large gaps (allow up to 35 days for monthly data)
            # FIXED: Sort dates first to avoid negative gaps from unsorted indices
            dates = returns_df.index.sort_values()
            date_gaps = dates.to_series().diff().dt.days
            max_gap = date_gaps.max(skipna=True)
            if max_gap > max(self.max_gap_days, self.gap_tolerance_days):
                return (
                    False,
                    f"FAILED: Maximum gap {max_gap} days > {max(self.max_gap_days, self.gap_tolerance_days)} days (tolerance for {self.rebalance_frequency} rebalancing)",
                )

            return (
                True,
                f"PASSED: {returns_df.shape[1]} assets, {returns_df.shape[0]} periods, completeness {completeness:.2%}",
            )

        except Exception as e:
            return False, f"FAILED: Error validating returns data: {str(e)}"

    def _validate_factor_data(
        self, factor_df: pd.DataFrame, data_type: str
    ) -> Tuple[bool, str]:
        """Validate factor exposures or regimes data."""
        try:
            if factor_df.empty:
                return False, f"FAILED: {data_type} DataFrame is empty"

            # Handle long format with date and asset columns
            if "date" in factor_df.columns and "asset" in factor_df.columns:
                # For long format, check numeric columns (excluding date and asset)
                numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return False, f"FAILED: {data_type} has no numeric columns"

                # Check for reasonable values in numeric columns
                if data_type == "exposures":
                    # Factor exposures should be reasonable (not extreme)
                    if (factor_df[numeric_cols].abs() > 20).any().any():
                        return False, f"FAILED: {data_type} contain extreme values > 20"
                elif data_type == "regimes":
                    # Find the regime column (should be the last column)
                    regime_col = factor_df.columns[-1]
                    if not np.issubdtype(factor_df[regime_col].dtype, np.integer):
                        return (
                            False,
                            f"FAILED: {data_type} should contain integer regime labels",
                        )

                # Check data completeness
                completeness = 1 - factor_df[numeric_cols].isnull().sum().sum() / (
                    factor_df.shape[0] * len(numeric_cols)
                )
                if completeness < self.min_data_completeness:
                    return (
                        False,
                        f"FAILED: {data_type} completeness {completeness:.2%} < {self.min_data_completeness:.2%}",
                    )

                return (
                    True,
                    f"PASSED: {data_type} shape {factor_df.shape}, completeness {completeness:.2%}",
                )
            # Handle MultiIndex format (date, asset)
            elif isinstance(factor_df.index, pd.MultiIndex):
                # For MultiIndex, check numeric columns (excluding date and asset)
                numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return False, f"FAILED: {data_type} has no numeric columns"

                # Check for reasonable values in numeric columns
                if data_type == "exposures":
                    # Factor exposures should be reasonable (not extreme)
                    if (factor_df[numeric_cols].abs() > 20).any().any():
                        return False, f"FAILED: {data_type} contain extreme values > 20"
                elif data_type == "regimes":
                    # Find the regime column (should be the last column)
                    regime_col = factor_df.columns[-1]
                    if not np.issubdtype(factor_df[regime_col].dtype, np.integer):
                        return (
                            False,
                            f"FAILED: {data_type} should contain integer regime labels",
                        )

                # Check data completeness
                completeness = 1 - factor_df[numeric_cols].isnull().sum().sum() / (
                    factor_df.shape[0] * len(numeric_cols)
                )
                if completeness < self.min_data_completeness:
                    return (
                        False,
                        f"FAILED: {data_type} completeness {completeness:.2%} < {self.min_data_completeness:.2%}",
                    )

                return (
                    True,
                    f"PASSED: {data_type} shape {factor_df.shape}, completeness {completeness:.2%}",
                )
            else:
                # Handle regular DataFrame format
                # Check for reasonable values
                if data_type == "exposures":
                    # Factor exposures should be reasonable (not extreme)
                    if (factor_df.abs() > 20).any().any():
                        return False, f"FAILED: {data_type} contain extreme values > 20"
                elif data_type == "regimes":
                    # Regimes should be integers
                    if not factor_df.dtypes.apply(
                        lambda x: np.issubdtype(x, np.integer)
                    ).all():
                        return (
                            False,
                            f"FAILED: {data_type} should contain integer regime labels",
                        )

                # Check data completeness
                completeness = 1 - factor_df.isnull().sum().sum() / (
                    factor_df.shape[0] * factor_df.shape[1]
                )
                if completeness < self.min_data_completeness:
                    return (
                        False,
                        f"FAILED: {data_type} completeness {completeness:.2%} < {self.min_data_completeness:.2%}",
                    )

                return (
                    True,
                    f"PASSED: {data_type} shape {factor_df.shape}, completeness {completeness:.2%}",
                )

        except Exception as e:
            return False, f"FAILED: Error validating {data_type}: {str(e)}"

    def _validate_sentiment_data(self, sentiment_df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate sentiment data quality."""
        try:
            if sentiment_df.empty:
                return False, "FAILED: Sentiment DataFrame is empty"

            # Check for reasonable sentiment values (typically -1 to 1)
            if (sentiment_df.abs() > 1.0).any().any():
                return False, "FAILED: Sentiment values outside expected range [-1, 1]"

            # Check data completeness
            completeness = 1 - sentiment_df.isnull().sum().sum() / (
                sentiment_df.shape[0] * sentiment_df.shape[1]
            )
            if completeness < self.min_data_completeness:
                return (
                    False,
                    f"FAILED: Sentiment completeness {completeness:.2%} < {self.min_data_completeness:.2%}",
                )

            return (
                True,
                f"PASSED: Sentiment shape {sentiment_df.shape}, completeness {completeness:.2%}",
            )

        except Exception as e:
            return False, f"FAILED: Error validating sentiment data: {str(e)}"

    def _validate_macro_data(self, macro_df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate macroeconomic data quality."""
        try:
            if macro_df.empty:
                return False, "FAILED: Macro DataFrame is empty"

            # Check for reasonable macro values (not extreme) - only numeric columns
            numeric_cols = macro_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return False, "FAILED: Macro data has no numeric columns"

            if (macro_df[numeric_cols].abs() > 25000).any().any():
                return False, "FAILED: Macro data contain extreme values > 25000"

            # Check data completeness
            completeness = 1 - macro_df[numeric_cols].isnull().sum().sum() / (
                macro_df.shape[0] * len(numeric_cols)
            )
            if completeness < self.min_data_completeness:
                return (
                    False,
                    f"FAILED: Macro completeness {completeness:.2%} < {self.min_data_completeness:.2%}",
                )

            return (
                True,
                f"PASSED: Macro shape {macro_df.shape}, completeness {completeness:.2%}",
            )

        except Exception as e:
            return False, f"FAILED: Error validating macro data: {str(e)}"

    def _validate_data_alignment(
        self,
        returns_df: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame],
        factor_regimes: Optional[pd.DataFrame],
        sentiment_scores: Optional[pd.DataFrame],
        macro_data: Optional[pd.DataFrame],
    ) -> Tuple[bool, str]:
        """Validate that all datasets are properly aligned."""
        try:
            datasets = {
                "returns": returns_df,
                "factor_exposures": factor_exposures,
                "factor_regimes": factor_regimes,
                "sentiment_scores": sentiment_scores,
                "macro_data": macro_data,
            }

            # Remove None datasets
            datasets = {k: v for k, v in datasets.items() if v is not None}

            if len(datasets) < 2:
                return True, "PASSED: Only one dataset provided, no alignment needed"

            # Check date range overlap
            date_ranges = {}
            for name, df in datasets.items():
                if name == "returns":
                    # Returns data has datetime index
                    date_ranges[name] = (df.index.min(), df.index.max())
                else:
                    # FIXED: Handle both long format (date column) and wide format (DatetimeIndex)
                    if "date" in df.columns:
                        # Long format: Convert string dates to datetime for comparison
                        df_dates = pd.to_datetime(df["date"])
                        date_ranges[name] = (df_dates.min(), df_dates.max())
                    elif isinstance(df.index, pd.DatetimeIndex):
                        # Wide format: Use DatetimeIndex directly
                        date_ranges[name] = (df.index.min(), df.index.max())
                    else:
                        # Skip datasets without date information
                        continue

            if len(date_ranges) < 2:
                return True, "PASSED: Insufficient date information for alignment check"

            # Find common date range
            common_start = max(r[0] for r in date_ranges.values())
            common_end = min(r[1] for r in date_ranges.values())

            if common_start >= common_end:
                return False, "FAILED: No common date range between datasets"

            # Check overlap period length
            overlap_days = (common_end - common_start).days
            min_overlap_days = (self.min_training_years + self.min_testing_years) * 365

            if overlap_days < min_overlap_days:
                return (
                    False,
                    f"FAILED: Overlap period {overlap_days} days < {min_overlap_days} days",
                )

            return (
                True,
                f"PASSED: Common date range {common_start.date()} to {common_end.date()}, {overlap_days} days",
            )

        except Exception as e:
            return False, f"FAILED: Error validating data alignment: {str(e)}"

    def _validate_sufficient_history(
        self, returns_df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Validate that we have sufficient historical data."""
        try:
            total_years = (
                returns_df.index.max() - returns_df.index.min()
            ).days / 365.25

            if total_years < (self.min_training_years + self.min_testing_years):
                return (
                    False,
                    f"FAILED: Total history {total_years:.1f} years < {self.min_training_years + self.min_testing_years} years",
                )

            # Suggest train/test split
            suggested_split_date = returns_df.index.max() - timedelta(
                days=self.min_testing_years * 365.25
            )
            training_years = (
                suggested_split_date - returns_df.index.min()
            ).days / 365.25

            if training_years < self.min_training_years:
                return (
                    False,
                    f"FAILED: Training period {training_years:.1f} years < {self.min_training_years} years",
                )

            return (
                True,
                f"PASSED: Total history {total_years:.1f} years, suggested split at {suggested_split_date.date()}",
            )

        except Exception as e:
            return False, f"FAILED: Error validating sufficient history: {str(e)}"

    def suggest_train_test_split(
        self, returns_df: pd.DataFrame
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Suggest optimal train/test split based on data availability.

        Args:
            returns_df: Returns DataFrame

        Returns:
            Tuple of (train_end_date, test_start_date)
        """
        total_years = (returns_df.index.max() - returns_df.index.min()).days / 365.25

        if total_years < (self.min_training_years + self.min_testing_years):
            raise ValueError(
                f"Insufficient data: {total_years:.1f} years < {self.min_training_years + self.min_testing_years} years"
            )

        # Use last min_testing_years for testing
        test_start = returns_df.index.max() - timedelta(
            days=self.min_testing_years * 365.25
        )
        train_end = test_start - timedelta(days=1)  # One day before test starts

        return train_end, test_start
