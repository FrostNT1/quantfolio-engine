"""
Main portfolio optimization engine for QuantFolio Engine.

This module orchestrates portfolio optimization using both Black-Litterman
and Monte Carlo approaches, integrating factor timing signals and sentiment data.
"""

from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd

from quantfolio_engine.config import PROCESSED_DATA_DIR
from quantfolio_engine.optimizer.black_litterman import BlackLittermanOptimizer
from quantfolio_engine.optimizer.monte_carlo import MonteCarloOptimizer


class PortfolioOptimizationEngine:
    """
    Main portfolio optimization engine.

    Combines Black-Litterman and Monte Carlo approaches for robust
    portfolio optimization with factor timing integration.
    """

    def __init__(
        self,
        method: str = "black_litterman",
        risk_free_rate: float = 0.02,
        max_drawdown: float = 0.15,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ):
        """
        Initialize portfolio optimization engine.

        Args:
            method: Optimization method ("black_litterman", "monte_carlo", "combined")
            risk_free_rate: Annual risk-free rate
            max_drawdown: Maximum allowed drawdown
            confidence_level: Confidence level for risk metrics
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.risk_free_rate = risk_free_rate
        self.max_drawdown = max_drawdown
        self.confidence_level = confidence_level
        self.random_state: Optional[int] = random_state

        # Initialize optimizers
        self.bl_optimizer = BlackLittermanOptimizer(risk_free_rate=risk_free_rate)
        self.mc_optimizer = MonteCarloOptimizer(
            risk_free_rate=risk_free_rate,
            max_cvar_loss=max_drawdown,
            confidence_level=confidence_level,
            random_state=random_state,
        )

        # BL parameter storage
        self.bl_lambda = "auto"
        self.bl_gamma = 0.3
        self.bl_view_strength = 1.5
        self.bl_lambda_range: Optional[str] = None  # Will use config default if None

    def set_bl_parameters(
        self,
        lambda_param: Optional[str] = None,
        gamma: Optional[float] = None,
        view_strength: Optional[float] = None,
        lambda_range: Optional[str] = None,
    ) -> None:
        """
        Set Black-Litterman optimization parameters.

        Args:
            lambda_param: λ parameter ('auto' for calibration or float value)
            gamma: Grand view blend parameter γ
            view_strength: View strength multiplier
        """
        if lambda_param is not None:
            self.bl_lambda = lambda_param
        if gamma is not None:
            self.bl_gamma = gamma
        if view_strength is not None:
            self.bl_view_strength = view_strength
        if lambda_range is not None:
            self.bl_lambda_range = lambda_range

        logger.info(
            f"Set BL parameters: λ={self.bl_lambda}, γ={self.bl_gamma}, view_strength={self.bl_view_strength}, λ_range={self.bl_lambda_range}"
        )

    def load_data(
        self,
        returns_file: Optional[Union[str, Path]] = None,
        factor_exposures_file: Optional[Union[str, Path]] = None,
        factor_regimes_file: Optional[Union[str, Path]] = None,
        sentiment_file: Optional[Union[str, Path]] = None,
        macro_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for portfolio optimization.

        Args:
            returns_file: Path to returns data
            factor_exposures_file: Path to factor exposures
            factor_regimes_file: Path to factor regimes
            sentiment_file: Path to sentiment data
            macro_file: Path to macro data

        Returns:
            Dictionary with loaded data
        """
        logger.info("Loading data for portfolio optimization...")

        # Convert string paths to Path objects for consistent handling
        if returns_file is not None:
            returns_file = Path(returns_file)
        if factor_exposures_file is not None:
            factor_exposures_file = Path(factor_exposures_file)
        if factor_regimes_file is not None:
            factor_regimes_file = Path(factor_regimes_file)
        if sentiment_file is not None:
            sentiment_file = Path(sentiment_file)
        if macro_file is not None:
            macro_file = Path(macro_file)

        data = {}

        # Load returns data
        if returns_file is None:
            returns_file = PROCESSED_DATA_DIR / "returns_monthly.csv"
        if returns_file.exists():
            data["returns"] = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded returns data: {data['returns'].shape}")
        else:
            logger.error(f"Returns file not found: {returns_file}")
            return {}

        # Load factor exposures
        if factor_exposures_file is None:
            factor_exposures_file = PROCESSED_DATA_DIR / "factor_exposures.csv"
        if factor_exposures_file.exists():
            data["factor_exposures"] = pd.read_csv(
                factor_exposures_file, index_col=0, parse_dates=True
            )
            logger.info(f"Loaded factor exposures: {data['factor_exposures'].shape}")
        else:
            logger.warning(f"Factor exposures file not found: {factor_exposures_file}")

        # Load factor regimes
        if factor_regimes_file is None:
            factor_regimes_file = PROCESSED_DATA_DIR / "factor_regimes_hmm.csv"
        if factor_regimes_file.exists():
            data["factor_regimes"] = pd.read_csv(
                factor_regimes_file, index_col=0, parse_dates=True
            )
            logger.info(f"Loaded factor regimes: {data['factor_regimes'].shape}")
        else:
            logger.warning(f"Factor regimes file not found: {factor_regimes_file}")

        # Load sentiment data
        if sentiment_file is None:
            sentiment_file = PROCESSED_DATA_DIR / "sentiment_monthly_normalized.csv"
        if sentiment_file.exists():
            data["sentiment"] = pd.read_csv(
                sentiment_file, index_col=0, parse_dates=True
            )
            logger.info(f"Loaded sentiment data: {data['sentiment'].shape}")
        else:
            logger.warning(f"Sentiment file not found: {sentiment_file}")

        # Load macro data
        if macro_file is None:
            macro_file = PROCESSED_DATA_DIR / "macro_monthly_normalized.csv"
        if macro_file.exists():
            data["macro"] = pd.read_csv(macro_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded macro data: {data['macro'].shape}")
        else:
            logger.warning(f"Macro file not found: {macro_file}")

        return data

    def optimize_portfolio(
        self,
        data: Dict[str, pd.DataFrame],
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
        sector_limits: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Optimize portfolio using specified method.

        Args:
            data: Dictionary with loaded data
            constraints: Portfolio constraints
            target_return: Target annual return
            max_volatility: Maximum annual volatility
            sector_limits: Sector allocation limits

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting portfolio optimization using {self.method} method...")

        if self.method == "black_litterman":
            return self._optimize_black_litterman(
                data, constraints, target_return, max_volatility, sector_limits
            )
        elif self.method == "monte_carlo":
            return self._optimize_monte_carlo(
                data, constraints, target_return, max_volatility, sector_limits
            )
        elif self.method == "combined":
            return self._optimize_combined(
                data, constraints, target_return, max_volatility, sector_limits
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

    def _optimize_black_litterman(
        self,
        data: Dict[str, pd.DataFrame],
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
        sector_limits: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Union[pd.Series, float, Dict]]:
        """Optimize using Black-Litterman method."""
        logger.info("Running Black-Litterman optimization...")

        returns_df = data["returns"]
        factor_exposures = data.get("factor_exposures")
        factor_regimes = data.get("factor_regimes")
        sentiment_scores = data.get("sentiment")

        # Add sector limits to constraints (create shallow copy to avoid mutation)
        if constraints is None:
            constraints = {}
        else:
            constraints = dict(
                constraints
            )  # Shallow copy to avoid mutating caller's dict

        if sector_limits:
            constraints["sector_limits"] = sector_limits

        # Apply BL parameter calibration if needed
        if self.bl_lambda == "auto":
            # Import config for calibration parameters

            from ..config import DEFAULT_BL_CONFIG

            # Parse λ range from CLI or use config default
            if self.bl_lambda_range:
                try:
                    min_lambda, max_lambda = map(float, self.bl_lambda_range.split(","))
                    lambda_range = (min_lambda, max_lambda)
                except ValueError:
                    logger.warning(
                        f"Invalid λ range format: {self.bl_lambda_range}. Using config default."
                    )
                    lambda_range = DEFAULT_BL_CONFIG.lambda_range
            else:
                lambda_range = DEFAULT_BL_CONFIG.lambda_range

            # Calibrate λ using the BL optimizer's method with config parameters
            calibrated_lambda = self.bl_optimizer.calibrate_market_risk_aversion(
                returns_df=returns_df,
                lambda_range=lambda_range,
                n_points=DEFAULT_BL_CONFIG.lambda_points,
            )
            self.bl_optimizer.lambda_mkt = calibrated_lambda
            logger.info(f"Auto-calibrated λ: {calibrated_lambda:.3f}")

        # Set grand view gamma and view strength in the optimizer
        self.bl_optimizer.grand_view_gamma = self.bl_gamma
        # Only set view_strength if the attribute exists
        if hasattr(self.bl_optimizer, "view_strength"):
            self.bl_optimizer.view_strength = self.bl_view_strength

        result = self.bl_optimizer.optimize_portfolio(
            returns_df=returns_df,
            factor_exposures=factor_exposures,
            factor_regimes=factor_regimes,
            sentiment_scores=sentiment_scores,
            constraints=constraints,
            target_return=target_return,
            max_volatility=max_volatility,
        )

        result["method"] = "black_litterman"
        return result

    def _optimize_monte_carlo(
        self,
        data: Dict[str, pd.DataFrame],
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
        sector_limits: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Union[pd.Series, float, Dict]]:
        """Optimize using Monte Carlo method."""
        logger.info("Running Monte Carlo optimization...")

        returns_df = data["returns"]

        # Generate scenarios using simplified single-regime approach
        scenarios = self.mc_optimizer.generate_scenarios(returns_df)

        # Get asset names for sector constraints
        asset_names = list(returns_df.columns)

        # Optimize with constraints
        result = self.mc_optimizer.optimize_with_constraints(
            scenarios,
            constraints=constraints,
            target_return=target_return,
            max_volatility=max_volatility,
            sector_limits=sector_limits,
            asset_names=asset_names,
        )

        # Add method identifier
        result["method"] = "monte_carlo"

        return result

    def _optimize_combined(
        self,
        data: Dict[str, pd.DataFrame],
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
        sector_limits: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Union[pd.Series, float, Dict]]:
        """Combine both optimization methods."""
        logger.info("Running combined optimization...")

        # Run both methods
        bl_result = self._optimize_black_litterman(
            data, constraints, target_return, max_volatility, sector_limits
        )
        mc_result = self._optimize_monte_carlo(
            data, constraints, target_return, max_volatility, sector_limits
        )

        # Ensure weights and expected_return are compatible types
        bl_weights = bl_result["weights"]
        mc_weights = mc_result["weights"]

        def to_array(w):
            if hasattr(w, "values") and not callable(w.values):
                return w.values
            elif isinstance(w, dict):
                return np.array(list(w.values()))
            return np.array(w)

        bl_weights = to_array(bl_weights)
        mc_weights = to_array(mc_weights)
        combined_weights = (bl_weights + mc_weights) / 2

        bl_return = bl_result["expected_return"]
        mc_return = mc_result["expected_return"]
        if isinstance(bl_return, dict) or isinstance(mc_return, dict):
            logger.warning(
                "Expected return is a dict, cannot combine. Using BL result."
            )
            combined_return = (
                bl_return if not isinstance(bl_return, dict) else mc_return
            )
        else:
            combined_return = (bl_return + mc_return) / 2

        # Recompute volatility for combined weights (volatility is nonlinear)
        returns_df = data["returns"]
        try:
            freqstr = returns_df.index.to_period().freqstr
            if freqstr.upper().startswith("M"):
                freq = 12
            elif freqstr.upper().startswith("Q"):
                freq = 4
            elif freqstr.upper().startswith("A") or freqstr.upper().startswith("Y"):
                freq = 1
            else:
                freq = 12
        except Exception:
            freq = 12
        portfolio_returns = (returns_df * combined_weights).sum(axis=1)
        combined_vol = portfolio_returns.std() * np.sqrt(freq)  # Annualized

        # Compute Sharpe only if combined_return is a float
        if isinstance(combined_return, dict):
            logger.warning(
                "Combined return is a dict, cannot compute Sharpe ratio. Setting to np.nan."
            )
            combined_sharpe = np.nan
        else:
            combined_sharpe = (combined_return - self.risk_free_rate) / combined_vol

        result = {
            "method": "combined",
            "weights": combined_weights,
            "expected_return": combined_return,
            "volatility": combined_vol,
            "sharpe_ratio": combined_sharpe,
            "black_litterman": bl_result,
            "monte_carlo": mc_result,
        }

        return result

    def generate_efficient_frontier(
        self,
        data: Dict[str, pd.DataFrame],
        n_points: int = 20,
        constraints: Optional[Dict] = None,
        sector_limits: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate efficient frontier.

        Args:
            data: Dictionary with loaded data
            n_points: Number of frontier points
            constraints: Portfolio constraints
            sector_limits: Sector allocation limits

        Returns:
            Dictionary with frontier data
        """
        logger.info(f"Generating efficient frontier with {n_points} points...")

        if self.method == "monte_carlo":
            returns_df = data["returns"]

            scenarios = self.mc_optimizer.generate_scenarios(returns_df)

            # Get asset names for sector constraints
            asset_names = list(returns_df.columns)

            frontier = self.mc_optimizer.generate_efficient_frontier(
                scenarios, n_points, constraints, asset_names=asset_names
            )
            return frontier
        elif self.method == "black_litterman":
            # Efficient frontier for BL: grid over max_volatility
            logger.info("Generating Black-Litterman efficient frontier...")
            min_vol = 0.05
            max_vol = 0.30
            grid = np.linspace(min_vol, max_vol, n_points)
            results: Dict[str, list] = {
                "returns": [],
                "volatilities": [],
                "weights": [],
            }
            for max_volatility in grid:
                try:
                    res = self._optimize_black_litterman(
                        data, constraints, None, max_volatility, sector_limits
                    )
                    if res and all(
                        k in res for k in ("expected_return", "volatility", "weights")
                    ):
                        results["returns"].append(res["expected_return"])
                        results["volatilities"].append(res["volatility"])
                        weights = res["weights"]
                        if hasattr(weights, "values"):
                            results["weights"].append(weights.values)
                        else:
                            results["weights"].append(weights)
                except Exception as e:
                    logger.warning(
                        f"BL frontier failed for vol={max_volatility:.3f}: {e}"
                    )
            results["returns"] = np.array(results["returns"])
            results["volatilities"] = np.array(results["volatilities"])
            results["weights"] = np.array(results["weights"])
            return results
        else:
            logger.warning(
                "Efficient frontier only available for Monte Carlo and Black-Litterman methods"
            )
            return {}

    def analyze_portfolio_risk(
        self,
        weights: pd.Series,
        data: Dict[str, pd.DataFrame],
        freq: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Analyze portfolio risk metrics.

        Args:
            weights: Portfolio weights
            data: Dictionary with loaded data
            freq: Frequency for annualization

        Returns:
            Dictionary with risk metrics
        """
        logger.info("Analyzing portfolio risk...")

        returns_df = data["returns"]

        # Infer frequency if not provided
        if freq is None:
            try:
                freqstr = returns_df.index.to_period().freqstr
                if freqstr.upper().startswith("M"):
                    freq = 12
                elif freqstr.upper().startswith("Q"):
                    freq = 4
                elif freqstr.upper().startswith("A") or freqstr.upper().startswith("Y"):
                    freq = 1
                else:
                    freq = 12  # Default to monthly
            except Exception:
                freq = 12  # Default to monthly

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Calculate risk metrics
        volatility = portfolio_returns.std() * np.sqrt(freq)  # Annualized
        expected_return = portfolio_returns.mean() * freq  # Annualized
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility

        # Calculate drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calculate VaR and CVaR
        var_95 = portfolio_returns.quantile(0.05)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        # Calculate beta (if market data available)
        if "SPY" in returns_df.columns:
            market_returns = returns_df["SPY"]
            # Use population moments for consistency
            beta = np.cov(portfolio_returns, market_returns, ddof=0)[0, 1] / np.var(
                market_returns, ddof=0
            )
        else:
            beta = np.nan

        risk_metrics = {
            "volatility": volatility,
            "expected_return": expected_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "beta": beta,
        }

        logger.info(f"Risk analysis completed. Sharpe: {sharpe_ratio:.3f}")
        return risk_metrics

    def save_results(
        self,
        results: Dict[str, Union[pd.Series, float, Dict]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Save optimization results.

        Args:
            results: Optimization results
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights = results["weights"]
        weights_file = output_dir / "optimal_weights.csv"
        if hasattr(weights, "to_csv"):
            weights.to_csv(weights_file)
            logger.info(f"Saved optimal weights to {weights_file}")
        else:
            logger.warning("Weights object is not a DataFrame/Series, skipping save.")

        # Save metrics
        metrics = {
            "expected_return": results["expected_return"],
            "volatility": results["volatility"],
            "sharpe_ratio": results["sharpe_ratio"],
            "method": results["method"],
        }

        if "max_drawdown" in results:
            metrics["max_drawdown"] = results["max_drawdown"]
        if "var_95" in results:
            metrics["var_95"] = results["var_95"]

        metrics_df = pd.DataFrame([metrics])
        metrics_file = output_dir / "portfolio_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Saved portfolio metrics to {metrics_file}")

        logger.success("Results saved successfully")


def main():
    """Main function for portfolio optimization."""
    logger.info("Starting portfolio optimization...")

    # Initialize engine
    engine = PortfolioOptimizationEngine(method="combined", random_state=42)

    # Load data
    data = engine.load_data()
    if not data:
        logger.error("Failed to load data")
        return

    # Define constraints
    constraints = {
        "max_weight": 0.25,  # Maximum 25% in any single asset
        "min_weight": 0.01,  # Minimum 1% in any single asset
    }

    # Define sector limits
    sector_limits = {
        "Tech": 0.30,  # Maximum 30% in tech
        "Financials": 0.25,  # Maximum 25% in financials
        "Energy": 0.15,  # Maximum 15% in energy
    }

    # Optimize portfolio
    results = engine.optimize_portfolio(
        data=data,
        constraints=constraints,
        target_return=0.08,  # 8% target return
        max_volatility=0.15,  # 15% maximum volatility
        sector_limits=sector_limits,
    )

    # Analyze risk
    risk_metrics = engine.analyze_portfolio_risk(results["weights"], data)

    # Save results
    engine.save_results(results)

    # Print summary
    logger.info("Portfolio Optimization Summary:")
    logger.info(f"Method: {results['method']}")
    logger.info(f"Expected Return: {results['expected_return']:.3f}")
    logger.info(f"Volatility: {results['volatility']:.3f}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    logger.info(f"Max Drawdown: {risk_metrics['max_drawdown']:.3f}")

    logger.success("Portfolio optimization completed!")


if __name__ == "__main__":
    main()
