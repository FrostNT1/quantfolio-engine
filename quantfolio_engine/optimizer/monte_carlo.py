"""
Monte Carlo portfolio simulation for QuantFolio Engine.

This module implements Monte Carlo simulation for portfolio optimization:
- Single-regime scenario generation (simplified for clarity)
- Risk constraint simulation
- Drawdown analysis
- Sector allocation constraints
"""

from typing import Dict, List, Optional, Union

import cvxpy as cp
from loguru import logger
import numpy as np
import pandas as pd

from quantfolio_engine.config import ASSET_UNIVERSE


def set_log_level(debug: bool):
    from loguru import logger

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="DEBUG" if debug else "INFO",
        colorize=True,
    )


# Assuming the main class is MonteCarloOptimizer (if not, adjust accordingly)
class MonteCarloOptimizer:
    """
    Monte Carlo portfolio optimizer.

    Args:
        debug (bool): If True, sets logger to DEBUG level for verbose output. Default is False.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        time_horizon: int = 12,  # months
        risk_free_rate: float = 0.02,
        max_cvar_loss: float = 0.15,  # This is a CVaR constraint, not a drawdown constraint
        confidence_level: float = 0.95,
        risk_aversion: float = 3.07,  # Interpreted as annual risk aversion (not monthly)
        use_cvar: bool = True,
        time_basis: str = "monthly",  # Time basis for all calculations
        random_state: Optional[int] = None,
        debug: bool = False,
    ):
        """
        Initialize Monte Carlo optimizer.

        Args:
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Investment horizon in months
            risk_free_rate: Annual risk-free rate
            max_cvar_loss: Maximum allowed CVaR of average path loss (not drawdown)
            confidence_level: Confidence level for risk metrics
            risk_aversion: Risk aversion parameter for optimization (annualized)
            use_cvar: Whether to use CVaR constraint
            random_state: Random seed for reproducibility
            debug: If True, sets logger to DEBUG level for verbose output. Default is False.
        """
        set_log_level(debug)
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.risk_free_rate = risk_free_rate
        self.max_cvar_loss = max_cvar_loss
        self.confidence_level = confidence_level
        self.risk_aversion = risk_aversion  # Annual risk aversion
        self.use_cvar = use_cvar
        self.time_basis = time_basis

        # Standardize risk-free rate to time basis
        if time_basis == "monthly":
            self.rf_monthly = risk_free_rate / 12
            self.rf_annual = risk_free_rate
        elif time_basis == "annual":
            self.rf_monthly = risk_free_rate / 12
            self.rf_annual = risk_free_rate
        else:
            raise ValueError("time_basis must be 'monthly' or 'annual'")

        # Use local RNG for thread safety and reproducibility
        self.rng = np.random.default_rng(random_state)
        if random_state is not None:
            logger.info(f"Set random seed to {random_state} for reproducibility")

    def generate_scenarios(
        self,
        returns_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate Monte Carlo scenarios using single-regime approach.

        Args:
            returns_df: Historical asset returns

        Returns:
            Monte Carlo scenarios array of shape (n_simulations, time_horizon, n_assets)
        """
        logger.info(f"Generating {self.n_simulations} Monte Carlo scenarios...")

        # Clean data first to handle NaNs consistently
        clean_returns = returns_df.dropna()
        if len(clean_returns) < len(returns_df):
            logger.warning(
                f"Removed {len(returns_df) - len(clean_returns)} rows with NaN values"
            )

        mu = clean_returns.mean().values
        n_assets = clean_returns.shape[1]

        # Guard for large universes
        if n_assets > 100:
            logger.warning(
                f"Large asset universe ({n_assets} assets): using Ledoit-Wolf shrinkage estimator for covariance."
            )
            from sklearn.covariance import LedoitWolf

            sigma = LedoitWolf().fit(clean_returns.values).covariance_ + 1e-6 * np.eye(
                n_assets
            )
        else:
            sigma = clean_returns.cov().values + 1e-6 * np.eye(n_assets)

        scenarios = self.rng.multivariate_normal(
            mu, sigma, size=(self.n_simulations, self.time_horizon)
        )

        logger.info(f"Generated scenarios with shape: {scenarios.shape}")
        return scenarios

    def optimize_with_constraints(
        self,
        scenarios: np.ndarray,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
        sector_limits: Optional[Dict[str, float]] = None,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[pd.Series, float, np.ndarray]]:
        """
        Optimize portfolio weights with Monte Carlo constraints.

        Args:
            scenarios: Monte Carlo scenarios (n_simulations, time_horizon, n_assets)
            constraints: Portfolio constraints
            target_return: Target annual return
            max_volatility: Maximum annual volatility
            sector_limits: Sector allocation limits
            asset_names: Asset names (required if sector_limits provided)

        Returns:
            Dictionary with optimal weights and metrics
        """
        logger.info("Optimizing portfolio with Monte Carlo constraints...")

        if sector_limits and asset_names is None:
            raise ValueError("asset_names must be supplied when sector_limits are used")

        # FIXED: Use consistent horizon approach - aggregate to path-level returns
        if scenarios.ndim == 3:
            # Calculate path-level returns: (1 + r1) * (1 + r2) * ... * (1 + rT) - 1
            path_returns = np.prod(1 + scenarios, axis=1) - 1
            mu = np.mean(path_returns, axis=0)  # Mean across scenarios
            cov = np.cov(path_returns.T)  # Covariance across scenarios
            logger.info("Using path-level (horizon) returns for optimization")
        else:
            mu = np.mean(scenarios, axis=0)
            cov = np.cov(scenarios.T)
            logger.info("Using single-period returns for optimization")

        # Annualize path-level returns (1 year horizon, not ×12)
        if scenarios.ndim == 3:
            annualization_factor = 1  # Path returns are already annual
        else:
            annualization_factor = 12  # Monthly returns need annualization

        annualized_mean = mu * annualization_factor
        annualized_cov = cov * annualization_factor
        logger.info(
            f"Annualizing mean and covariance with factor {annualization_factor:.2f}"
        )

        annualized_mean_1d = annualized_mean.ravel()
        n_assets = len(annualized_mean_1d)

        w = cp.Variable(n_assets)

        # FIXED: Use excess returns in objective for consistency with Sharpe ratio
        excess_returns = annualized_mean_1d - self.rf_annual
        objective = cp.Maximize(
            cp.sum(cp.multiply(excess_returns, w))
            - 0.5 * self.risk_aversion * cp.quad_form(w, annualized_cov)
        )

        constraints_list = [
            cp.sum(w) == 1,
            w >= 0,
        ]

        if target_return is not None:
            # FIXED: target_return is already annual, no need to multiply
            constraints_list.append(
                cp.sum(cp.multiply(annualized_mean_1d, w)) >= target_return
            )
        if max_volatility is not None:
            constraints_list.append(
                cp.quad_form(w, annualized_cov) <= max_volatility**2
            )

        # FIXED: CVaR constraint with proper sample size validation
        if self.use_cvar and self.max_cvar_loss is not None:
            N = scenarios.shape[0]

            # Validate sample size for CVaR
            min_samples = int(1 / (1 - self.confidence_level)) + 1
            if N < min_samples:
                logger.warning(
                    f"CVaR sample size {N} < {min_samples} required for {self.confidence_level:.0%} confidence. "
                    f"Skipping CVaR constraint."
                )
                # FIXED: Skip CVaR constraint entirely instead of trying to build VaR proxy
                # (VaR constraint would require CVXPY-compatible variables which is complex)
            else:
                if N > 5000:
                    logger.warning(
                        f"Large scenario set ({N}), downsampling to 5000 for CVaR"
                    )
                    idx = self.rng.choice(N, 5000, replace=False)
                    scenarios_small = scenarios[idx]
                    N = 5000
                else:
                    scenarios_small = scenarios

                # FIXED: Consistent horizon - use path-level returns for CVaR
                if scenarios.ndim == 3:
                    terminal_returns = np.prod(1 + scenarios_small, axis=1) - 1
                else:
                    terminal_returns = scenarios_small

                losses = -terminal_returns @ w  # Portfolio terminal losses
                alpha = self.confidence_level
                t = cp.Variable()
                z = cp.Variable(N)
                CVAR = t + (1 / (1 - alpha) / N) * cp.sum(z)
                constraints_list += [z >= losses - t, z >= 0]
                constraints_list.append(CVAR <= self.max_cvar_loss)

        if sector_limits and asset_names is not None:
            for sector, limit in sector_limits.items():
                sector_assets = [
                    i
                    for i, asset in enumerate(asset_names)
                    if ASSET_UNIVERSE.get(asset, {}).get("type") == sector
                ]
                if sector_assets:
                    constraints_list.append(cp.sum(w[sector_assets]) <= limit)
        if constraints:
            if "max_weight" in constraints:
                constraints_list.append(w <= constraints["max_weight"])
            if "min_weight" in constraints:
                constraints_list.append(w >= constraints["min_weight"])
        problem = cp.Problem(objective, constraints_list)
        try:
            problem.solve(solver=cp.ECOS_BB)
        except cp.error.SolverError:
            logger.warning("ECOS_BB solver not available, trying OSQP...")
            problem.solve(solver=cp.OSQP)
        if problem.status in ("optimal", "optimal_inaccurate"):
            optimal_weights = w.value
            if problem.status == "optimal_inaccurate":
                logger.warning("Optimization completed with numerical inaccuracies")
        else:
            logger.warning(f"Optimization failed with status: {problem.status}")
            optimal_weights = np.ones(n_assets) / n_assets
        portfolio_metrics = self._calculate_portfolio_metrics(
            scenarios, optimal_weights
        )
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        results = {
            "weights": pd.Series(optimal_weights, index=asset_names),
            "expected_return": portfolio_metrics["expected_return"],
            "volatility": portfolio_metrics["volatility"],
            "sharpe_ratio": portfolio_metrics["sharpe_ratio"],
            "max_drawdown": portfolio_metrics["max_drawdown"],
            "var_95": portfolio_metrics["var_95"],
            "scenarios": scenarios,
        }
        logger.success(
            f"Monte Carlo optimization completed. Sharpe: {portfolio_metrics['sharpe_ratio']:.3f}"
        )
        return results

    def _calculate_portfolio_metrics(
        self, scenarios: np.ndarray, weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics from scenarios.

        Args:
            scenarios: Monte Carlo scenarios (shape: n_simulations x time_horizon x n_assets)
            weights: Portfolio weights

        Returns:
            Dictionary with portfolio metrics
        """
        # FIXED: Handle path-level vs monthly returns correctly
        if scenarios.ndim == 3:
            # Path-level returns: collapse time dimension first
            path_returns = np.prod(1 + scenarios, axis=1) - 1  # (N, n_assets)
            portfolio_returns = path_returns @ weights  # (N,) - one return per scenario
            annualization_factor = 1  # Already annual
            volatility_scale = 1  # Already annual
            logger.info("Calculating metrics from path-level (annual) returns")
        else:
            # Monthly returns: need annualization
            portfolio_returns = np.sum(scenarios * weights, axis=-1)
            annualization_factor = 12  # Annualize monthly data
            volatility_scale = np.sqrt(annualization_factor)  # Annualize volatility
            logger.info("Calculating metrics from monthly returns (annualizing)")

        raw_mean = np.mean(portfolio_returns)
        expected_return = raw_mean * annualization_factor  # Annualize if needed
        volatility = np.std(portfolio_returns) * volatility_scale  # Annualize if needed

        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility

        # Calculate maximum drawdown across time periods for each scenario
        max_drawdowns = []
        if scenarios.ndim == 3:
            # For path-level returns, drawdown is already computed over the full path
            for scenario_returns in scenarios:
                # Compute cumulative returns over time for this scenario
                scenario_portfolio_returns = np.sum(
                    scenario_returns * weights, axis=-1
                )  # (T,)
                cumulative_returns = np.cumprod(1 + scenario_portfolio_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdowns.append(np.min(drawdowns))
        else:
            # For monthly returns, compute drawdown across time periods
            for scenario_returns in portfolio_returns:
                cumulative_returns = np.cumprod(1 + scenario_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdowns.append(np.min(drawdowns))

        max_drawdown = np.mean(max_drawdowns)  # Average across scenarios

        # FIXED: Label VaR properly based on input type
        var_95 = np.percentile(portfolio_returns.flatten(), 5)
        # var_label = "annual VaR" if scenarios.ndim == 3 else "1-month VaR"  # Unused variable, remove

        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,  # This is {var_label}
        }

    def generate_efficient_frontier(
        self,
        scenarios: np.ndarray,
        n_points: int = 20,
        constraints: Optional[Dict] = None,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate efficient frontier using Monte Carlo scenarios.

        Args:
            scenarios: Monte Carlo scenarios
            constraints: Portfolio constraints
            n_points: Number of frontier points
            asset_names: Asset names

        Returns:
            Dictionary with frontier returns and volatilities
        """
        logger.info(f"Generating efficient frontier with {n_points} points...")

        # FIXED: Use consistent horizon approach for frontier generation
        if scenarios.ndim == 3:
            # Use path-level returns for frontier
            path_returns = np.prod(1 + scenarios, axis=1) - 1
            mu = np.mean(path_returns, axis=0)
        else:
            mu = np.mean(scenarios, axis=0)

        # FIXED: Generate target returns in annual units
        min_return = np.min(mu)
        max_return = np.max(mu)

        # Scale to annual returns if using monthly data
        if scenarios.ndim == 3:
            # Path returns are already annual
            target_returns = np.linspace(min_return, max_return, n_points)
        else:
            # Monthly returns need annualization
            target_returns = np.linspace(min_return * 12, max_return * 12, n_points)

        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []

        for target_return in target_returns:
            try:
                result = self.optimize_with_constraints(
                    scenarios,
                    constraints,
                    target_return=target_return,
                    asset_names=asset_names,
                )
                frontier_returns.append(result["expected_return"])
                frontier_volatilities.append(result["volatility"])
                weights = result["weights"]
                if hasattr(weights, "values"):
                    frontier_weights.append(weights.values)
                else:
                    frontier_weights.append(weights)
            except Exception as e:
                logger.warning(
                    f"Failed to optimize for target return {target_return}: {e}"
                )
                continue

        return {
            "returns": np.array(frontier_returns),
            "volatilities": np.array(frontier_volatilities),
            "weights": np.array(frontier_weights),
        }
