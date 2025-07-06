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


class MonteCarloOptimizer:
    """
    Monte Carlo portfolio optimization with single-regime scenarios.

    Simulates portfolio performance under a single regime and optimizes
    for risk-adjusted returns with constraints.
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
        random_state: Optional[int] = None,
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
        """
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.risk_free_rate = risk_free_rate
        self.max_cvar_loss = max_cvar_loss
        self.confidence_level = confidence_level
        self.risk_aversion = risk_aversion  # Annual risk aversion
        self.use_cvar = use_cvar

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

        mu = returns_df.mean().values
        n_assets = returns_df.shape[1]
        # Guard for large universes
        if n_assets > 100:
            logger.warning(
                f"Large asset universe ({n_assets} assets): using Ledoit-Wolf shrinkage estimator for covariance."
            )
            from sklearn.covariance import LedoitWolf

            sigma = LedoitWolf().fit(returns_df.values).covariance_ + 1e-6 * np.eye(
                n_assets
            )
        else:
            sigma = returns_df.cov().values + 1e-6 * np.eye(n_assets)

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

        if scenarios.ndim == 3:
            mu = np.mean(scenarios, axis=(0, 1))
            cov = np.cov(scenarios.reshape(-1, scenarios.shape[-1]).T)
        else:
            mu = np.mean(scenarios, axis=0)
            cov = np.cov(scenarios.T)

        annualization_factor = 12
        annualized_mean = mu * annualization_factor
        annualized_cov = cov * annualization_factor
        logger.info(
            f"Annualizing mean and covariance with factor {annualization_factor:.2f}"
        )

        annualized_mean_1d = annualized_mean.ravel()
        n_assets = len(annualized_mean_1d)

        w = cp.Variable(n_assets)

        # Risk aversion is now interpreted as annual
        objective = cp.Maximize(
            cp.sum(cp.multiply(annualized_mean_1d, w))
            - 0.5 * self.risk_aversion * cp.quad_form(w, annualized_cov)
        )

        constraints_list = [
            cp.sum(w) == 1,
            w >= 0,
        ]

        if target_return is not None:
            constraints_list.append(
                cp.sum(cp.multiply(annualized_mean_1d, w)) >= target_return
            )
        if max_volatility is not None:
            constraints_list.append(
                cp.quad_form(w, annualized_cov) <= max_volatility**2
            )
        # CVaR constraint (not drawdown)
        if self.use_cvar and self.max_cvar_loss is not None:
            N = scenarios.shape[0]
            if N > 5000:
                logger.warning(
                    f"Large scenario set ({N}), downsampling to 5000 for CVaR"
                )
                idx = self.rng.choice(N, 5000, replace=False)
                scenarios_small = scenarios[idx]
                N = 5000
            else:
                scenarios_small = scenarios
            scenario_returns = np.mean(scenarios_small, axis=1)
            losses = -scenario_returns @ w
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
        # Portfolio returns for each scenario and time period
        # Shape: (n_simulations, time_horizon)
        portfolio_returns = np.sum(scenarios * weights, axis=-1)

        # Calculate metrics across all scenarios and time periods
        # Always annualize monthly data to annual
        annualization_factor = 12
        volatility_scale = np.sqrt(annualization_factor)

        raw_mean = np.mean(portfolio_returns)
        expected_return = raw_mean * annualization_factor  # Annualize
        volatility = np.std(portfolio_returns) * volatility_scale  # Annualize

        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility

        # Calculate maximum drawdown across time periods for each scenario
        max_drawdowns = []
        for scenario_returns in portfolio_returns:
            cumulative_returns = np.cumprod(1 + scenario_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdowns.append(np.min(drawdowns))

        max_drawdown = np.mean(max_drawdowns)  # Average across scenarios

        # Calculate VaR across all returns
        var_95 = np.percentile(portfolio_returns.flatten(), 5)

        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
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

        # Handle 3D scenarios: (n_simulations, time_horizon, n_assets)
        if scenarios.ndim == 3:
            mu = np.mean(scenarios, axis=(0, 1))
        else:
            mu = np.mean(scenarios, axis=0)

        # Generate target returns
        min_return = np.min(mu)
        max_return = np.max(mu)
        target_returns = np.linspace(min_return, max_return, n_points)

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
