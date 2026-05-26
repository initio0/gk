from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class SimulationConfig:
    """
    Configuration for the market risk simulation system.
    """
    lookback_days: int = 250
    end_date: Optional[pd.Timestamp] = None
    lambda_ewma: float = 0.95
    pairwise_sample_size: int = 1000
    var_alpha: float = 0.01      # 99% VaR
    es_alpha: float = 0.025      # 97.5% ES (can be different from var_alpha)


# ============================================================
# 2. Returns and volatility
# ============================================================

class ReturnCalculator:
    """
    Compute returns from raw price/level data using log or absolute returns.
    """

    def compute_returns(
        self,
        raw_df: pd.DataFrame,
        log_return: np.ndarray
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        raw_df : pd.DataFrame
            Historical time series, index = dates (descending), columns = factors.
        log_return : array-like of bool
            Boolean flags per column; True => log return, False => absolute return.

        Returns
        -------
        rtn_df : pd.DataFrame
            Returns with same columns as raw_df and index shifted by one (first row dropped).
        """
        if len(log_return) != raw_df.shape[1]:
            raise ValueError("log_return length must match number of columns in raw_df")

        log_return = np.asarray(log_return, dtype=bool)
        cols = raw_df.columns
        idx = raw_df.index

        # We assume index is descending; returns are P(t) - P(t-1) in that order
        # So we use values[:-1] (later dates) minus values[1:] (earlier dates)
        data = raw_df.values
        rtn_data = np.empty((data.shape[0] - 1, data.shape[1]), dtype=float)

        for i in range(data.shape[1]):
            series = data[:, i]
            if log_return[i]:
                rtn = np.log(series[:-1]) - np.log(series[1:])
            else:
                rtn = series[:-1] - series[1:]
            rtn_data[:, i] = rtn

        rtn_df = pd.DataFrame(rtn_data, index=idx[1:], columns=cols)
        return rtn_df


class VolatilityEstimator:
    """
    Compute EWMA local volatilities and standardized returns.
    """

    def compute_ewma_sigma(
        self,
        rtn_df: pd.DataFrame,
        lambda_ewma: float = 0.95
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        rtn_df : pd.DataFrame
            Returns data.
        lambda_ewma : float
            EWMA decay factor.

        Returns
        -------
        sigma_df : pd.DataFrame
            Local volatilities (same shape as rtn_df).
        """
        if not (0.0 < lambda_ewma < 1.0):
            raise ValueError("lambda_ewma must be in (0, 1)")

        sigma_df = pd.DataFrame(index=rtn_df.index, columns=rtn_df.columns, dtype=float)

        for col in rtn_df.columns:
            r = rtn_df[col].values.astype(float)
            if len(r) == 0:
                sigma_df[col] = np.nan
                continue

            sigma2 = np.zeros_like(r, dtype=float)

            # Initialize with sample variance of first few points or first squared return
            if len(r) > 1:
                init_window = min(20, len(r))
                sigma2[0] = np.var(r[:init_window], ddof=1)
            else:
                sigma2[0] = r[0] ** 2

            for t in range(1, len(r)):
                sigma2[t] = lambda_ewma * sigma2[t - 1] + (1.0 - lambda_ewma) * (r[t] ** 2)

            sigma_df[col] = np.sqrt(sigma2)

        return sigma_df

    def standardize_returns(
        self,
        rtn_df: pd.DataFrame,
        sigma_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        rtn_df : pd.DataFrame
            Returns data.
        sigma_df : pd.DataFrame
            Local volatilities.

        Returns
        -------
        sd_rtn_df : pd.DataFrame
            Standardized returns (r / sigma).
        """
        if rtn_df.shape != sigma_df.shape:
            raise ValueError("rtn_df and sigma_df must have the same shape")

        # Avoid division by zero
        sigma_safe = sigma_df.replace(0.0, np.nan)
        sd_rtn_df = rtn_df / sigma_safe

        return sd_rtn_df


# ============================================================
# 3. Scenario generation
# ============================================================

class ScenarioGenerator:
    """
    Generate scenarios using different methods based on returns and standardized returns.
    """

    def _select_window(
        self,
        df: pd.DataFrame,
        lookback_days: int,
        end_date: Optional[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Select a lookback window of length lookback_days ending at end_date.
        Assumes index is dates in descending order.
        """
        if df.empty:
            return df

        if end_date is None:
            end_date = df.index[0]

        # Ensure end_date is in index or find closest earlier date
        if end_date not in df.index:
            valid_dates = df.index[df.index <= end_date]
            if len(valid_dates) == 0:
                raise ValueError("end_date is before earliest date in df")
            end_date = valid_dates[0]

        # Because index is descending, slice by position
        end_pos = df.index.get_loc(end_date)
        start_pos = end_pos + lookback_days
        start_pos = min(start_pos, len(df))  # cap at length
        window = df.iloc[end_pos:start_pos]

        return window

    def _latest_sigma_vector(
        self,
        sigma_df: pd.DataFrame,
        end_date: Optional[pd.Timestamp]
    ) -> pd.Series:
        """
        Get the most recent sigma vector at end_date (or nearest earlier date).
        """
        if sigma_df.empty:
            raise ValueError("sigma_df is empty")

        if end_date is None:
            return sigma_df.iloc[0]

        if end_date in sigma_df.index:
            return sigma_df.loc[end_date]

        valid_dates = sigma_df.index[sigma_df.index <= end_date]
        if len(valid_dates) == 0:
            raise ValueError("end_date is before earliest date in sigma_df")

        return sigma_df.loc[valid_dates[0]]

    # ---------- Method 1: Simple lookback ----------

    def generate_scenarios_simple_lookback(
        self,
        rtn_df: pd.DataFrame,
        lookback_days: int,
        end_date: Optional[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Method 1: Simple lookback of one year using raw returns.

        Returns
        -------
        scenarios_df : pd.DataFrame
            Each row is a scenario equal to the realized return vector on that day.
        """
        return self._select_window(rtn_df, lookback_days, end_date)

    # ---------- Method 2: Standardized returns rescaled by latest sigma ----------

    def generate_scenarios_rescaled_sd(
        self,
        sd_rtn_df: pd.DataFrame,
        sigma_df: pd.DataFrame,
        lookback_days: int,
        end_date: Optional[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Method 2: Simple lookback of one year using standardized returns
        multiplied by the most recent sigma.
        """
        window_sd = self._select_window(sd_rtn_df, lookback_days, end_date)
        latest_sigma = self._latest_sigma_vector(sigma_df, end_date)

        # Broadcast multiplication
        scenarios_df = window_sd.multiply(latest_sigma, axis=1)
        return scenarios_df

    # ---------- Method 3: Pairwise differences (sampled) + antithetic ----------

    def generate_scenarios_pairwise_diff(
        self,
        sd_rtn_df: pd.DataFrame,
        sigma_df: pd.DataFrame,
        lookback_days: int,
        end_date: Optional[pd.Timestamp],
        sample_size: int = 1000,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Method 3: Pairwise combinations of dates (t_i < t_j) in standardized returns.
        Scenario = (z(t_j) - z(t_i)) / sqrt(2) * latest_sigma.

        Only a random sample of M pairs is used (default 1000), then antithetic
        scenarios are added by flipping the sign of each scenario.
        """
        window_sd = self._select_window(sd_rtn_df, lookback_days, end_date)
        latest_sigma = self._latest_sigma_vector(sigma_df, end_date)

        if window_sd.shape[0] < 2:
            return pd.DataFrame(columns=window_sd.columns)

        # Ensure chronological order for t_i < t_j
        # window_sd is descending; reverse to ascending for clarity
        window_sd_chrono = window_sd.iloc[::-1]
        dates = window_sd_chrono.index.to_list()
        z_values = window_sd_chrono.values
        n = len(dates)

        # Total number of possible pairs
        total_pairs = n * (n - 1) // 2
        if total_pairs == 0:
            return pd.DataFrame(columns=window_sd.columns)

        # Determine how many pairs to actually use
        m = min(sample_size, total_pairs)

        rng = np.random.default_rng(random_state)

        # Enumerate all pairs (i < j) if total_pairs is small; otherwise sample indices
        # Efficient sampling of pair indices:
        # Map a linear index k in [0, total_pairs) to (i, j) if needed.
        # But for simplicity and clarity, we can prebuild all pairs if not huge.
        if total_pairs <= sample_size * 5:  # heuristic threshold
            all_pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
            sampled_pairs = rng.choice(len(all_pairs), size=m, replace=False)
            chosen_pairs = [all_pairs[k] for k in sampled_pairs]
        else:
            # Sample pairs directly by rejection sampling until we have m unique pairs
            chosen_pairs_set = set()
            while len(chosen_pairs_set) < m:
                i = rng.integers(0, n - 1)
                j = rng.integers(i + 1, n)
                chosen_pairs_set.add((i, j))
            chosen_pairs = list(chosen_pairs_set)

        sqrt2 = np.sqrt(2.0)
        scenarios = []

        sigma_vec = latest_sigma.values.astype(float)

        for (i, j) in chosen_pairs:
            z_diff = (z_values[j] - z_values[i]) / sqrt2
            r_scenario = z_diff * sigma_vec
            scenarios.append(r_scenario)
            # Antithetic scenario
            scenarios.append(-r_scenario)

        scenarios_arr = np.array(scenarios)
        # Build an index label if desired; here we just use integer index
        scenarios_df = pd.DataFrame(
            data=scenarios_arr,
            columns=window_sd.columns
        )

        return scenarios_df


# ============================================================
# 4. Risk metrics
# ============================================================

class RiskCalculator:
    """
    Compute P&L, VaR, and ES from scenario returns.
    """

    def compute_pnl_from_scenarios(
        self,
        scenarios_df: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """
        Parameters
        ----------
        scenarios_df : pd.DataFrame
            Scenario returns (rows = scenarios, columns = factors).
        weights : np.ndarray
            Portfolio weights (length = number of columns).

        Returns
        -------
        pnl_series : pd.Series
            Scenario P&L values.
        """
        if scenarios_df.shape[1] != len(weights):
            raise ValueError("weights length must match number of columns in scenarios_df")

        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        pnl = scenarios_df.values @ w
        pnl_series = pd.Series(pnl.flatten(), index=scenarios_df.index, name="PnL")
        return pnl_series

    def compute_var(
        self,
        pnl_series: pd.Series,
        alpha: float
    ) -> float:
        """
        Compute VaR at tail probability alpha (e.g. 0.01 for 99% VaR).
        Returns positive loss number.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        q = pnl_series.quantile(alpha)
        var_value = -float(q)
        return var_value

    def compute_es(
        self,
        pnl_series: pd.Series,
        alpha: float
    ) -> float:
        """
        Compute Expected Shortfall (ES) at tail probability alpha.
        Returns positive loss number.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        q = pnl_series.quantile(alpha)
        tail = pnl_series[pnl_series <= q]
        if len(tail) == 0:
            return 0.0
        es_value = -float(tail.mean())
        return es_value

    def compute_var_es(
        self,
        pnl_series: pd.Series,
        var_alpha: float,
        es_alpha: float
    ) -> Dict[str, float]:
        """
        Compute both VaR and ES, possibly at different tail probabilities.

        Returns
        -------
        result : dict
            {'VaR': float, 'ES': float}
        """
        var_val = self.compute_var(pnl_series, var_alpha)
        es_val = self.compute_es(pnl_series, es_alpha)
        return {"VaR": var_val, "ES": es_val}


# ============================================================
# 5. Rolling historical VaR/ES
# ============================================================

class RollingRiskCalculator:
    """
    Compute rolling VaR and ES over historical periods.
    """

    def compute_rolling_var_es_from_returns(
        self,
        rtn_df: pd.DataFrame,
        weights: np.ndarray,
        window_size: int,
        var_alpha: float,
        es_alpha: float
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        rtn_df : pd.DataFrame
            Realized returns (descending index).
        weights : np.ndarray
            Portfolio weights.
        window_size : int
            Rolling window length in days.
        var_alpha : float
            Tail probability for VaR.
        es_alpha : float
            Tail probability for ES.

        Returns
        -------
        rolling_df : pd.DataFrame
            Index = dates (end of each window), columns = ['VaR', 'ES'].
        """
        if rtn_df.shape[1] != len(weights):
            raise ValueError("weights length must match number of columns in rtn_df")

        risk_calc = RiskCalculator()
        w = np.asarray(weights, dtype=float)

        dates = rtn_df.index
        n = len(dates)

        var_list = []
        es_list = []
        out_dates = []

        # Iterate over windows in descending index order
        for start_pos in range(0, n - window_size + 1):
            end_pos = start_pos + window_size
            window = rtn_df.iloc[start_pos:end_pos]

            # Compute P&L for each day in the window
            pnl = window.values @ w
            pnl_series = pd.Series(pnl, index=window.index)

            var_val = risk_calc.compute_var(pnl_series, var_alpha)
            es_val = risk_calc.compute_es(pnl_series, es_alpha)

            # End date of window is the first row (since descending)
            out_dates.append(window.index[0])
            var_list.append(var_val)
            es_list.append(es_val)

        rolling_df = pd.DataFrame(
            {
                "VaR": var_list,
                "ES": es_list
            },
            index=pd.Index(out_dates, name="date")
        )

        return rolling_df


# ============================================================
# 6. Orchestration engine
# ============================================================

class MarketRiskEngine:
    """
    High-level façade to run the full market risk simulation pipeline.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.return_calc = ReturnCalculator()
        self.vol_estimator = VolatilityEstimator()
        self.scenario_gen = ScenarioGenerator()
        self.risk_calc = RiskCalculator()
        self.rolling_risk_calc = RollingRiskCalculator()

    # ---------- Data preparation ----------

    def prepare_data(
        self,
        raw_df: pd.DataFrame,
        log_return: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        From raw data to returns, local volatilities, and standardized returns.

        Returns
        -------
        rtn_df, sigma_df, sd_rtn_df : tuple of DataFrames
        """
        rtn_df = self.return_calc.compute_returns(raw_df, log_return)
        sigma_df = self.vol_estimator.compute_ewma_sigma(
            rtn_df,
            lambda_ewma=self.config.lambda_ewma
        )
        sd_rtn_df = self.vol_estimator.standardize_returns(rtn_df, sigma_df)
        return rtn_df, sigma_df, sd_rtn_df

    # ---------- Scenario generation ----------

    def generate_scenarios(
        self,
        rtn_df: pd.DataFrame,
        sd_rtn_df: pd.DataFrame,
        sigma_df: pd.DataFrame,
        method: str,
        sample_size: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate scenarios using the specified method.

        Parameters
        ----------
        method : {'simple', 'rescaled_sd', 'pairwise_diff'}

        Returns
        -------
        scenarios_df : pd.DataFrame
        """
        lb = self.config.lookback_days
        end = self.config.end_date

        if method == "simple":
            return self.scenario_gen.generate_scenarios_simple_lookback(
                rtn_df, lb, end
            )
        elif method == "rescaled_sd":
            return self.scenario_gen.generate_scenarios_rescaled_sd(
                sd_rtn_df, sigma_df, lb, end
            )
        elif method == "pairwise_diff":
            if sample_size is None:
                sample_size = self.config.pairwise_sample_size
            return self.scenario_gen.generate_scenarios_pairwise_diff(
                sd_rtn_df,
                sigma_df,
                lb,
                end,
                sample_size=sample_size,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown scenario generation method: {method}")

    # ---------- VaR / ES for scenarios ----------

    def compute_var_es_for_scenarios(
        self,
        scenarios_df: pd.DataFrame,
        weights: np.ndarray,
        var_alpha: Optional[float] = None,
        es_alpha: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute VaR and ES for a given set of scenarios and portfolio weights.

        Returns
        -------
        result : dict
            {'VaR': float, 'ES': float}
        """
        if var_alpha is None:
            var_alpha = self.config.var_alpha
        if es_alpha is None:
            es_alpha = self.config.es_alpha

        pnl_series = self.risk_calc.compute_pnl_from_scenarios(scenarios_df, weights)
        return self.risk_calc.compute_var_es(pnl_series, var_alpha, es_alpha)

    # ---------- Rolling VaR / ES ----------

    def compute_rolling_var_es(
        self,
        rtn_df: pd.DataFrame,
        weights: np.ndarray,
        window_size: Optional[int] = None,
        var_alpha: Optional[float] = None,
        es_alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compute rolling VaR and ES from realized returns.

        Returns
        -------
        rolling_df : pd.DataFrame
            Index = dates, columns = ['VaR', 'ES'].
        """
        if window_size is None:
            window_size = self.config.lookback_days
        if var_alpha is None:
            var_alpha = self.config.var_alpha
        if es_alpha is None:
            es_alpha = self.config.es_alpha

        return self.rolling_risk_calc.compute_rolling_var_es_from_returns(
            rtn_df=rtn_df,
            weights=weights,
            window_size=window_size,
            var_alpha=var_alpha,
            es_alpha=es_alpha
        )


# ============================================================
# 7. Example usage sketch (can be removed in production)
# ============================================================

if __name__ == "__main__":
    # Example synthetic data to illustrate usage
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")[::-1]  # descending
    raw_df_example = pd.DataFrame(
        {
            "Factor1": 100 + np.cumsum(np.random.normal(0, 1, 300)),
            "Factor2": 50 + np.cumsum(np.random.normal(0, 0.5, 300)),
            "Factor3": 200 + np.cumsum(np.random.normal(0, 2, 300)),
        },
        index=dates
    )
    log_return_flags = np.array([True, True, False])

    config = SimulationConfig(
        lookback_days=250,
        lambda_ewma=0.95,
        pairwise_sample_size=1000,
        var_alpha=0.01,
        es_alpha=0.025
    )
    engine = MarketRiskEngine(config)

    # Prepare data
    rtn_df, sigma_df, sd_rtn_df = engine.prepare_data(raw_df_example, log_return_flags)

    # Portfolio weights
    weights_example = np.array([1.0, -0.5, 0.3])

    # Generate scenarios for each method and compute VaR/ES
    for method in ["simple", "rescaled_sd", "pairwise_diff"]:
        scenarios_df = engine.generate_scenarios(
            rtn_df=rtn_df,
            sd_rtn_df=sd_rtn_df,
            sigma_df=sigma_df,
            method=method,
            random_state=123
        )
        res = engine.compute_var_es_for_scenarios(scenarios_df, weights_example)
        print(f"Method: {method:13s} | VaR: {res['VaR']:.4f} | ES: {res['ES']:.4f}")

    # Rolling historical VaR/ES from realized returns
    rolling_df = engine.compute_rolling_var_es(rtn_df, weights_example)
    print("\nRolling VaR/ES (head):")
    print(rolling_df.head())

