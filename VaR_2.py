import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple, Union

class MarketRiskSimulator:
    def __init__(self, raw_df: pd.DataFrame, log_return: List[bool], lmbda: float = 0.95):
        # Reverse descending order to chronological order (oldest to newest)
        self.raw_df = raw_df.iloc[::-1].copy()
        self.log_return = np.array(log_return)
        self.lmbda = lmbda
        
        # Run preprocessing pipeline
        self.rtn_df = self._calculate_returns()
        self.sigma_df = self._calculate_ewma_volatility()
        self.sd_rtn_df = self._standardize_returns()
        
    def _calculate_returns(self) -> pd.DataFrame:
        rtn_dict = {}
        for i, col in enumerate(self.raw_df.columns):
            series = self.raw_df[col]
            if self.log_return[i]:
                # log(P(t)) - log(P(t-1))
                rtn_dict[col] = np.log(series) - np.log(series.shift(1))
            else:
                # P(t) - P(t-1)
                rtn_dict[col] = series - series.shift(1)
        
        return pd.DataFrame(rtn_dict, index=self.raw_df.index).dropna()

    def _calculate_ewma_volatility(self) -> pd.DataFrame:
        # Initialize variance with the variance of the first 20 days or global variance
        v_0 = self.rtn_df.var()
        ewma_var = self.rtn_df.pow(2).copy()
        
        # Apply EWMA recursive filter
        # var(t) = lambda * var(t-1) + (1 - lambda) * r(t)^2
        for i in range(1, len(ewma_var)):
            ewma_var.iloc[i] = self.lmbda * ewma_var.iloc[i-1] + (1 - self.lmbda) * ewma_var.iloc[i]
            
        return np.sqrt(ewma_var)

    def _standardize_returns(self) -> pd.DataFrame:
        return self.rtn_df / self.sigma_df

    def _get_window_data(self, df: pd.DataFrame, N: int, end_date: Union[str, None]) -> pd.DataFrame:
        if end_date is None:
            loc_idx = len(df)
        else:
            loc_idx = df.index.get_loc(end_date) + 1
            
        start_idx = max(0, loc_idx - N)
        return df.iloc[start_idx:loc_idx], df.iloc[loc_idx-1]

    # --- Simulation Methods ---
    
    def simulate_method_1(self, N: int = 250, end_date: str = None) -> pd.DataFrame:
        """Simple lookback of N days using raw historical returns."""
        window_rtn, _ = self._get_window_data(self.rtn_df, N, end_date)
        return window_rtn.reset_index(drop=True)

    def simulate_method_2(self, N: int = 250, end_date: str = None) -> pd.DataFrame:
        """Filtered Historical Simulation using standardized returns times current volatility."""
        window_sd, _ = self._get_window_data(self.sd_rtn_df, N, end_date)
        _, current_sigma = self._get_window_data(self.sigma_df, N, end_date)
        
        return window_sd * current_sigma

    def simulate_method_3(self, N: int = 250, end_date: str = None) -> pd.DataFrame:
        """Pairwise differences of standardized returns normalized by sqrt(2) and scaled by sigma."""
        window_sd, _ = self._get_window_data(self.sd_rtn_df, N, end_date)
        _, current_sigma = self._get_window_data(self.sigma_df, N, end_date)
        
        n_days = len(window_sd)
        pairwise_scenarios = []
        
        # Extract internal matrix for faster processing
        matrix = window_sd.values 
        
        # Sample all combinations i < j (where t_i is chronologically before t_j)
        for i, j in combinations(range(n_days), 2):
            diff = (matrix[j] - matrix[i]) / np.sqrt(2)
            pairwise_scenarios.append(diff)
            
        sim_df = pd.DataFrame(pairwise_scenarios, columns=self.rtn_df.columns)
        return sim_df * current_sigma

    # --- Risk Metrics Framework ---

    def calculate_var_es(self, scenarios: pd.DataFrame, percentile: float) -> Tuple[pd.Series, pd.Series]:
        """Calculates VaR and Expected Shortfall per column for given scenarios."""
        alpha = 1 - percentile
        var_series = pd.Series(index=scenarios.columns, dtype=float)
        es_series = pd.Series(index=scenarios.columns, dtype=float)
        
        for col in scenarios.columns:
            sorted_scen = scenarios[col].sort_values(ascending=True)
            
            # Find VaR using the lower alpha-quantile
            var_val = sorted_scen.quantile(alpha, interpolation='linear')
            var_series[col] = var_val
            
            # Expected Shortfall: Mean of all scenario losses worse than or equal to VaR
            tail_losses = sorted_scen[sorted_scen <= var_val]
            es_series[col] = tail_losses.mean() if not tail_losses.empty else var_val
            
        return var_series, es_series

    def calculate_rolling_risk(self, method: int, window: int = 250, 
                               var_pct: float = 0.99, es_pct: float = 0.975) -> pd.DataFrame:
        """Generates historical time series of risk metrics."""
        sim_map = {1: self.simulate_method_1, 2: self.simulate_method_2, 3: self.simulate_method_3}
        sim_func = sim_map[method]
        
        results = []
        # Loop through available dates that have at least 'window' history
        for i in range(window, len(self.rtn_df) + 1):
            current_date = self.rtn_df.index[i-1]
            
            # Run simulation for specific historical target date
            scen = sim_func(N=window, end_date=current_date)
            
            # Calculate metrics
            var_s, _ = self.calculate_var_es(scen, var_pct)
            _, es_s = self.calculate_var_es(scen, es_pct)
            
            # Format outputs
            for col in self.rtn_df.columns:
                results.append({
                    'Date': current_date,
                    'Asset': col,
                    f'VaR_{var_pct}': var_s[col],
                    f'ES_{es_pct}': es_s[col]
                })
                
        return pd.DataFrame(results).set_index(['Date', 'Asset'])