def simulate_method_3(self, N: int = 250, end_date: str = None, M: int = 1000) -> pd.DataFrame:
        """
        Pairwise differences of standardized returns sampled randomly M times,
        augmented with antithetic scenarios (flipped signs) to yield 2M total scenarios.
        """
        window_sd, _ = self._get_window_data(self.sd_rtn_df, N, end_date)
        _, current_sigma = self._get_window_data(self.sigma_df, N, end_date)
        
        n_days = len(window_sd)
        matrix = window_sd.values 
        
        # Calculate total possible unique pairs (i < j)
        total_possible_pairs = n_days * (n_days - 1) // 2
        actual_M = min(M, total_possible_pairs)
        
        # Efficiently sample unique pair indices without generating all combinations in memory
        sampled_indices = set()
        rng = np.random.default_rng() # Uses a modern numpy random generator
        
        while len(sampled_indices) < actual_M:
            # Generate random pairs of indices
            i_idx = rng.integers(0, n_days, size=actual_M - len(sampled_indices))
            j_idx = rng.integers(0, n_days, size=actual_M - len(sampled_indices))
            
            for i, j in zip(i_idx, j_idx):
                if i < j:
                    sampled_indices.add((i, j))
                elif j < i:
                    sampled_indices.add((j, i))
                    
        # Generate the base sampled scenarios
        base_scenarios = []
        for i, j in sampled_indices:
            diff = (matrix[j] - matrix[i]) / np.sqrt(2)
            base_scenarios.append(diff)
            
        base_scenarios = np.array(base_scenarios)
        
        # Apply Antithetic Variates: Flip the signs to double the scenarios
        antithetic_scenarios = -base_scenarios
        
        # Combine base and antithetic scenarios -> Shape: (2 * actual_M, n_assets)
        total_scenarios = np.vstack([base_scenarios, antithetic_scenarios])
        
        # Scale by current volatility and convert back to DataFrame
        sim_df = pd.DataFrame(total_scenarios, columns=self.rtn_df.columns)
        return sim_df * current_sigma