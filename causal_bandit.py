import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

class CausalBandit:
    def __init__(self, data_df, macro_cols, n_thompson=100, exploration_bonus=0.5):
        self.data = data_df
        self.macro_cols = [c for c in macro_cols if c in data_df.columns]  # only existing
        self.etf_cols = [c for c in data_df.columns if c not in self.macro_cols]
        self.n_thompson = n_thompson
        self.exploration_bonus = exploration_bonus
        self.causal_model = None
        self.causal_effects = {}  # {etf: (parents, coefs)} or empty if no macro

    def fit_causal_graph(self):
        """Try LiNGAM if there are macro columns and enough samples, else fallback."""
        if len(self.macro_cols) == 0 or len(self.data) < len(self.macro_cols) + len(self.etf_cols) + 10:
            # Fallback: no causal model, use empirical mean and variance
            return self
        try:
            import lingam
            X = self.data.values
            model = lingam.DirectLiNGAM()
            model.fit(X)
            self.causal_model = model
            adj = model.adjacency_matrix_
            var_names = self.data.columns.tolist()
            for etf in self.etf_cols:
                etf_idx = var_names.index(etf)
                parents = []
                coefs = []
                for i, pname in enumerate(var_names):
                    if adj[i, etf_idx] != 0 and pname in self.macro_cols:
                        parents.append(pname)
                        coefs.append(adj[i, etf_idx])
                self.causal_effects[etf] = (parents, coefs)
        except Exception:
            # Fallback to simple empirical Thompson sampling
            pass
        return self

    def sample_interventional_reward(self, etf, n_samples):
        """Return Thompson samples for the ETF."""
        if etf not in self.etf_cols:
            return np.zeros(n_samples)
        # If we have a causal effect with macro parents, use it
        if etf in self.causal_effects:
            parents, coefs = self.causal_effects[etf]
            if parents:
                # Sample parent values from their empirical distribution
                parent_vals = self.data[parents].values
                idx = np.random.choice(len(parent_vals), size=n_samples, replace=True)
                sampled_parents = parent_vals[idx]
                pred_mean = sampled_parents @ np.array(coefs)
                # Residual variance
                actual = self.data[etf].values
                resid = actual[idx] - pred_mean
                noise_std = max(np.std(resid), 1e-6)
                noise = np.random.normal(0, noise_std, n_samples)
                return pred_mean + noise
        # Fallback: empirical Thompson sampling (sample from historical returns)
        hist_returns = self.data[etf].dropna().values
        if len(hist_returns) == 0:
            return np.zeros(n_samples)
        return np.random.choice(hist_returns, size=n_samples, replace=True)

    def compute_thompson_score(self, etf):
        samples = self.sample_interventional_reward(etf, self.n_thompson)
        mean = np.mean(samples)
        std = np.std(samples)
        score = mean + self.exploration_bonus * std
        return score, mean, std

    def rank_etfs(self):
        scores = []
        for etf in self.etf_cols:
            sc, _, _ = self.compute_thompson_score(etf)
            scores.append((etf, sc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
