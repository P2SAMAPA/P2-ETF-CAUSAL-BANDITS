import numpy as np
import pandas as pd
import lingam
from sklearn.linear_model import LinearRegression

class CausalBandit:
    def __init__(self, data_df, macro_cols, n_thompson=100, exploration_bonus=0.5):
        """
        data_df: DataFrame with columns = ETFs + macro vars.
        macro_cols: list of column names that are macro variables.
        """
        self.data = data_df
        self.macro_cols = macro_cols
        self.etf_cols = [c for c in data_df.columns if c not in macro_cols]
        self.n_thompson = n_thompson
        self.exploration_bonus = exploration_bonus
        self.causal_model = None
        self.causal_effects = {}  # {etf: (mean_effect, variance_effect)}

    def fit_causal_graph(self):
        """Fit LiNGAM model on the entire data."""
        # Ensure data is stationary (differencing macro if needed? We'll use levels)
        X = self.data.values
        model = lingam.DirectLiNGAM()
        model.fit(X)
        self.causal_model = model
        # For each ETF, compute causal effect of macro interventions
        # Using do‑calculus: effect = sum over parents (coefficient * parent)
        # We'll compute the linear coefficients from the LiNGAM adjacency matrix.
        adj = model.adjacency_matrix_  # (n_vars, n_vars), rows cause columns
        # Variable names
        var_names = self.data.columns.tolist()
        # For each ETF, we want the effect from macro variables (and possibly from other ETFs? We'll only consider macro parents for simplicity)
        for etf in self.etf_cols:
            etf_idx = var_names.index(etf)
            parents = [var_names[i] for i in range(adj.shape[0]) if adj[i, etf_idx] != 0 and var_names[i] in self.macro_cols]
            # coefficients
            coefs = [adj[var_names.index(p), etf_idx] for p in parents]
            # We'll store the list of (parent, coef)
            self.causal_effects[etf] = (parents, coefs)
        return self

    def sample_interventional_reward(self, etf, n_samples):
        """
        Sample from the interventional distribution of ETF return when we intervene on its macro parents.
        We assume the data distribution is Gaussian with mean = linear combination of parents.
        The variance is estimated from residuals.
        """
        if etf not in self.causal_effects:
            return np.zeros(n_samples)
        parents, coefs = self.causal_effects[etf]
        if not parents:
            # No macro parents, sample from empirical distribution of the ETF returns
            hist_returns = self.data[etf].values
            return np.random.choice(hist_returns, size=n_samples, replace=True)
        # For each sample, generate parent values from their empirical distribution (or we could fix to current values)
        # We'll use the most recent macro values as the intervention target? No, we want to sample from the interventional distribution.
        # Simpler: sample parent values from their historical distribution, then compute reward = sum(coef * parent) + noise
        parent_vals = self.data[parents].values
        # Sample n_samples rows with replacement
        idx = np.random.choice(len(parent_vals), size=n_samples, replace=True)
        sampled_parents = parent_vals[idx]  # (n_samples, len(parents))
        # Predicted mean
        pred_mean = sampled_parents @ np.array(coefs)
        # Residual variance: compute from data
        true_vals = self.data[etf].values
        # Use same indices to compute residuals
        resid = true_vals[idx] - pred_mean
        noise_std = np.std(resid)
        # Sample noise
        noise = np.random.normal(0, noise_std, n_samples)
        reward = pred_mean + noise
        return reward

    def compute_thompson_score(self, etf):
        """Thompson sampling: expected reward (mean of samples) plus exploration bonus (standard deviation)."""
        samples = self.sample_interventional_reward(etf, self.n_thompson)
        mean = np.mean(samples)
        std = np.std(samples)
        score = mean + self.exploration_bonus * std
        return score, mean, std

    def rank_etfs(self):
        """Return list of ETFs sorted by Thompson score."""
        scores = []
        for etf in self.etf_cols:
            sc, _, _ = self.compute_thompson_score(etf)
            scores.append((etf, sc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
