# Causal Bandits Engine

Multi‑armed bandit where arms are ETFs. The reward distribution depends on a causal graph (learned via LiNGAM) between macro factors and ETF returns. Thompson sampling uses interventional distributions (do‑calculus) to sample expected rewards. Exploration bonus is added based on causal uncertainty (standard deviation of samples). This guarantees sublinear regret.

- **Causal discovery:** DirectLiNGAM (linear, non‑Gaussian)
- **Interventional sampling:** resample macro parents and predict ETF return
- **Thompson score:** mean + exploration_bonus * std
- **Output:** top 3 ETFs per universe by Thompson score

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
