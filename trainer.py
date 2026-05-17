import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from causal_bandit import CausalBandit

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Causal Bandits) ===")
        combined = data_manager.prepare_combined_data(df, tickers)
        if combined.empty or len(combined) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(combined) < win + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            train_data = combined.iloc[-win:]

            # Available macro columns (only those present in the data)
            available_macro = [c for c in config.MACRO_COLUMNS if c in train_data.columns]
            if not available_macro:
                print(f"    No macro columns available for window {win}d – using empirical Thompson sampling")
                # Use empty macro list; CausalBandit will fallback to empirical sampling
                available_macro = []

            bandit = CausalBandit(train_data, available_macro,
                                  n_thompson=config.N_THOMPSON_SAMPLES,
                                  exploration_bonus=config.EXPLORATION_BONUS)
            bandit.fit_causal_graph()
            rankings = bandit.rank_etfs()
            win_scores = {etf: score for etf, score in rankings}
            window_results[win] = win_scores
            for etf, score in win_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            # Fallback: use mean return of last 252 days
            for etf in tickers:
                if etf in returns.columns:  # need returns, not combined; we have returns elsewhere
                    pass
            # Simpler: fallback to zero but we will use historical returns below
            # We'll compute historical means from the returns DataFrame
            returns = data_manager.prepare_returns_matrix(df, tickers)
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (mean_ret, 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        # Store full scores for all ETFs
        full_scores = {ticker: {"score": score, "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "thompson_score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/causal_bandit_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Causal Bandits Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
