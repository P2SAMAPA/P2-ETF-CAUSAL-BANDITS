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
        if combined.empty:
            print("  No data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Ensure macro columns exist
        available_macro = [c for c in config.MACRO_COLUMNS if c in combined.columns]
        if not available_macro:
            print("  No macro columns found – skipping universe")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Store best per ETF across windows: ticker -> (best_score, best_window)
        best_per_etf = {}

        for win in config.WINDOWS:
            if len(combined) < win + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            train_data = combined.iloc[-win:]
            bandit = CausalBandit(train_data, available_macro,
                                  n_thompson=config.N_THOMPSON_SAMPLES,
                                  exploration_bonus=config.EXPLORATION_BONUS)
            try:
                bandit.fit_causal_graph()
            except Exception as e:
                print(f"  Failed to fit causal graph for window {win}d: {e}")
                continue
            # Compute Thompson scores for all ETFs in this universe (that are in train_data)
            scores = bandit.rank_etfs()   # list of (etf, score)
            # Update best per ETF
            for etf, score in scores:
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)
            print(f"  Window {win}d: top ETF {scores[0][0]} with score {scores[0][1]:.4f}")

        if not best_per_etf:
            print("  No valid predictions across windows")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Rank by best score descending
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, (score, win) in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "thompson_score": float(score), "best_window": win})
            full_scores[ticker] = {"score": float(score), "best_window": win}
        print(f"  Top 3 ETFs by best Thompson score across windows: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
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
