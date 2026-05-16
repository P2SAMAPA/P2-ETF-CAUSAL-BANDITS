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
        if combined.empty or len(combined) < config.WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Use last WINDOW days
        train_data = combined.iloc[-config.WINDOW:]

        # Ensure all macro columns are present; if not, skip or impute
        missing_macro = [c for c in config.MACRO_COLUMNS if c not in train_data.columns]
        if missing_macro:
            print(f"  Missing macro columns: {missing_macro} – using only available")
            # Continue with available macro columns
            available_macro = [c for c in config.MACRO_COLUMNS if c in train_data.columns]
            config.MACRO_COLUMNS = available_macro  # update locally

        # Fit causal bandit
        bandit = CausalBandit(train_data, config.MACRO_COLUMNS,
                              n_thompson=config.N_THOMPSON_SAMPLES,
                              exploration_bonus=config.EXPLORATION_BONUS)
        bandit.fit_causal_graph()
        rankings = bandit.rank_etfs()

        top_etfs = []
        full_scores = {}
        for etf, score in rankings[:config.TOP_N]:
            top_etfs.append({"ticker": etf, "thompson_score": float(score)})
        for etf, score in rankings:
            full_scores[etf] = float(score)

        print(f"  Top 3 ETFs by Thompson score: {[e['ticker'] for e in top_etfs]}")
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
    print("\n=== Causal Bandits Engine complete ===")

if __name__ == "__main__":
    main()
