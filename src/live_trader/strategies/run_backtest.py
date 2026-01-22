from live_trader.strategies.backtest import run_multi_symbol_backtest
from live_trader.config import load_watchlist

from live_trader.strategies.strategies import strategies as old_strategies

import asyncio

TEST_MODE = True  

all_strategies = dict(old_strategies)
all_strategies.pop("rule_based_strategy", None)

financial_strategies = dict(all_strategies)
ml_strategies_list = ["lstm", "bilstm", "tcn", "patchtst", "gnn", "nad", "cnn_gru", "random_forest", "lightgbm", "xgboost", "catboost"]

for strat in ml_strategies_list:
    financial_strategies.pop(strat, None)

ml_strategies = {
    k: v for k, v in all_strategies.items()
    if k in ml_strategies_list
}

symbols = load_watchlist()

async def main(strategies, symbols):
    print("Starting Backtest")
    print("───────────────────────────────────────────────")
    print(f"Symbol: {symbols}")
    print(f"Strategies: {len(financial_strategies)}")
    print(f"Mode: {'TEST' if TEST_MODE else 'LIVE'}\n")
    
    try:
        results = await run_multi_symbol_backtest(
            symbols = symbols,
            strategies=strategies,
            initial_cash=10000,
            test_mode=TEST_MODE,
            days = 60
        )
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n Results type: {type(results)}")
    print(f"Results columns: {results.columns.tolist()}")
    print(f"Results shape: {results.shape}")
    
    if results is not None and not results.empty:
        print("\n Backtest Results:")
        print(results.to_string(index=False))
        
        if 'total_return_pct' in results.columns:
            results = results.sort_values("total_return_pct", ascending=False)
            results.to_csv("backtest_results.csv", index=False)
            
            best = results.iloc[0]
            print("\n Best strategy:")
            print(f"   → {best['strategy']} with {best['total_return_pct']:.2f}% return")
            print("\n Results saved to backtest_results.csv")
        else:
            print("\n No 'total_return_pct' column found!")
            print(f"Available columns: {results.columns.tolist()}")
    else:
        print("\n No results returned from compare_strategies()")

if __name__ == "__main__":
    asyncio.run(main(ml_strategies, symbols))