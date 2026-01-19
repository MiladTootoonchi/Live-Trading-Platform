from live_trader.strategies.backtest import run_multi_symbol_backtest
from live_trader.config import load_watchlist

from live_trader.strategies.strategies import strategies as old_strategies

import asyncio

TEST_MODE = True  

strategies = dict(old_strategies)
strategies.pop("rule_based_strategy", None)
symbols = load_watchlist()

async def main():
    print("Starting Backtest")
    print("───────────────────────────────────────────────")
    print(f"Symbol: {symbols}")
    print(f"Strategies: {len(strategies)}")
    print(f"Mode: {'TEST' if TEST_MODE else 'LIVE'}\n")
    
    try:
        results = await run_multi_symbol_backtest(
            symbols = symbols,
            strategies=strategies,
            days=80,
            initial_cash=10000,
            test_mode=TEST_MODE
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
    asyncio.run(main())