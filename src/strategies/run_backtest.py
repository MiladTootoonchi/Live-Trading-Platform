from src.strategies.backtest import compare_strategies
from src.strategies.bollinger_bands_strategy import bollinger_bands_strategy
from src.strategies.macd import macd_strategy
from src.strategies.mean_reversion import mean_reversion_strategy
from src.strategies.momentum import momentum_strategy
from src.strategies.moving_average_strategy import moving_average_strategy
from src.strategies.rsi import rsi_strategy
import pandas as pd

TEST_MODE = True  

strategies = {
    'Bollinger Bands': bollinger_bands_strategy,
    'MACD': macd_strategy,
    'Mean Reversion': mean_reversion_strategy,
    'Momentum': momentum_strategy,
    'MA': moving_average_strategy,
    'RSI': rsi_strategy,
}

def main():
    print("Starting Backtest")
    print("───────────────────────────────────────────────")
    print(f"Symbol: TSLA")
    print(f"Strategies: {len(strategies)}")
    print(f"Mode: {'TEST' if TEST_MODE else 'LIVE'}\n")
    
    try:
        results = compare_strategies(
            symbol='TSLA',
            strategies=strategies,
            days=30,
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
    main()