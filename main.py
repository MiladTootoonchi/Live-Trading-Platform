import argparse
import asyncio

from config import load_api_keys
from live_trader import AlpacaTrader, find_strategy


def parseInput():
    """
    Makes the arguments for running different methods.
    """
    parser = argparse.ArgumentParser(description = "Put the Alpaca keys in settings.toml and type:\n" \
    "")

    parser.add_argument("--buy", 
                        "-b", 
                        action = "store_true", 
                        help = "places an order. The program will ask questions about the order " \
                        "and sends it to Alpaca")
    
    parser.add_argument("--sell", 
                        "-s", 
                        action = "store_true", 
                        help = "places an order. The program will ask questions about the order " \
                        "and sends it to Alpaca")
    
    parser.add_argument("--cancel", 
                        "-c", 
                        action = "store_true", 
                        help = "Cancels orders that is not placed.")
    
    parser.add_argument("--cancel-last",
                        "-cl",
                        action = "store_true",
                        help = "Cancels the most recently submitted open order.")

    parser.add_argument("--live",
                        "-l",
                        action = "store_true", 
                        help = "Activates the live loop. " \
                        "The loop runs until user presses Ctrl + c. " \
                        "Updates every positions for ALL symbols.")

    parser.add_argument("--update",
                        "-u",
                        help = "Analyzes the specified position(s) to determine whether to buy "
                        "or sell based on strategy prompt given by the user. " \
                        "Automatically places an order based on the result. " \
                        "Type 'ALL' (all uppercase) to update all positions. " \
                        "Only one position can be specified at a time.")
    
    args = parser.parse_args()
    return args


async def main():
    args = parseInput()
    key, secret_key = load_api_keys()
    trader = AlpacaTrader(key, secret_key)

    
    if args.cancel_last:
        await trader.cancel_last_order()
    if args.cancel:
        await trader.cancel_all_orders()
    if args.sell:
        await trader.create_sell_order()
    if args.buy:
        await trader.create_buy_order()


    if args.update:
        symbol = args.update
        strategy = find_strategy()
        await trader.update(strategy, symbol)
    
    if args.live:
        strategy = find_strategy()
        try:
            while True:
                await trader.update(strategy, "ALL")
                await asyncio.sleep(60) # sleep for a minute

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nShutting down... ")

        finally:
            pass

def cli():
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())