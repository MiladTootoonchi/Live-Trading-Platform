import argparse
import asyncio

from config import load_api_keys
from src.api_getter.live_trading import AlpacaTrader
from src.strategies import strategy


def parseInput():
    """
    Makes the arguments for running different methods.
    """
    parser = argparse.ArgumentParser(description = "Put the Alpaca keys in settings.toml and type:\n" \
    "")

    parser.add_argument("--order", 
                        "-o", 
                        action = "store_true", 
                        help = "places an order. The program will ask questions about the order " \
                        "and send it to Alpaca")
    
    parser.add_argument("--cancel", 
                        "-c", 
                        action = "store_true", 
                        help = "Cancels orders that is not placed.")

    parser.add_argument("--live",
                        "-l",
                        help = "Activates live trading. Loops through updater multiple times with" \
                        "different strategies chosen by ML. Sells after given number og iterations.")

    parser.add_argument("--update",
                        "-u",
                        help = "Analyzes the specified position(s) to determine whether to buy "
                        "or sell based on current strategy. " \
                        "Automatically places an order based on the result. " \
                        "Leave blank to update all positions. " \
                        "Only one position can be specified at a time.")
    
    args = parser.parse_args()
    return args



async def main():
    args = parseInput()

    key, secret_key = load_api_keys()
    trader = AlpacaTrader(key, secret_key)
    
    if args.cancel:
        await trader.cancel_all_orders()

    if args.order:
        await trader.create_buy_order()

    if args.live:
        interations = int(args.live)
        if interations > 0:
            for _ in range(interations):
                await trader.update(strategy.rule_based_strategy)
                await asyncio.sleep(60)  # sleep for a minute

    if args.update:
        symbol = args.update
        await trader.update(strategy.rule_based_strategy, symbol)


if __name__ == "__main__":
    asyncio.run(main())