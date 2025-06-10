import argparse
import asyncio

from config import load_api_keys
from src.api_getter.live_trading import AlpacaTrader
from src.strategies.strategy import find_strategy


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
                        "and sends it to Alpaca")
    
    parser.add_argument("--cancel", 
                        "-c", 
                        action = "store_true", 
                        help = "Cancels orders that is not placed.")

    parser.add_argument("--live",
                        "-l",
                        type = int,
                        default = 5,
                        help = "Specifie the number of iterations for the live loop." \
                        " If you wish to use the default number of iterations, " \
                        "you will not need this argument (run main.py).")

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
    
    if args.cancel:
        await trader.cancel_all_orders()

    if args.order:
        await trader.create_buy_order()

    if args.live:
        interations = int(args.live)
        if interations > 0:
            strategy = find_strategy()
            for _ in range(interations):
                await trader.update(strategy, "ALL")
                await asyncio.sleep(60)  # sleep for a minute

    if args.update:
        symbol = args.update
        strategy = find_strategy()
        await trader.update(strategy, symbol)


if __name__ == "__main__":
    asyncio.run(main())