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

    parser.add_argument("--strategy", 
                        "-s", 
                        help = "Executes the specified strategy for the live trading " \
                        "If the strategy does not exist or is not specified, " \
                        "the rule based strategy will run as default.",
                        default = strategy.rule_based_strategy)
    
    parser.add_argument("--cancel", 
                        "-c", 
                        action = "store_true", 
                        help = "Cancels orders that is not placed.")

    parser.add_argument("--live",
                        "-l",
                        help = "Activates live trading. Loops through updater multiple times with" \
                        "different strategies chosen by ML. Sells after given number og iterations.")
    
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
                await asyncio.sleep(86400)  # sleep for a day
    

if __name__ == "__main__":
    asyncio.run(main())