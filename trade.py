import argparse
import asyncio

from live_trader import AlpacaTrader, Config


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
                        "Only one stock can be specified at a time.")
    
    parser.add_argument("--backtest",
                        "-bt",
                        action = "store_true", 
                        help = "Analyzes the specified position(s) for historical data to determine whether to buy "
                        "or sell based on strategy prompt given by the user. " \
                        "Automatically places an order based on the result. " \
                        "Type 'ALL' (all uppercase) to update all positions. " \
                        "The metrics of the test will be printet to a csv.")
    
    args = parser.parse_args()
    return args


async def main():
    args = parseInput()
    conf = Config("settings.toml")
    trader = AlpacaTrader(conf)

    
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
        await trader.update(symbol)
    
    if args.live:
        await trader.live()

    if args.backtest:
        await trader._run_backtest()

def cli():
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())