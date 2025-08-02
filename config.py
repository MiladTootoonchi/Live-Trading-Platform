from dotenv import load_dotenv
import os
import toml
import logging
import os

load_dotenv()

def make_logger():
    log_dir = "logfiles"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_name = os.path.join(log_dir, "live_trading")

    logger = logging.getLogger(file_name)
    handler = logging.FileHandler(f"{file_name}.log", mode = "a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s -%(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

logger = make_logger()



def load_strategy_name(config_file: str = "settings.toml") -> str:
    """
    Load the name of the strategy the user want to use for the live trading.
    If the strategy is not given, the program will ask for an input.

    Returns:
        str: The name of the strategy:
    """

    try:
        with open(config_file, "r") as file:
            conf = toml.load(file)
            live = conf.get("live", {})
            strategy = live.get("strategy", strategy)
        
    except Exception:
        strategy = input("Which strategy do you want to use? ")

    return strategy


def load_api_keys(config_file: str = "settings.toml") -> tuple:
    """
    Load API keys from a TOML config file, with fallback to environment variables.

    Args:
        config_file (str): Path to the TOML config file.

    Returns:
        tuple: (ALPACA_KEY, ALPACA_SECRET_KEY)
    """

    alpaca_key = None
    alpaca_secret = None

    try:
        with open(config_file, "r") as file:
            conf = toml.load(file)
            keys = conf.get("keys", {})
            alpaca_key = keys.get("alpaca_key", alpaca_key)
            alpaca_secret = keys.get("alpaca_secret_key", alpaca_secret)

    except Exception:
        logger.info(f"Config file not found: {config_file}, falling back to environment variables.\n")
        alpaca_key = os.getenv("ALPACA_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")

    if not alpaca_key or not alpaca_secret:
        print("Missing Alpaca API credentials. Provide them in the config file or as environment variables.")
        
    return alpaca_key, alpaca_secret

        
if __name__ == "__main__":
    print(load_api_keys())