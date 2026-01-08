from dotenv import load_dotenv
import os
import toml
import logging

load_dotenv()

def make_logger():
    """
    Creates and configures a logger that writes INFO-level messages to a file.

    The logger writes to 'logfiles/live_trading.log', creating the directory if it 
    doesn't exist. Log messages are formatted with a timestamp, log level, and message.
    
    Returns:
        logging.Logger: Configured logger instance ready for use.
    """
    
    log_dir = "logfiles"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_name = os.path.join(log_dir, "live_trading")

    logger = logging.getLogger(file_name)

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        handler = logging.FileHandler(f"{file_name}.log", mode = "a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

logger = make_logger()



def load_strategy_name(config_file: str = "settings.toml") -> str:
    """
    Load the name of the strategy the user want to use for the live trading.
    If the strategy is not given, the program will ask for an input.

    Args:
        config_file (str): Path to the TOML config file.

    Returns:
        str: The name of the strategy:
    """

    try:
        with open(config_file, "r") as file:
            conf = toml.load(file)
            live = conf.get("live", {})
            strategy = live.get("strategy")

    except FileNotFoundError:
        logger.info(f"Config file not found: {config_file}, falling back to environment variables.")
        strategy = None

    except Exception:
        logger.info(f"Could not find strategy from {config_file}, falling back to environment variables.")
        strategy = None

    if not strategy:
        strategy = os.getenv("strategy")

    if not strategy:
        logger.critical("Strategy name missing from both config and environment variables.")
        raise RuntimeError("Missing strategy name")
 
    return strategy


def load_watchlist(config_file: str = "settings.toml") -> list[str]:
    """
    Load the list of stocks the program is going to watch over whenever it updates posistions.

    Args:
        config_file (str): Path to the TOML config file.

    Returns:
        list[str]: The watchlist with the symbols of the stocks as strings:
    """

    try:
        with open(config_file, "r") as file:
            conf = toml.load(file)
            live = conf.get("live", {})
            watchlist = live.get("watchlist")

            if watchlist:
                return watchlist

    except Exception:
        logger.info(
            f"Could not find watchlist in {config_file}, falling back to environment variables.\n"
        )

    # Fallback to env var
    env_watchlist = os.getenv("watchlist")
    if env_watchlist:
        return env_watchlist

    logger.warning("Watchlist not found; defaulting to empty list.\n")
    return []


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

    except FileNotFoundError:
        logger.info(f"Config file not found: {config_file}, falling back to environment variables.")

    except Exception:
        logger.info(f"Could not find Alpaca API credentials in {config_file}, falling back to environment variables.")

    if not alpaca_key:
        alpaca_key = os.getenv("alpaca_key")

    if not alpaca_secret:
        alpaca_secret = os.getenv("alpaca_secret_key")

    if not alpaca_key or not alpaca_secret:
        logger.critical("Missing Alpaca API credentials. Provide them in the config file or as environment variables.")
        raise RuntimeError("Missing Alpaca API credentials")
        
    return alpaca_key, alpaca_secret

        
if __name__ == "__main__":
    info = f"""
---INFO---
keys:        {load_api_keys()}, 
strategy:    {load_strategy_name()}, 
watchlist:   {load_watchlist()}
----------
"""
    
    print(info)