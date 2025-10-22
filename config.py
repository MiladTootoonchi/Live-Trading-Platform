from dotenv import load_dotenv
import os
import toml
import logging
import os
from pathlib import Path

# Force load the .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

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

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
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

    Args:
        config_file (str): Path to the TOML config file.

    Returns:
        str: The name of the strategy:
    """

    try:
        with open(config_file, "r") as file:
            conf = toml.load(file)
            live = conf.get("live", {})
            strategy = live.get("strategy", None)
        
    except Exception:
        logger.info(f"Could not find strategy name in {config_file}, falling back to environment variables.\n")
        strategy = os.getenv("strategy")
        if strategy == None:
            logger.info(f"strategy name string missing from environment variables")
 
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

    except FileNotFoundError:
        logger.info(f"Config file not found: {config_file}, falling back to environment variables.\n")
        alpaca_key = os.getenv("alpaca_key")
        alpaca_secret = os.getenv("alpaca_secret_key")

    except Exception:
        logger.info(f"Could not find Alpaca API credentials in {config_file}, falling back to environment variables.\n")
        alpaca_key = os.getenv("alpaca_key")
        alpaca_secret = os.getenv("alpaca_secret_key")

    if not alpaca_key or not alpaca_secret:
        print("Missing Alpaca API credentials. Provide them in the config file or as environment variables.")
        
    return alpaca_key, alpaca_secret

        
if __name__ == "__main__":
    print(load_api_keys())