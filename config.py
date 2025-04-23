from dotenv import load_dotenv
import os
import toml

load_dotenv()

# LAGE ARGUMENTS FOR Å SNAKKE MED ORDERS

def load_api_keys(config_file: str = "config.toml") -> tuple:
    """
    Load API keys from a TOML config file, with fallback to environment variables.

    Args:
        config_file (str): Path to the TOML config file.

    Returns:
        tuple: (ALPACA_KEY, ALPACA_SECRET_KEY)
    """

    alpaca_key = os.getenv("ALPACA_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")

    try:
        with open(config_file, "rb") as file:  # "rb" for tomli, use "r" for toml
            conf = toml.load(file)
            keys = conf.get("keys", {})
            alpaca_key = keys.get("alpaca_key", alpaca_key)
            alpaca_secret = keys.get("alpaca_secret_key", alpaca_secret)

    except FileNotFoundError:
        print(f"Config file not found: {config_file}, falling back to environment variables.")

    except Exception:
        print(f"Error reading config, using environment variables as fallback.")

    return alpaca_key, alpaca_secret

        
if __name__ == "__main__":
    print(load_api_keys())