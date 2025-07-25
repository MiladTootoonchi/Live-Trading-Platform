from dotenv import load_dotenv
import os
import toml

load_dotenv()

def load_api_keys(config_file: str = "settings.toml") -> tuple:
    """
    Load API keys from a TOML config file, with fallback to environment variables.

    Args:
        config_file (str): Path to the TOML config file.

    Returns:
        tuple: (ALPACA_KEY, ALPACA_SECRET_KEY)
    """

    try:
        alpaca_key = os.getenv("ALPACA_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    except: # Can not find the .env file, or the right variables in the file.
        pass

    try:
        with open(config_file, "r") as file:
            conf = toml.load(file)
            keys = conf.get("keys", {})
            alpaca_key = keys.get("alpaca_key", alpaca_key)
            alpaca_secret = keys.get("alpaca_secret_key", alpaca_secret)

    except FileNotFoundError:
        print(f"Config file not found: {config_file}, falling back to environment variables.\n")

    except Exception:
        print(f"Error reading config, using environment variables as fallback.\n")

    return alpaca_key, alpaca_secret

        
if __name__ == "__main__":
    print(load_api_keys())