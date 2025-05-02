from config import load_api_keys
from src.api_getter.live_trading import AlpacaTrader

def main():
    key, secret_key = load_api_keys()
    trader = AlpacaTrader(key, secret_key)

    # If argument = strategy: velg strategy basert på svar.
        # Man skal kunne velge hvilken strategy som skal generere et signal
        # signalen lager en order som skal sendes til trader, f.eks.
        # trader.update(rsa())

    

if __name__ == "__main__":
    main()