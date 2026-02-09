from dotenv import load_dotenv
import os
import toml
import logging

load_dotenv()


class Config:

    def __init__(self, config_file_path: str = "settings.toml"):
        self._config_file = config_file_path
        self._logger = self._make_logger()
        self._alpaca_key, self._alpaca_secret = self._load_api_keys()

        self.watchlist = self._load_watchlist()
        self.strategy_name = self._load_strategy_name()
        self.apca_url = "https://paper-api.alpaca.markets"

        self._macd_stabilization = self.load_ml_var("macd_slow") * 3

    # ----- logging -----

    def log_info(self, info: str):
        self._logger.info(info)

    def log_error(self, error: str):
        self._logger.error(error)

    def log_warning(self, warning: str):
        self._logger.warning(warning)

    def log_critical(self, message: str):
        self._logger.critical(message)

    def log_debug(self, message: str):
        self._logger.debug(message)

    def log_expectation(self, message: str):
        self._logger.exception(message)

    def _make_logger(self):
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



    def _load_strategy_name(self) -> str:
        """
        Load the name of the strategy the user want to use for the live trading.
        If the strategy is not given, the program will ask for an input.

        Returns:
            str: The name of the strategy:
        """

        try:
            with open(self._config_file, "r") as file:
                conf = toml.load(file)
                live = conf.get("live", {})
                strategy = live.get("strategy")

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file}, falling back to environment variables.")
            strategy = None

        except Exception:
            self.log_info(f"Could not find strategy from {self._config_file}, falling back to environment variables.")
            strategy = None

        if not strategy:
            strategy = os.getenv("strategy")

        if not strategy:
            self.log_critical("Strategy name missing from both config and environment variables.")
            raise RuntimeError("Missing strategy name")
    
        return strategy



    # ----- watchlist -----
    @staticmethod
    def _normalize_watchlist(value) -> list[str]:
        if not value:
            return []

        # Already a list (correct TOML)
        if isinstance(value, list):
            return [
                str(symbol).strip().upper()
                for symbol in value
                if str(symbol).strip()
            ]

        # Env var or misconfigured TOML → comma-separated string
        if isinstance(value, str):
            return [
                symbol.strip().upper()
                for symbol in value.split(",")
                if symbol.strip()
            ]

        raise TypeError("Watchlist must be a list or a comma-separated string")

    def _load_watchlist(self):
        """
        Load the list of stocks the program is going to watch over whenever it updates posistions.
        """

        try:
            with open(self._config_file, "r") as file:
                conf = toml.load(file)
                live = conf.get("live", {})
                watchlist = live.get("watchlist")

                normalized = self._normalize_watchlist(watchlist)
                if normalized:
                    return normalized

        except Exception:
            self.log_info(
                f"Could not find watchlist in {self._config_file}, falling back to environment variables.\n"
            )

        # Fallback to env var
        env_watchlist = os.getenv("watchlist")
        normalized = self._normalize_watchlist(env_watchlist)
        if normalized:
            return normalized

        self.log_warning("Watchlist not found; defaulting to empty list.\n")
        return []



    # ----- api_keys -----

    def _load_api_keys(self) -> tuple:
        """
        Load API keys from a TOML config file, with fallback to environment variables.

        Returns:
            tuple: (ALPACA_KEY, ALPACA_SECRET_KEY)
        """

        alpaca_key = None
        alpaca_secret = None

        try:
            with open(self._config_file, "r") as file:
                conf = toml.load(file)
                keys = conf.get("keys", {})
                alpaca_key = keys.get("alpaca_key", alpaca_key)
                alpaca_secret = keys.get("alpaca_secret_key", alpaca_secret)

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file}, falling back to environment variables.")

        except Exception:
            self.log_info(f"Could not find Alpaca API credentials in {self._config_file}, falling back to environment variables.")

        if not alpaca_key:
            alpaca_key = os.getenv("alpaca_key")

        if not alpaca_secret:
            alpaca_secret = os.getenv("alpaca_secret_key")

        if not alpaca_key or not alpaca_secret:
            self.log_critical("Missing Alpaca API credentials. Provide them in the config file or as environment variables.")
            raise RuntimeError("Missing Alpaca API credentials")
            
        return alpaca_key, alpaca_secret
    
    def load_keys(self):
        return (self._alpaca_key, self._alpaca_secret)
    

    def _load_strategy_list(self):
        """
        Load the list of strategies the program is going to use for backtesting.
        """

        try:
            with open(self._config_file, "r") as file:
                conf = toml.load(file)
                backtesting = conf.get("backtesting", {})
                list = backtesting.get("strategy_list")

                normalized = self._normalize_watchlist(list)
                if normalized:
                    return normalized

        except Exception:
            self.log_info(
                f"Could not find strategy_list in {self._config_file}, falling back to environment variables.\n"
            )

        # Fallback to env var
        env_list = os.getenv("strategy_list")
        normalized = self._normalize_watchlist(env_list)
        if normalized:
            return normalized

        self.log_warning("strategy_list not found; defaulting to empty list.\n")
        return []
    

    def _load_initial_cash(self) -> int:
        """
        Load the name of the strategy the user want to use for the live trading.
        If the strategy is not given, the program will ask for an input.

        Returns:
            int: The initial_cash:
        """

        try:
            with open(self._config_file, "r") as file:
                conf = toml.load(file)
                backtesting = conf.get("backtesting", {})
                cash = backtesting.get("initial_cash")

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file}, falling back to environment variables.")
            cash = None

        except Exception:
            self.log_info(f"Could not find initial_cash from {self._config_file}, falling back to environment variables.")
            cash = None

        if not cash:
            cash = os.getenv("initial_cash")

        if not cash:
            self.log_critical("initial_cash missing from both config and environment variables.")
            raise RuntimeError("Missing initial cash")
    
        return int(cash)


    def _load_days(self) -> int:
        """
        Total amount of days the user wants to use for backtesting

        Returns:
            int: days.
        """

        try:
            with open(self._config_file, "r") as file:
                conf = toml.load(file)
                backtesting = conf.get("backtesting", {})
                days = backtesting.get("backtesting_days")

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file}, falling back to environment variables.")
            days = None

        except Exception:
            self.log_info(f"Could not find backtesting_days from {self._config_file}, falling back to environment variables.")
            days = None

        if not days:
            days = os.getenv("backtesting_days")

        if not days:
            self.log_critical("backtesting_days missing from both config and environment variables.")
            raise RuntimeError("Missing backtesting days")
    
        return int(days)
    

    def load_backtesting_variables(self):
        return self._load_days(), self._load_initial_cash(), self._load_strategy_list()


    def load_sma_windows(self):
        try:
            with open(self._config_file, "r") as file:
                conf = toml.load(file)
                ml_variables = conf.get("ml-variables", {})
                list = ml_variables.get("sma_windows")

                normalized = self._normalize_watchlist(list)
                if normalized:
                    normalized = [int(x) for x in normalized]
                    return normalized

        except Exception:
            self.log_info(
                f"Could not find sma_windows in {self._config_file}, falling back to environment variables.\n"
            )

        # Fallback to env var
        env_list = os.getenv("sma_windows")
        normalized = self._normalize_watchlist(env_list)
        if normalized:
            normalized = [int(x) for x in normalized]
            return normalized

        self.log_warning("sma_windows not found; defaulting to empty list.\n")
        return [0]
    

    def load_ml_var(self, variable_name):
        try:
            with open(self._config_file, "r") as file:
                conf = toml.load(file)
                ml_variables = conf.get("ml-variables", {})
                variable = ml_variables.get(variable_name)

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file}, falling back to environment variables.")
            variable = None

        except Exception:
            self.log_info(f"Could not find {variable_name} from {self._config_file}, falling back to environment variables.")
            variable = None

        if not variable:
            variable = os.getenv(variable_name)

        if not variable:
            self.log_critical(f"{variable_name} missing from both config and environment variables.")
            raise RuntimeError(f"Missing {variable_name}")
    
        return int(variable)
    
    def load_min_lookback(self):
        min_lookback = max(
            *self.load_sma_windows(),
            self.load_ml_var("rsi_window"),
            self._macd_stabilization,
            self.load_ml_var("zscore_window"),
        )

        return min_lookback


if __name__ == "__main__":
    conf = Config()

    info = f"""
---INFO---
keys:           {conf.load_keys()}, 
strategy:       {conf.strategy_name}, 
watchlist:      {conf.watchlist}
---BACKTESTING---
days:           {conf._load_days(), type(conf._load_days())}, 
cash:           {conf._load_initial_cash(), type(conf._load_initial_cash)}, 
strategy list:  {conf._load_strategy_list(), type(conf._load_strategy_list())}
---INFO---
min_lookback:   {conf.load_min_lookback()}
----------


"""
    
    print(info)