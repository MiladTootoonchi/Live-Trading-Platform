from dotenv import load_dotenv
import os
import toml
import logging
from typing import Tuple, List

load_dotenv()


class Config:
    """
    Central configuration manager for live trading and backtesting.

    Loads settings from a TOML file with fallback to environment variables.
    Provides access to API credentials, strategy configuration, and
    ML-related parameters used across the system.
    """

    def __init__(self, config_file_path: str = "settings.toml"):
        """
        Initialize configuration and compute derived values.

        Loads credentials, strategy settings, watchlists, and ML variables.
        Also calculates lookback requirements used in model training.

        Args:
            config_file_path: Path to the TOML configuration file.
        """
        self._config_file_path = config_file_path
        self._logger = self._make_logger()
        self._alpaca_key, self._alpaca_secret = self._load_api_keys()

        self.watchlist = self._load_watchlist()
        self.strategy_name = self._load_strategy_name()
        self.apca_url = "https://paper-api.alpaca.markets"

        # calculated ml-variables
        self.macd_stabilization = self.load_ml_variable("macd_slow") * 3
        self.number_of_sma_windows = 3
        self.sma_windows = [self.load_ml_variable(f"sma_window{i}") for i in range(1, self.number_of_sma_windows+1)]
        self.min_lookback = max(
            *self.sma_windows,
            self.load_ml_variable("rsi_window"),
            self.macd_stabilization,
            self.load_ml_variable("zscore_window"),
        )

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

    def _make_logger(self) -> logging.Logger:
        """
        Create and configure a file-based logger.

        The logger writes formatted log messages to a file inside
        the 'logfiles' directory and ensures no duplicate handlers
        are attached.

        Returns:
            logging.Logger: Configured logger instance.
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
        Load the active trading strategy name.

        Attempts to read the value from the config file and
        falls back to environment variables if necessary.

        Returns:
            str: Name of the selected strategy.

        Raises:
            RuntimeError: If no strategy is defined.
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                live = conf.get("live", {})
                strategy = live.get("strategy")

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file_path}, falling back to environment variables.")
            strategy = None

        except Exception:
            self.log_info(f"Could not find strategy from {self._config_file_path}, falling back to environment variables.")
            strategy = None

        if not strategy:
            strategy = os.getenv("strategy")

        if not strategy:
            self.log_critical("Strategy name missing from both config and environment variables.")
            raise RuntimeError("Missing strategy name")
    
        return strategy



    # ----- watchlist -----
    @staticmethod
    def _normalize_watchlist(value: object) -> List[str]:
        """
        Normalize watchlist input into a list of symbols.

        Accepts either a list or a comma-separated string and
        removes empty or whitespace-only entries.

        Args:
            value: Raw watchlist value from config or environment.

        Returns:
            List[str]: Cleaned list of ticker symbols.

        Raises:
            TypeError: If the value type is unsupported.
        """
        if not value:
            return []

        # Already a list (correct TOML)
        if isinstance(value, list):
            return [
                str(symbol).strip()
                for symbol in value
                if str(symbol).strip()
            ]

        # Env var or misconfigured TOML → comma-separated string
        if isinstance(value, str):
            return [
                symbol.strip()
                for symbol in value.split(",")
                if symbol.strip()
            ]

        raise TypeError("Watchlist must be a list or a comma-separated string")

    def _load_watchlist(self) -> List[str]:
        """
        Load the trading watchlist.

        Attempts to read symbols from the config file and
        falls back to environment variables if necessary.
        Returns an empty list if no symbols are defined.

        Returns:
            List[str]: List of ticker symbols to monitor.
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                live = conf.get("live", {})
                watchlist = live.get("watchlist")

                normalized = self._normalize_watchlist(watchlist)
                if normalized:
                    return normalized

        except Exception:
            self.log_info(
                f"Could not find watchlist in {self._config_file_path}, falling back to environment variables.\n"
            )

        # Fallback to env var
        env_watchlist = os.getenv("watchlist")
        normalized = self._normalize_watchlist(env_watchlist)
        if normalized:
            return normalized

        self.log_warning("Watchlist not found; defaulting to empty list.\n")
        return []



    # ----- settings -----

    def _load_api_keys(self) -> Tuple[str, str]:
        """
        Load Alpaca API credentials.

        Reads credentials from the configuration file with
        fallback to environment variables if missing.

        Returns:
            Tuple[str, str]: Alpaca API key and secret key.

        Raises:
            RuntimeError: If credentials are not provided.
        """

        alpaca_key = None
        alpaca_secret = None

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                keys = conf.get("keys", {})
                alpaca_key = keys.get("alpaca_key", alpaca_key)
                alpaca_secret = keys.get("alpaca_secret_key", alpaca_secret)

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file_path}, falling back to environment variables.")

        except Exception:
            self.log_info(f"Could not find Alpaca API credentials in {self._config_file_path}, falling back to environment variables.")

        if not alpaca_key:
            alpaca_key = os.getenv("alpaca_key")

        if not alpaca_secret:
            alpaca_secret = os.getenv("alpaca_secret_key")

        if not alpaca_key or not alpaca_secret:
            self.log_critical("Missing Alpaca API credentials. Provide them in the config file or as environment variables.")
            raise RuntimeError("Missing Alpaca API credentials")
            
        return alpaca_key, alpaca_secret
    
    def load_keys(self) -> Tuple[str, str]:
        """
        Return loaded Alpaca API credentials.

        Provides access to the API key and secret
        that were validated during initialization.

        Returns:
            Tuple[str, str]: Alpaca API key and secret key.
        """
        return (self._alpaca_key, self._alpaca_secret)
    

    def _load_strategy_list(self) -> List[str]:
        """
        Load the list of strategies for backtesting.

        Reads strategy names from the configuration file
        with fallback to environment variables if missing.

        Returns:
            List[str]: List of strategy names.
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                backtesting = conf.get("backtesting", {})
                list = backtesting.get("strategy_list")

                normalized = self._normalize_watchlist(list)
                if normalized:
                    return normalized

        except Exception:
            self.log_info(
                f"Could not find strategy_list in {self._config_file_path}, falling back to environment variables.\n"
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
        Load the initial capital for backtesting.

        Attempts to read the value from the config file
        and falls back to environment variables if needed.

        Returns:
            int: Initial cash amount.

        Raises:
            RuntimeError: If the value is not defined.
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                backtesting = conf.get("backtesting", {})
                cash = backtesting.get("initial_cash")

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file_path}, falling back to environment variables.")
            cash = None

        except Exception:
            self.log_info(f"Could not find initial_cash from {self._config_file_path}, falling back to environment variables.")
            cash = None

        if not cash:
            cash = os.getenv("initial_cash")

        if not cash:
            self.log_critical("initial_cash missing from both config and environment variables.")
            raise RuntimeError("Missing initial cash")
    
        return int(cash)

    def _load_days(self) -> int:
        """
        Load the number of backtesting days.

        Reads the configured time horizon from file
        or environment variables if necessary.

        Returns:
            int: Number of days for backtesting.

        Raises:
            RuntimeError: If the value is not defined.
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                backtesting = conf.get("backtesting", {})
                days = backtesting.get("backtesting_days")

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file_path}, falling back to environment variables.")
            days = None

        except Exception:
            self.log_info(f"Could not find backtesting_days from {self._config_file_path}, falling back to environment variables.")
            days = None

        if not days:
            days = os.getenv("backtesting_days")

        if not days:
            self.log_critical("backtesting_days missing from both config and environment variables.")
            raise RuntimeError("Missing backtesting days")
    
        return int(days)
    
    def load_backtesting_variables(self) -> Tuple[int, int, List[str]]:
        """
        Load core backtesting parameters.

        Combines days, initial capital, and strategy list
        into a single return for convenience.

        Returns:
            Tuple[int, int, List[str]]: Days, initial cash, and strategy list.
        """
        return self._load_days(), self._load_initial_cash(), self._load_strategy_list()


    def load_ml_variable(self, variable_name) -> int:
        """
        Load a machine learning configuration variable.

        Retrieves the value from the config file or
        environment variables and converts it to int.

        Args:
            variable_name: Name of the ML variable to load.

        Returns:
            int: Parsed ML variable value.

        Raises:
            RuntimeError: If the variable is not defined.
        """
        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                ml_variables = conf.get("ml-variables", {})
                variable = ml_variables.get(variable_name)

        except FileNotFoundError:
            self.log_info(f"Config file not found: {self._config_file_path}, falling back to environment variables.")
            variable = None

        except Exception:
            self.log_info(f"Could not find {variable_name} from {self._config_file_path}, falling back to environment variables.")
            variable = None

        if not variable:
            variable = os.getenv(variable_name)

        if not variable:
            self.log_critical(f"{variable_name} missing from both config and environment variables.")
            raise RuntimeError(f"Missing {variable_name}")
    
        return int(variable)


if __name__ == "__main__":
    conf = Config()

    info = f"""
---INFO---
keys:               {conf.load_keys()}, 
strategy:           {conf.strategy_name}, 
watchlist:          {conf.watchlist}

---BACKTESTING---
days:               {conf._load_days()}, 
cash:               {conf._load_initial_cash()}, 
strategy list:      {conf._load_strategy_list()}

---ML-TRAINING---
min_lookback:       {conf.min_lookback},
sma_windows:        {conf.sma_windows}
macd_stabilization: {conf.macd_stabilization}

-----------------

"""
    
    print(info)