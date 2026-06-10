from dotenv import load_dotenv
import os
import toml
import logging
from typing import Tuple, List

from pathlib import Path

config_path = Path(__file__).parent / "settings.toml"
config_path = str(config_path)

load_dotenv()


class Config:
    """
    Central configuration manager for live trading and backtesting.

    Loads settings from a TOML file with fallback to environment variables.
    Provides access to API credentials, strategy configuration, and
    ML-related parameters used across the system.
    """

    def __init__(self, config_file_path: str = config_path):
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
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                live = conf.get("live", {})
                strategy = live.get("strategy")

        except Exception:
            strategy = None

        if not strategy:
            strategy = os.getenv("strategy")

        if not strategy:
            self.log_critical("Strategy name missing from both config and environment variables.")
            strategy = ""
    
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
        """

        alpaca_key = None
        alpaca_secret = None

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                keys = conf.get("keys", {})
                alpaca_key = keys.get("alpaca_key", alpaca_key)
                alpaca_secret = keys.get("alpaca_secret_key", alpaca_secret)

        except:
            pass
        
        if not alpaca_key:
            alpaca_key = os.getenv("alpaca_key")

        if not alpaca_secret:
            alpaca_secret = os.getenv("alpaca_secret_key")

        if not alpaca_key:
            alpaca_key = ""

        if not alpaca_secret:
            alpaca_secret = ""
            self.log_critical("Missing Alpaca API credentials. Provide them in the config file or as environment variables.")

            
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
            pass

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
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                backtesting = conf.get("backtesting", {})
                cash = backtesting.get("initial_cash")

        except Exception:
            cash = None

        if not cash:
            cash = os.getenv("initial_cash")

        if not cash:
            self.log_critical("initial_cash missing from both config and environment variables.")
            cash = 100000
    
        return int(cash)

    def _load_days(self) -> int:
        """
        Load the number of backtesting days.

        Reads the configured time horizon from file
        or environment variables if necessary.

        Returns:
            int: Number of days for backtesting.
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                backtesting = conf.get("backtesting", {})
                days = backtesting.get("backtesting_days")

        except Exception:
            days = None

        if not days:
            days = os.getenv("backtesting_days")

        if not days:
            self.log_critical("backtesting_days missing from both config and environment variables.")
            days = 365

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
        """
        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)
                ml_variables = conf.get("ml-variables", {})
                variable = ml_variables.get(variable_name)

        except Exception:
            variable = None

        if not variable:
            variable = os.getenv(variable_name)

        if not variable:
            self.log_critical(f"{variable_name} missing from both config and environment variables.")
            variable = 0
    
        return int(variable)


    def _reload(self):
        self.watchlist = self._load_watchlist()
        self.strategy_name = self._load_strategy_name()

        self.macd_stabilization = self.load_ml_variable("macd_slow") * 3

        self.sma_windows = [
            self.load_ml_variable(f"sma_window{i}")
            for i in range(1, self.number_of_sma_windows + 1)
        ]

        self.min_lookback = max(
            *self.sma_windows,
            self.load_ml_variable("rsi_window"),
            self.macd_stabilization,
            self.load_ml_variable("zscore_window"),
        )

    def update_variable(self, section: str, key: str, value) -> None:
        """
        Update a value in the TOML configuration file.

        Args:
            section: TOML section name (e.g. 'live', 'backtesting')
            key: Variable name inside the section
            value: New value
        """

        try:
            with open(self._config_file_path, "r") as file:
                conf = toml.load(file)

            if section not in conf:
                conf[section] = {}

            conf[section][key] = value

            with open(self._config_file_path, "w") as file:
                toml.dump(conf, file)

            self.log_info(
                f"Updated config: [{section}] {key} = {value}"
            )

            # Refresh derived values if ML variables changed
            self._reload()

        except Exception as e:
            self.log_error(f"Failed updating config: {e}")
            raise

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "watchlist": self.watchlist,
            "apca_url": self.apca_url,

            "alpaca_key": self._alpaca_key,
            "alpaca_secret": self._alpaca_secret,

            "backtesting": {
                "days": self._load_days(),
                "initial_cash": self._load_initial_cash(),
                "strategy_list": self._load_strategy_list(),
            },

            "ml": {
                "ml_training_lookback": self.load_ml_variable("ml_training_lookback"),

                "sma_window1": self.load_ml_variable("sma_window1"),
                "sma_window2": self.load_ml_variable("sma_window2"),
                "sma_window3": self.load_ml_variable("sma_window3"),

                "rsi_window": self.load_ml_variable("rsi_window"),

                "macd_fast": self.load_ml_variable("macd_fast"),
                "macd_slow": self.load_ml_variable("macd_slow"),
                "macd_signal": self.load_ml_variable("macd_signal"),

                "time_steps": self.load_ml_variable("time_steps"),

                "zscore_window": self.load_ml_variable("zscore_window"),
            }
        }



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