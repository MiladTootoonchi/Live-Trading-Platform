from dotenv import load_dotenv
import os
import toml

load_dotenv()

ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_KEY = os.getenv("ALPACA_KEY")

class Config:
    """
    A class to handle the reading configurations 

    ...
    """

    def __init__(self, config_file: str) -> None:
        """
        Initialize configuration reader
        """
        try:
            with open(config_file, "r") as file:
                conf = toml.load(file)
        except:
            raise FileNotFoundError(f"The {config_file} toml file does not exist")

        # Makes the folder with config files name
        self._folder_name = os.path.splitext(os.path.basename(config_file))[0]
        os.makedirs(self._folder_name, exist_ok = True)