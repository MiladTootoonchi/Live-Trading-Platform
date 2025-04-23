# Config
from dotenv import load_dotenv
import os

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
        pass
