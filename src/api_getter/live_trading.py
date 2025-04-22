import requests

# Config
ALPACA_SECRET_KEY = "jLwj0J33MMhcGVP9oFTZngMTXDbowMpLj2VN8dmL"
ALPACA_KEY = "PKIAS4EHSEJ45519IHHC"

APCA_API_BASE_URL =  "https://paper-api.alpaca.markets"

# Package

ORDERS_URL = "{}/v2/orders".format(APCA_API_BASE_URL)

HEADERS = {
    "APCA-API-KEY-ID" : ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

def createMarketOrder():
    ticker = "AAPL"
    qty = "2"
    side = "buy"
    ordertype = "market"

    data = {
        "symbol" : ticker,
        "qty" : qty,
        "side" : side,
        "type" : ordertype,
        "time_in_force" : "day"
    }

    response = requests.post(ORDERS_URL, json = data, headers = HEADERS)
    return response.content

def getOrders():
    url = f"{APCA_API_BASE_URL}/v2/orders"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def getPositions():
    url = f"{APCA_API_BASE_URL}/v2/positions"
    response = requests.get(url, headers=HEADERS)
    return response.json()