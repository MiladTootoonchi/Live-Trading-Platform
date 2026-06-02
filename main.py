from live_trader import AlpacaTrader, SideSignal, Config
from fastapi import FastAPI
from pathlib import Path

app = FastAPI()

config = Config()

trader = AlpacaTrader(config)

@app.get("/live")
async def live():
    await trader.live()
    return {"message": "Live trading-bot started."}

@app.get("/orders")
async def get_orders():
    orders = await trader.get_orders()
    return {"orders": orders}

@app.get("/positions")
async def get_positions():
    positions = await trader.get_positions()
    return {"positions": positions}

@app.get("/equity_history")
async def get_equity_history(period: str = "1D", timeframe: str = "1Min"):
    equity_history = await trader.get_equity_history(period=period, timeframe=timeframe)
    return {"equity_history": equity_history}

@app.get("/account_info")
async def get_account_info():
    account_info = await trader.get_account_info()
    return {"account_info": account_info}

@app.get("/ismarketopen")
async def is_market_open():
    is_open = await trader.is_market_open()
    return {"is_open": is_open}


LOG_FILE = Path("logfiles/live_trading.log")
@app.get("/logs")
def get_logs(offset: int = 0):
    if not LOG_FILE.exists():
        return {
            "offset": 0,
            "logs": [],
            "has_log": False
        }

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        f.seek(offset)

        logs = f.readlines()
        offset = f.tell()

    return {
        "offset": offset,
        "logs": [line.rstrip() for line in logs],
        "has_log": True
    }


EVAL_DIR = Path("logfiles/evaluations")

@app.get("/evaluation")
async def list_evaluations():
    files = []

    for file in EVAL_DIR.rglob("*"):
        if file.is_file():
            files.append(str(file.relative_to(EVAL_DIR)))

    return {"files": files}