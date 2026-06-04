from live_trader import AlpacaTrader, SideSignal, Config
from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrderRequest(BaseModel):
    symbol: str
    qty: int
    side: str
    order_type: str

config = Config()

trader = AlpacaTrader(config)



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



@app.post("/place_order")
async def place_order(order: OrderRequest):

    order_id = await trader.place_order(
        symbol=order.symbol,
        qty=order.qty,
        side=order.side,
        order_type=order.order_type,
    )

    return {
        "success": True,
        "order_id": order_id,
    }

@app.get("/validate_order")
async def validate_order(symbol: str, qty: int, order_type: str):
    is_valid, message = await trader.validate_order(symbol, qty, order_type)
    return {"is_valid": is_valid, "message": message}

@app.post("/start")
async def start():
    await trader.start_live()

    return {
        "status": "started",
        "live": trader.is_live_running
    }

@app.post("/stop")
async def stop():
    await trader.stop_live()

    return {
        "status": "stopped",
        "live": trader.is_live_running
    }

@app.get("/status")
async def status():
    return {
        "live": await trader.get_live_status()
    }


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