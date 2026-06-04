from live_trader import AlpacaTrader, Config, STRATEGIES
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, List

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

class ConfigUpdate(BaseModel):
    section: str
    key: str
    value: Any

class ConfigUpdateRequest(BaseModel):
    strategy: str
    watchlist: List[str]

    alpaca_key: str
    alpaca_secret: str

    initial_cash: int
    days: int
    strategy_list: List[str]

    sma1: int
    sma2: int
    sma3: int

    rsi: int
    zscore: int

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




@app.get("/config")
def get_config():
    return config.to_dict()

@app.put("/config")
def update_config(data: ConfigUpdate):
    try:
        config.update_variable(
            section=data.section,
            key=data.key,
            value=data.value
        )

        return {
            "success": True,
            "message": f"Updated [{data.section}] {data.key}",
            "value": data.value
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.put("/config/all")
def update_all_settings(
    request: ConfigUpdateRequest,
):
    config.update_variable(
        "live",
        "strategy",
        request.strategy,
    )

    config.update_variable(
        "live",
        "watchlist",
        ", ".join(request.watchlist),
    )

    config.update_variable(
        "backtesting",
        "initial_cash",
        request.initial_cash,
    )

    config.update_variable(
        "backtesting",
        "backtesting_days",
        request.days,
    )

    config.update_variable(
        "backtesting",
        "strategy_list",
        ", ".join(request.strategy_list),
    )

    config.update_variable(
        "ml-variables",
        "sma_window1",
        request.sma1,
    )

    config.update_variable(
        "ml-variables",
        "sma_window2",
        request.sma2,
    )

    config.update_variable(
        "ml-variables",
        "sma_window3",
        request.sma3,
    )

    config.update_variable(
        "ml-variables",
        "rsi_window",
        request.rsi,
    )

    config.update_variable(
        "ml-variables",
        "zscore_window",
        request.zscore,
    )

    config.update_variable(
        "keys",
        "alpaca_key",
        request.alpaca_key,
    )

    config.update_variable(
        "keys",
        "alpaca_secret_key",
        request.alpaca_secret,
    )

    return {
        "success": True
    }

@app.get("/strategies")
def get_strategies():
    return [
        {
            "id": key,
            "name": key.replace("_", " ").title()
        }
        for key in list(STRATEGIES.keys())
    ]



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