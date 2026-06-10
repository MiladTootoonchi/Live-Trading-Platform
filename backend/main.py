from live_trader import AlpacaTrader, Config, STRATEGIES
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from multiprocessing import Process
import csv
from typing import Any, List
import asyncio
import traceback


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

    ml_training_lookback: int

    macd_fast: int
    macd_slow: int
    macd_signal: int

    time_steps: int

config = None
trader = None

try:
    config = Config()
    trader = AlpacaTrader(config)
except Exception:
    traceback.print_exc()


def trader_ready():
    return trader is not None


@app.get("/orders")
async def get_orders():
    if not trader_ready():
        return {"orders": []}

    orders = await trader.get_orders()

    if not isinstance(orders, list):
        return {"orders": []}

    return {"orders": orders}

@app.get("/positions")
async def get_positions():
    if not trader:
        return {"positions": []}

    positions = await trader.get_positions()

    if not isinstance(positions, list):
        return {"positions": []}

    return {"positions": positions}

@app.get("/equity_history")
async def get_equity_history(period: str = "1D", timeframe: str = "1Min"):
    if not trader_ready():
        return {"equity_history": []}

    history = await trader.get_equity_history(
        period=period,
        timeframe=timeframe,
    )

    if not isinstance(history, list):
        return {"equity_history": []}

    return {"equity_history": history}

@app.get("/account_info")
async def get_account_info():
    if not trader_ready():
        return {"account_info": {}}

    account_info = await trader.get_account_info()

    if not isinstance(account_info, dict):
        return {"account_info": {}}

    return {"account_info": account_info}

@app.get("/account_metrics")
async def get_account_metrics():
    if not trader_ready():
        return {
            "equity": 0,
            "cash": 0,
            "unrealized_pnl": 0,
            "realized_pnl": 0,
            "buying_power": 0,
            "maintenance_margin": 0,
            "initial_margin": 0,
            "account_leverage": 0,
            "margin_cushion": 0,
        }
    account = await trader.get_account_info()

    equity = float(account.get("equity", 0))
    maintenance_margin = float(
        account.get("maintenance_margin", 0)
    )

    return {
        "equity": equity,
        "cash": float(account.get("cash", 0)),
        "unrealized_pnl": float(
            account.get("unrealized_pl", 0)
        ),
        "realized_pnl": (
            equity - float(account.get("last_equity", 0))
        ),
        "buying_power": float(
            account.get("buying_power", 0)
        ),
        "maintenance_margin": maintenance_margin,
        "initial_margin": float(
            account.get("initial_margin", 0)
        ),
        "account_leverage": float(
            account.get("multiplier", 1)
        ),
        "margin_cushion": (
            (equity - maintenance_margin) / equity * 100
            if equity > 0
            else 0
        ),
    }


@app.get("/config")
def get_config():
    if config is None:
        return {}

    return config.to_dict()

@app.put("/config")
def update_config(data: ConfigUpdate):
    global config

    if config is None:
        config = Config()
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
def update_all_settings(request: ConfigUpdateRequest):
    global config
    global trader

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

    config.update_variable(
    "ml-variables",
    "ml_training_lookback",
    request.ml_training_lookback,
    )

    config.update_variable(
        "ml-variables",
        "macd_fast",
        request.macd_fast,
    )

    config.update_variable(
        "ml-variables",
        "macd_slow",
        request.macd_slow,
    )

    config.update_variable(
        "ml-variables",
        "macd_signal",
        request.macd_signal,
    )

    config.update_variable(
        "ml-variables",
        "time_steps",
        request.time_steps,
    )

    try:
        config = Config()
        trader = AlpacaTrader(config)
    except Exception:
        traceback.print_exc()
        trader = None

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
    if not trader:
        return {
            "success": False,
            "order_id": None
        }

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

@app.delete("/positions/{symbol}")
async def close_position(symbol: str):
    if not trader:
        return {
            "success": False,
            "message": "Trader not initialized"
        }
    await trader.close_position(symbol)

    return {
        "success": True,
        "message": f"Position {symbol} closed"
    }

@app.get("/validate_order")
async def validate_order(symbol: str, qty: int, order_type: str):
    if not trader:
        return {
            "is_valid": False,
            "message": "Trader not initialized"
        }
    
    is_valid, message = await trader.validate_order(symbol, qty, order_type)
    return {"is_valid": is_valid, "message": message}

@app.post("/start")
async def start():
    if not trader:
        return {
            "status": "not_initialized",
            "live": False
        }
    await trader.start_live()

    return {
        "status": "started",
        "live": await trader.get_live_status()
    }

@app.post("/stop")
async def stop():
    if not trader:
        return {
            "status": "not_initialized",
            "live": False
        }   
    await trader.stop_live()

    return {
        "status": "stopped",
        "live": await trader.get_live_status()
    }


@app.get("/status")
async def status():
    if not trader:
        return {
            "live": False,
            "market_open": False
        }
    return {
        "live": await trader.get_live_status(),
        "market_open": await trader.is_market_open()
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

@app.delete("/logs")
def clear_logs():
    try:
        if LOG_FILE.exists():
            LOG_FILE.unlink()

        return {
            "success": True,
            "message": "Logs cleared"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

EVAL_DIR = Path("logfiles/evaluations")

@app.get("/evaluation")
async def list_evaluations():
    if not EVAL_DIR.exists():
        return {
            "success": False,
            "message": "No evaluations found.",
            "files": []
        }

    files = []

    for file in EVAL_DIR.rglob("*"):
        if file.is_file():
            files.append(str(file.relative_to(EVAL_DIR)))

    if not files:
        return {
            "success": False,
            "message": "No evaluations found.",
            "files": []
        }

    return {
        "success": True,
        "message": "",
        "files": files
    }

@app.get("/evaluation/image")
def get_evaluation_image(path: str):
    file_path = EVAL_DIR / path

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    return FileResponse(file_path)

@app.get("/evaluation/report")
def get_evaluation_report(path: str):
    file_path = EVAL_DIR / path

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    return {
        "success": True,
        "content": file_path.read_text(encoding="utf-8")
    }


BACKTEST_RESULTS = Path("backtest_results.csv")

@app.get("/backtest_results")
def get_backtest_results():
    if not BACKTEST_RESULTS.exists():
        return {
            "success": False,
            "message": "No backtest results found. Run a backtest first.",
            "data": []
        }

    with open(BACKTEST_RESULTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {
            "success": False,
            "message": "The backtest results file is empty.",
            "data": []
        }

    return {
        "success": True,
        "message": "",
        "data": rows
    }


backtest_process = None
def run_backtest_process():
    if not trader:
        return
    
    asyncio.run(trader.run_backtest())

@app.post("/backtest/start")
async def start_backtest():
    if not trader:
        return {
            "success": False,
            "message": "Trader not initialized"
        }
    
    global backtest_process

    if (
        backtest_process is not None
        and backtest_process.is_alive()
    ):
        return {
            "success": False,
            "message": "Backtest already running"
        }

    backtest_process = Process(
        target=run_backtest_process
    )

    backtest_process.start()

    return {
        "success": True,
        "status": "started"
    }

@app.get("/backtest/status")
async def backtest_status():
    global backtest_process

    return {
        "running": (
            backtest_process is not None
            and backtest_process.is_alive()
        )
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )