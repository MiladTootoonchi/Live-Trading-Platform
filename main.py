from live_trader import AlpacaTrader, SideSignal, Config
from fastapi import FastAPI

app = FastAPI()

config = Config()

trader = AlpacaTrader(config)

@app.get("/live")
async def live():
    await trader.live_loop()
    return {"message": "Live trading-bot started."}