"use client";

import { useEffect, useState } from "react";

import styles from "./settings.module.css";

import NavBar from "../components/NavBar/NavBar";
import LiveButton from "../components/LiveButton/LiveButton";
import IsMarketOpen from "../components/IsMarketOpen/IsMarketOpen";

import SettingsCard from "../components/SettingsCard/SettingsCard";
import SaveSettingsButton from "../components/SaveSettingsButton/SaveSettingsButton";
import BacktestButton from "../components/BacktestButton/BacktestButton";
import Footer from "../components/Footer/Footer";

export default function SettingsPage() {
  const [liveRunning, setLiveRunning] = useState(false);
  const [marketOpen, setMarketOpen] = useState(false);

  const [saving, setSaving] = useState(false);

  const [backtestRunning, setBacktestRunning] = useState(false);
  const [loadingBacktest, setLoadingBacktest] = useState(false);

  const [strategy, setStrategy] = useState("");
  const [strategies, setStrategies] = useState<
    { id: string; name: string }[]
  >([]);
  const [strategyList, setStrategyList] = useState<string[]>([]);

  const [alpacaKey, setAlpacaKey] = useState("");
  const [alpacaSecret, setAlpacaSecret] = useState("");

  const [watchlist, setWatchlist] = useState("");

  const [initialCash, setInitialCash] = useState(0);
  const [days, setDays] = useState(0);

  const [sma1, setSma1] = useState(0);
  const [sma2, setSma2] = useState(0);
  const [sma3, setSma3] = useState(0);

  const [rsi, setRsi] = useState(0);
  const [zscore, setZscore] = useState(0);

  const [mlTrainingLookback, setMlTrainingLookback] =
    useState(0);

  const [macdFast, setMacdFast] = useState(0);
  const [macdSlow, setMacdSlow] = useState(0);
  const [macdSignal, setMacdSignal] = useState(0);

  const [timeSteps, setTimeSteps] = useState(0);

  const fetchBacktestStatus = async () => {
    try {
      const res = await fetch(
        "http://localhost:8000/backtest/status"
      );

      const data = await res.json();

      setBacktestRunning(data.running);
    } catch (err) {
      console.error(err);
    }
  };

  const runBacktest = async () => {
    try {
      setLoadingBacktest(true);

      await fetch(
        "http://localhost:8000/backtest/start",
        {
          method: "POST",
        }
      );

      setBacktestRunning(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingBacktest(false);
    }
  };

  const fetchStrategies = async () => {
    try {
      const res = await fetch(
        "http://localhost:8000/strategies"
      );

      const data = await res.json();

      setStrategies(data);

    } catch (err) {
      console.error(err);
    }
  };

  const fetchSettings = async () => {
    try {
      const res = await fetch(
        "http://localhost:8000/config"
      );

      const data = await res.json();

      setStrategy(data.strategy_name);

      setStrategyList(
        data.backtesting.strategy_list
      );

      setWatchlist(
        data.watchlist.join(", ")
      );

      setAlpacaKey(data.alpaca_key);
      setAlpacaSecret(data.alpaca_secret);

      setInitialCash(
        data.backtesting.initial_cash
      );

      setDays(
        data.backtesting.days
      );

      setMlTrainingLookback(
        data.ml.ml_training_lookback
      );

      setSma1(data.ml.sma_window1);
      setSma2(data.ml.sma_window2);
      setSma3(data.ml.sma_window3);

      setMacdFast(data.ml.macd_fast);
      setMacdSlow(data.ml.macd_slow);
      setMacdSignal(data.ml.macd_signal);

      setTimeSteps(data.ml.time_steps);

      setRsi(data.ml.rsi_window);
      setZscore(data.ml.zscore_window);

    } catch (err) {
      console.error(err);
    }
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(
        "http://localhost:8000/status"
      );

      const data = await res.json();

      setLiveRunning(data.live);
      setMarketOpen(data.market_open);

    } catch (err) {
      console.error(err);
    }
  };

  const saveSettings = async () => {
    try {
      setSaving(true);

      await fetch(
        "http://localhost:8000/config/all",
        {
          method: "PUT",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            strategy,
            strategy_list: strategyList,

            watchlist: watchlist
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean),

            alpaca_key: alpacaKey,
            alpaca_secret: alpacaSecret,

            initial_cash: initialCash,
            days,

            sma1,
            sma2,
            sma3,

            rsi,
            zscore,

            ml_training_lookback:
              mlTrainingLookback,

            macd_fast: macdFast,
            macd_slow: macdSlow,
            macd_signal: macdSignal,

            time_steps: timeSteps,
          }),
        }
      );

      alert("Settings saved");

    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const addStrategy = (strategyId: string) => {
    if (
      strategyList.includes(strategyId)
    )
      return;

    setStrategyList([
      ...strategyList,
      strategyId,
    ]);
  };

  const removeStrategy = (strategyId: string) => {
    setStrategyList(
      strategyList.filter(
        (s) => s !== strategyId
      )
    );
  };

  useEffect(() => {
    const loadPage = async () => {
      await Promise.all([
        fetchSettings(),
        fetchStatus(),
        fetchStrategies(),
        fetchBacktestStatus(),
      ]);
    };

    loadPage();
  }, []);

  useEffect(() => {
    if (!backtestRunning) return;

    const interval = setInterval(
      fetchBacktestStatus,
      5000
    );

    return () =>
      clearInterval(interval);
  }, [backtestRunning]);


  return (
    <main>
      <div className={styles.container}>

        <div className={styles.header}>
          <h1 className="Title">
            Settings
          </h1>

          <NavBar />
        </div>

        <div className={styles.content}>

          <div className={styles.settingsArea}>

            <SettingsCard title="API Keys">

              <label>Alpaca Key</label>
              <input
                value={alpacaKey}
                onChange={(e) =>
                  setAlpacaKey(e.target.value)
                }
              />

              <label>Alpaca Secret</label>
              <input
                type="password"
                value={alpacaSecret}
                onChange={(e) =>
                  setAlpacaSecret(e.target.value)
                }
              />

            </SettingsCard>

            <SettingsCard title="Trading">

              <label>
                Strategy
              </label>

              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
              >
                {strategies.map((s) => (
                  <option
                    key={s.id}
                    value={s.id}
                  >
                    {s.name.toUpperCase()}
                  </option>
                ))}
              </select>

              <label>
                Watchlist
              </label>

              <input
                value={watchlist}
                onChange={(e) =>
                  setWatchlist(
                    e.target.value
                  )
                }
              />

            </SettingsCard>

            <SettingsCard title="Backtesting">

              <label>
                Initial Cash
              </label>

              <input
                type="number"
                value={initialCash}
                onChange={(e) =>
                  setInitialCash(
                    Number(
                      e.target.value
                    )
                  )
                }
              />

              <label>
                Days
              </label>

              <input
                type="number"
                value={days}
                onChange={(e) =>
                  setDays(
                    Number(
                      e.target.value
                    )
                  )
                }
              />

              <label>Available Strategies</label>

              <div className={styles.strategyPicker}>
                {strategies
                  .filter(
                    (s) =>
                      !strategyList.includes(s.id)
                  )
                  .map((s) => (
                    <div
                      key={s.id}
                      className={styles.strategyItem}
                      onClick={() =>
                        addStrategy(s.id)
                      }
                    >
                      {s.name.toUpperCase()}
                    </div>
                  ))}
              </div>

              <label>Strategy List</label>

              <div className={styles.selectedStrategies}>
                {strategyList.map((strategyId) => {
                  const strategy =
                    strategies.find(
                      (s) => s.id === strategyId
                    );

                  return (
                    <div
                      key={strategyId}
                      className={styles.selectedStrategy}
                      onClick={() =>
                        removeStrategy(strategyId)
                      }
                    >
                      {strategy?.name.toUpperCase()}
                    </div>
                  );
                })}
              </div>

            </SettingsCard>

            <SettingsCard title="ML Settings">

              <label>ML Training Lookback</label>
              <input
                type="number"
                value={mlTrainingLookback}
                onChange={(e) =>
                  setMlTrainingLookback(Number(e.target.value))
                }
              />

              <label>SMA 1</label>
              <input
                type="number"
                value={sma1}
                onChange={(e) =>
                  setSma1(Number(e.target.value))
                }
              />

              <label>SMA 2</label>
              <input
                type="number"
                value={sma2}
                onChange={(e) =>
                  setSma2(Number(e.target.value))
                }
              />

              <label>SMA 3</label>
              <input
                type="number"
                value={sma3}
                onChange={(e) =>
                  setSma3(Number(e.target.value))
                }
              />

              <label>RSI Window</label>
              <input
                type="number"
                value={rsi}
                onChange={(e) =>
                  setRsi(Number(e.target.value))
                }
              />

              <label>MACD Fast</label>
              <input
                type="number"
                value={macdFast}
                onChange={(e) =>
                  setMacdFast(Number(e.target.value))
                }
              />

              <label>MACD Slow</label>
              <input
                type="number"
                value={macdSlow}
                onChange={(e) =>
                  setMacdSlow(Number(e.target.value))
                }
              />

              <label>MACD Signal</label>
              <input
                type="number"
                value={macdSignal}
                onChange={(e) =>
                  setMacdSignal(Number(e.target.value))
                }
              />

              <label>Time Steps</label>
              <input
                type="number"
                value={timeSteps}
                onChange={(e) =>
                  setTimeSteps(Number(e.target.value))
                }
              />

              <label>ZScore Window</label>
              <input
                type="number"
                value={zscore}
                onChange={(e) =>
                  setZscore(Number(e.target.value))
                }
              />

            </SettingsCard>

          </div>

          <div className={styles.sidebar}>
            <IsMarketOpen
              isOpen={marketOpen}
            />

            <LiveButton
              liveRunning={liveRunning}
              loadingLive={false}
              toggleLive={() => {}}
            />

           <BacktestButton backtestRunning={backtestRunning} loadingBacktest={loadingBacktest} runBacktest={runBacktest}/>
            
            <SaveSettingsButton
              onClick={saveSettings}
              loading={saving}
            />
          </div>

        </div>

        <Footer />

      </div>
    </main>
  );
}