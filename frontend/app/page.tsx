"use client";
import { useEffect, useState } from "react";
import styles from "./home.module.css";
import EquityChart from "./components/EquityChart/EquityChart";
import PositionsList from "./components/PositionsList/PositionsList";
import LiveButton from "./components/LiveButton/LiveButton";
import IsMarketOpen from "./components/IsMarketOpen/IsMarketOpen";
import OrderingPanel from "./components/OrderingPanel/OrderingPanel";
import NavBar from "./components/NavBar/NavBar";
import Footer from "./components/Footer/Footer";
import AccountMetricsPanel from "./components/AccountMetricsPanel/AccountMetricsPanel";

type EquityPoint = {
  timestamp: number;
  equity: number;
};

type Position = {
  symbol: string;
  qty: string;
  current_price: string;
  avg_entry_price: string;
  market_value: string;
  unrealized_pl: string;
};


export default function Home() {
  const [equityData, setEquityData] = useState<EquityPoint[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [liveRunning, setLiveRunning] = useState(false);
  const [loadingLive, setLoadingLive] = useState(false);
  const [marketOpen, setMarketOpen] = useState(false);
  const [selectedPeriod, setSelectedPeriod] = useState("1M");
  const [accountMetrics, setAccountMetrics] = useState({
    equity: 0,
    cash: 0,
    unrealized_pnl: 0,
    realized_pnl: 0,
    buying_power: 0,
    maintenance_margin: 0,
    initial_margin: 0,
    account_leverage: 0,
    margin_cushion: 0,
  });

  const currentEquity =
    equityData.length > 0
      ? equityData[equityData.length - 1].equity
      : 0;

  const startEquity =
    equityData.length > 0
      ? equityData[0].equity
      : 0;

  const pnl = currentEquity - startEquity;
  const pnlPct =
    startEquity > 0
      ? (pnl / startEquity) * 100
      : 0;


  const closePosition = async (symbol: string) => {
    try {
      const res = await fetch(
        `http://localhost:8000/positions/${symbol}`,
        {
          method: "DELETE",
        }
      );

      if (!res.ok) {
        throw new Error("Failed to close position");
      }

      // Refresh positions immediately
      const positionsRes = await fetch(
        "http://localhost:8000/positions"
      );

      const positionsJson = await positionsRes.json();

      setPositions(positionsJson.positions);

    } catch (err) {
      console.error(err);
    }
  };

  const fetchLiveStatus = async () => {
    try {
      const res = await fetch("http://localhost:8000/status");
      const data = await res.json();

      setLiveRunning(data.live);
      setMarketOpen(data.market_open);
    } catch (err) {
      console.error(err);
    }
  };

  const toggleLive = async () => {
    try {
      setLoadingLive(true);

      const endpoint = liveRunning ? "stop" : "start";

      const res = await fetch(
        `http://localhost:8000/${endpoint}`,
        {
          method: "POST",
        }
      );

      const data = await res.json();

      setLiveRunning(data.live);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingLive(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Equity data
        const timeframe =
          selectedPeriod === "1D"
            ? "1Min"
            : "1D";

        const equityRes = await fetch(
          `http://localhost:8000/equity_history?period=${selectedPeriod}&timeframe=${timeframe}`
        );

        const equityJson = await equityRes.json();

        const chartData = equityJson.equity_history.map(
          (point: { timestamp: number; equity: number }) => ({
            timestamp: point.timestamp * 1000,
            equity: point.equity,
          })
        );
        
        setEquityData(chartData);

        // Positions
        const positionsRes = await fetch(
          "http://localhost:8000/positions"
        );

        const positionsJson = await positionsRes.json();

        setPositions(positionsJson.positions);

        
        // Account metrics
        const accountRes = await fetch(
          "http://localhost:8000/account_metrics"
        );

        const accountJson = await accountRes.json();

        setAccountMetrics(accountJson);

      } catch (err) {
        console.error(err);
      }

      console.log("Refreshing dashboard...");
    };

    fetchData();
    fetchLiveStatus();

    const interval = setInterval(() => {
      fetchData();
      fetchLiveStatus();
    }, 5000);

    return () => clearInterval(interval);
  }, [selectedPeriod]);

  return (
    <main>
      <div className={styles.container}>

        <div className={styles.header}>
          <h1 className={"Title"}> Portfolio Dashboard</h1>
          <NavBar />
        </div>



        <div className={styles.content}>
          <EquityChart data={equityData} 
            currentEquity={currentEquity} 
            pnl={pnl} pnlPct={pnlPct} 
            selectedPeriod={selectedPeriod} 
            onPeriodChange={setSelectedPeriod}/>

          <AccountMetricsPanel metrics={accountMetrics} />

          <div className={styles.content_grid}>
            <PositionsList positions={positions} onClosePosition={closePosition}/>

            <div className={styles.sidebar}>
              <IsMarketOpen isOpen={marketOpen} /> 
              <LiveButton liveRunning={liveRunning} loadingLive={loadingLive} toggleLive={toggleLive} />
              <OrderingPanel />
            </div>

          </div>
        </div>

        <Footer />
      </div>
    </main>
  );
}