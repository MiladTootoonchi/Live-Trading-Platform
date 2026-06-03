"use client";
import { useEffect, useState } from "react";
import EquityChart from "./components/EquityChart";
import PositionsList from "./components/PositionsList";
import styles from "./main.module.css";

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

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Equity data
        const equityRes = await fetch(
          "http://localhost:8000/equity_history"
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

      } catch (err) {
        console.error(err);
      }

      console.log("Refreshing dashboard...");
    };

    fetchData();

    const interval = setInterval(fetchData, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <main>
      <div className={styles.container}>

        <div className={styles.header}>
          <h1 className={"Title"}>Dashboard</h1>

          <div className={styles.navbar}>

          </div>
        </div>



        <div className={styles.content}>
          <EquityChart data={equityData} currentEquity={currentEquity} pnl={pnl} pnlPct={pnlPct} />

          <div className={styles.content_grid}>
            <PositionsList positions={positions} />

            <div className={styles.sidebar}>
              <h3>Order Here</h3>
            </div>

          </div>
        </div>



        <div className={styles.footer}>
        </div>
      </div>
    </main>
  );
}