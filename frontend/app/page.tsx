"use client";
import { useEffect, useState } from "react";
import EquityChart from "./components/EquityChart";

type EquityPoint = {
  date: string;
  equity: number;
};


export default function Home() {
  const [equityData, setEquityData] = useState<EquityPoint[]>([]);

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
  fetch("http://localhost:8000/equity_history")
    .then((res) => res.json())
    .then((data) => {
      const chartData = data.equity_history.map(
        (point: { timestamp: number; equity: number }) => {
          const date = new Date(point.timestamp * 1000);

          return {
            date: `${date.toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
            })}\n${date.toLocaleTimeString("en-US", {
              hour: "2-digit",
              minute: "2-digit",
            })}`,
            equity: point.equity,
          };
        }
      );

      console.log(chartData);
      setEquityData(chartData);
    })
    .catch(console.error);
  }, []);

  return (
    <main>
      <div className="main">
        <div className="container">

          <div className="header">
            <h1 className="Title">Dashboard</h1>

            <div className="navbar">

            </div>
          </div>



          <div className="content">
            <div className="equity-summary">
              <h2>
                ${currentEquity.toLocaleString("en-US", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </h2>

              <p className={pnl >= 0 ? "profit" : "loss"}>
                {pnl >= 0 ? "+" : ""}
                ${pnl.toLocaleString("en-US", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
                {" "}
                ({pnlPct.toFixed(2)}%)
              </p>
            </div>

            <div className="equity-chart">
              <EquityChart data={equityData} />
            </div>

            <p className="description">
              Welcome to the dashboard! Here you can find an overview of your data and insights.
            </p>
          </div>


          <div className="sidebar">
          </div>



          <div className="footer">
          </div>
        </div>
      </div>
    </main>
  );
}