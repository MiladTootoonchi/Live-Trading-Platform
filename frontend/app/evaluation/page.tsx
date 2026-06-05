"use client";

import { useEffect, useState } from "react";

import styles from "./evaluation.module.css"
import NavBar from "../components/NavBar/NavBar"
import EvaluationPanel from "../components/EvaluationPanel/EvaluationPanel"
import IsMarketOpen from "../components/IsMarketOpen/IsMarketOpen"
import LiveButton from "../components/LiveButton/LiveButton"
import BacktestButton from "../components/BacktestButton/BacktestButton";

export default function EvaluationPage(){
    const [liveRunning, setLiveRunning] = useState(false);
    const [loadingLive, setLoadingLive] = useState(false);
    const [marketOpen, setMarketOpen] = useState(false);
    const [backtestRunning, setBacktestRunning] = useState(false);
    const [loadingBacktest, setLoadingBacktest] = useState(false);

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
      fetchLiveStatus();
      fetchBacktestStatus();

      const interval = setInterval(() => {
        fetchLiveStatus();
        fetchBacktestStatus();
      }, 5000);

      return () => clearInterval(interval);
    }, []);
    
    return(
      <main>
        <div className={styles.container}>
          <div className={styles.header}>
            <h1 className="Title">Evaluation</h1>
            <NavBar />
          </div>

          <div className={styles.content}>
            <EvaluationPanel></EvaluationPanel>
            <div className={styles.sidebar}>
              <IsMarketOpen isOpen = {marketOpen}></IsMarketOpen>
              <LiveButton liveRunning={liveRunning} loadingLive={loadingLive} toggleLive={toggleLive}></LiveButton>
              <BacktestButton backtestRunning={backtestRunning} loadingBacktest={loadingBacktest} runBacktest={runBacktest}/>
            </div>
          </div>

          <div className = "footer">
            <p>© 2024 Live Trader Too</p>
          </div>
        </div>
      </main>
    )
}