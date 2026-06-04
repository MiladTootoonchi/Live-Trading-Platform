"use client";

import { useEffect, useState } from "react";

import styles from "./history.module.css";
import NavBar from "../components/NavBar/NavBar";
import LiveButton from "../components/LiveButton/LiveButton";
import IsMarketOpen from "../components/IsMarketOpen/IsMarketOpen";
import LogsList from "../components/LogsList/LogsList";


export default function HistoryPage() {
  const [liveRunning, setLiveRunning] = useState(false);
  const [loadingLive, setLoadingLive] = useState(false);
  const [marketOpen, setMarketOpen] = useState(false);

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

    const interval = setInterval(() => {
      fetchLiveStatus();
    }, 5000);

    return () => clearInterval(interval);
  }, []);
  
  
  return (
    <main>
      <div className={styles.container}>
        <div className={styles.header}>
          <h1 className="Title">History</h1>
          <NavBar />
        </div>

        <div className={styles.content}>
          <LogsList />
          <div className={styles.sidebar}>
            <IsMarketOpen isOpen={marketOpen} />
            <LiveButton liveRunning={liveRunning} loadingLive={loadingLive} toggleLive={toggleLive} />
          </div>
        </div>

        <div className={styles.footer}>
          <p>© 2024 Live Trader Too</p>
        </div>
      </div>
    </main>
  );
}