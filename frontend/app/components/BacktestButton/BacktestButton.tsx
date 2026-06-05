"use client";

import styles from "./BacktestButton.module.css";

interface BacktestButtonProps {
  backtestRunning: boolean;
  loadingBacktest: boolean;
  runBacktest: () => void;
}

export default function BacktestButton({
  backtestRunning,
  loadingBacktest,
  runBacktest,
}: BacktestButtonProps) {
  return (
    <button
      className={`${styles.button} ${
        backtestRunning ? styles.running : ""
      }`}
      onClick={runBacktest}
      disabled={backtestRunning || loadingBacktest}
    >
      {loadingBacktest
        ? "Starting..."
        : backtestRunning
        ? "Running Backtest..."
        : "Run Backtest"}
    </button>
  );
}