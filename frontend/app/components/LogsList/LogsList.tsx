"use client";

import { useEffect, useState, useRef } from "react";
import styles from "./LogsList.module.css";

export default function LogsList() {
    const [logs, setLogs] = useState<string[]>([]);
    const offsetRef = useRef(0);

    const clearLogs = async () => {
      try {
        const res = await fetch(
          "http://localhost:8000/logs",
          {
            method: "DELETE",
          }
        );

        if (!res.ok) {
          throw new Error("Failed to clear logs");
        }

        setLogs([]);
        offsetRef.current = 0;
      } catch (err) {
        console.error(err);
      }
    };

    useEffect(() => {
      const fetchLogs = async () => {
        try {
          const res = await fetch(
            `http://localhost:8000/logs?offset=${offsetRef.current}`
          );

          const data = await res.json();

          if (data.logs.length > 0) {
            setLogs(prev => [...prev, ...data.logs]);
          }

          offsetRef.current = data.offset;
        } catch (err) {
          console.error(err);
        }
      };

      fetchLogs();

      const interval = setInterval(fetchLogs, 2000);

      return () => clearInterval(interval);
    }, []);

  return (
    <div className={styles.logsPanel}>
      <div className={styles.header}>
        <h3>Logs</h3>

        <button
          onClick={clearLogs}
          className={styles.clearButton}
        >
          Clear Logs
        </button>
      </div>

      <div className={styles.logsList}>
        {[...logs].reverse().map((log, index) => (
          <div key={index} className={styles.logRow}>
              {log}
          </div>
        ))}
      </div>
    </div>
  );
}