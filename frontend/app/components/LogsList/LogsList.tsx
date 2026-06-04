"use client";

import { useEffect, useState } from "react";
import styles from "./LogsList.module.css";

export default function LogsList() {
    const [logs, setLogs] = useState<string[]>([]);
    const [offset, setOffset] = useState(0);

    useEffect(() => {
    const fetchLogs = async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/logs?offset=${offset}`
        );

        const data = await res.json();

        if (data.logs.length > 0) {
          setLogs((prev) => [...prev, ...data.logs]);
        }

        setOffset(data.offset);
      } catch (err) {
        console.error(err);
      }
    };

    fetchLogs();

    const interval = setInterval(fetchLogs, 2000);

    return () => clearInterval(interval);

  }, [offset]);

  return (
    <div className={styles.logsPanel}>
      <h3>Logs</h3>

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