"use client"

import { useEffect, useState } from "react";
import styles from "./EvaluationPanel.module.css";

export default function EvaluationPanel() {
    const [results, setResults] = useState<any[]>([]);
    const [files, setFiles] = useState<string[]>([]);
    const [message, setMessage] = useState("");
    const [reports, setReports] = useState<Record<string, string>>({});
    
    const columnNames: Record<string, string> = {
        total_return_pct: "Return %",
        final_value: "Final Value",
        sharpe_ratio: "Sharpe Ratio",
        max_drawdown_pct: "Max Drawdown %",
        num_trades: "Trades",
        win_rate_pct: "Win Rate %",
        strategy: "Strategy",
        symbol: "Symbol",
    };

    const columns =
    results.length > 0
        ? Object.keys(results[0] as Record<string, string>)
        : [];

    const models = files.reduce((acc: Record<string, string[]>, file) => {
      const model = file.split("/")[0];

      if (!acc[model]) {
          acc[model] = [];
      }

      acc[model].push(file);

      return acc;
    }, {});

    console.log("FILES:", files);
    console.log("MODELS:", models);

    
    useEffect(() => {
			async function loadData() {
					const backtestResponse = await fetch(
            "http://localhost:8000/backtest_results"
          );
					const backtestData = await backtestResponse.json();
          console.log("BACKTEST:", backtestData);

					const evalResponse = await fetch(
            "http://localhost:8000/evaluation"
          );
					const evalData = await evalResponse.json();
          console.log("EVALUATION:", evalData);

					setResults(backtestData.data || []);
					setFiles(evalData.files || []);
          setResults(backtestData.data || []);
          setFiles(evalData.files || []);

          const reportFiles = (evalData.files || []).filter(
              (file: string) => file.endsWith(".txt")
          );

          const loadedReports: Record<string, string> = {};

          for (const file of reportFiles) {
              const response = await fetch(
                  `http://localhost:8000/evaluation/report?path=${encodeURIComponent(file)}`
              );

              const reportData = await response.json();

              loadedReports[file] = reportData.content;
          }

          setReports(loadedReports);

					if (
							(!backtestData.data || backtestData.data.length === 0) &&
							(!evalData.files || evalData.files.length === 0)
					) {
							setMessage(
									"No backtesting or machine learning evaluation results are available yet."
							);
					}
			}

			loadData();
		}, []);
    if (message) {
      return (
        <main className={styles.emptyState}>
          <h2>No Results Available</h2>
          <p>{message}</p>
        </main>
      );
		}
		return (
      <main>
          <h2 className={styles.sectionTitle}>Backtesting Results</h2>

          {results.length > 0 ? (
              <table className={styles.resultsTable}>
                  <thead>
                      <tr>
                          {columns.map((column) => (
                              <th key={column}>
                                  {columnNames[column] || column}
                              </th>
                          ))}
                      </tr>
                  </thead>

                  <tbody>
                      {results.map((row: any, index) => (
                          <tr key={index}>
                              {columns.map((column) => (
                                  <td key={column}>
                                      {row[column]}
                                  </td>
                              ))}
                          </tr>
                      ))}
                  </tbody>
              </table>
          ) : (
              <p>No backtesting results available.</p>
          )}

          <h2 className={styles.sectionTitle}>ML Evaluations</h2>

          {Object.entries(models).map(([model, modelFiles]) => (
              <details key={model} className={styles.modelCard}>
                  <summary className={styles.modelSummary}>{model}</summary>

                  <div>
                      {Array.from(
                          new Set(
                              modelFiles.map((file) => {
                                  const parts = file.split("_");
                                  return parts[parts.length - 1]
                                      .replace(".png", "")
                                      .replace(".txt", "");
                              })
                          )
                      ).map((symbol) => (
                          <details key={symbol}>
                              <summary className={styles.symbolSummary}>
                                  {symbol}
                              </summary>

                              <div className={styles.symbolContent}>
                                  <div className={styles.evaluationContent}>
                                      <div>
                                          {modelFiles
                                              .filter(
                                                  (file) =>
                                                      file.includes(symbol) &&
                                                      file.endsWith(".png")
                                              )
                                              .map((file) => (
                                                  <img
                                                      key={file}
                                                      src={`http://localhost:8000/evaluation/image?path=${encodeURIComponent(file)}`}
                                                      alt={symbol}
                                                      className={styles.confusionMatrix}
                                                  />
                                              ))}
                                      </div>

                                      <div>
                                          {modelFiles
                                              .filter(
                                                  (file) =>
                                                      file.includes(symbol) &&
                                                      file.endsWith(".txt")
                                              )
                                              .map((file) => (
                                                  <pre
                                                      key={file}
                                                      className={styles.report}
                                                  >
                                                      {reports[file]}
                                                  </pre>
                                              ))}
                                      </div>
                                  </div>
                              </div>
                          </details>
                      ))}
                  </div>
              </details>
          ))}
      </main>
    );
}