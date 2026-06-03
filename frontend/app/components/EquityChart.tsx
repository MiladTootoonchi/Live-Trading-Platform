"use client";

import styles from "./EquityChart.module.css";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

type EquityPoint = {
  timestamp: number;
  equity: number;
};

type Props = {
  data: EquityPoint[];
  currentEquity: number;
  pnl: number;
  pnlPct: number;
};

export default function EquityChart({data, currentEquity, pnl, pnlPct,}: Props) {
  return (
    <div className = {styles.equitySection}>
        <div className={styles.equitySummary}>
            <h2 className = {styles.equityValue}>
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
        <div className={styles.equityChart}>
            <ResponsiveContainer width="100%" height="100%">
            <LineChart 
                data={data}
                margin={{
                    top: 20,
                    right: 30,
                    left: 20,
                    bottom: 20,
                }
                }>
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis
                    dataKey="timestamp"
                    tick={({ x, y, payload }) => {
                        const date = new Date(payload.value);

                        const day = date.toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        });

                        const time = date.toLocaleTimeString("en-US", {
                        hour: "2-digit",
                        minute: "2-digit",
                        });

                        return (
                        <g transform={`translate(${x},${y})`}>
                            <text
                            x={0}
                            y={0}
                            dy={16}
                            textAnchor="middle"
                            fill="#999"
                            fontSize={12}
                            >
                            <tspan x="0" dy="0">
                                {day}
                            </tspan>

                            <tspan x="0" dy="14">
                                {time}
                            </tspan>
                            </text>
                        </g>
                        );
                    }}
                />

                <YAxis
                    domain={[
                        (min: number) => min * 0.995,
                        (max: number) => max * 1.005,
                    ]}
                    tickFormatter={(value) => {
                        if (value >= 1_000_000)
                        return `$${(value / 1_000_000).toFixed(3)}M`;

                        if (value >= 1_000)
                        return `$${(value / 1_000).toFixed(0)}K`;

                        return `$${value.toFixed(0)}`;
                    }}
                />

                <Tooltip
                    contentStyle={{
                        backgroundColor: "#1a1a1a",
                        border: "1px solid #333",
                        borderRadius: "8px",
                        color: "#fff",
                    }}
                    formatter={(value) => [
                    `$${Number(value).toLocaleString("en-US", {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                    })}`,
                    "Equity",
                    ]}
                    labelFormatter={(label) => {
                        const d = new Date(Number(label));

                        return d.toLocaleString("en-US", {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                        });
                    }}
                />

                <Line
                    type="monotone"
                    dataKey="equity"
                    stroke="#00ff00d0"
                    dot={false}
                    strokeWidth={3}
                />
            </LineChart>
            </ResponsiveContainer>
        </div>
    </div>
  );
}