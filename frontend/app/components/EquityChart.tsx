"use client";

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
  date: string;
  equity: number;
};

export default function EquityChart({data,}: {
  data: EquityPoint[];
}) {
  return (
    <ResponsiveContainer width="100%" height={500}>
      <LineChart 
        data={data}
        style={{
        backgroundColor: "rgba(255, 255, 255, 0.08)",
        borderRadius: "8px",
        }}
        margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 20,
        }
        }>
        <CartesianGrid strokeDasharray="3 3" />

        <XAxis
            dataKey="date"
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
            formatter={(value: number) => [
                `$${value.toLocaleString("en-US", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
                })}`,
                "Equity",
            ]}
            labelFormatter={(label) => {
                const d = new Date(label);

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
  );
}