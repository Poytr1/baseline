"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { EloHistoryEntry } from "@/lib/queries/player";

interface EloChartProps {
  history: EloHistoryEntry[];
}

function formatDate(dateStr: string): string {
  if (!dateStr || dateStr.length !== 8) return dateStr;
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  return `${y}-${m}`;
}

export function EloChart({ history }: EloChartProps) {
  if (history.length === 0) {
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Elo Rating History</h2>
        <p className="text-sm text-muted-foreground py-8 text-center">
          No Elo history available.
        </p>
      </div>
    );
  }

  const data = history.map((entry) => ({
    date: formatDate(entry.date),
    Overall: Math.round(entry.overall),
    Hard: entry.hard !== null ? Math.round(entry.hard) : undefined,
    Clay: entry.clay !== null ? Math.round(entry.clay) : undefined,
    Grass: entry.grass !== null ? Math.round(entry.grass) : undefined,
  }));

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Elo Rating History</h2>
      <div className="w-full h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
          >
            <XAxis
              dataKey="date"
              tick={{ fontSize: 12 }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 12 }}
              domain={["auto", "auto"]}
            />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="Overall"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="Hard"
              stroke="#2563eb"
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="Clay"
              stroke="#d97706"
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="Grass"
              stroke="#16a34a"
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
