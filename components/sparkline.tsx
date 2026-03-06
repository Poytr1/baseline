"use client";

import { LineChart, Line } from "recharts";
import type { SparklinePoint } from "@/lib/queries/rankings";

interface SparklineProps {
  data: SparklinePoint[];
}

export function Sparkline({ data }: SparklineProps) {
  if (!data || data.length < 2) {
    return <div className="h-6 w-20" />;
  }

  const first = data[0].elo;
  const last = data[data.length - 1].elo;
  const trendingUp = last >= first;
  const color = trendingUp ? "#16a34a" : "#dc2626";

  return (
    <LineChart width={80} height={24} data={data}>
      <Line
        type="monotone"
        dataKey="elo"
        stroke={color}
        strokeWidth={1.5}
        dot={false}
        isAnimationActive={false}
      />
    </LineChart>
  );
}
