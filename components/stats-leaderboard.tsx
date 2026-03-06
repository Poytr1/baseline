"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type {
  PlayerStatRow,
  Tour,
  StatsSurface,
  SortColumn,
} from "@/lib/queries/stats";

interface StatsLeaderboardProps {
  data: PlayerStatRow[];
  tour: Tour;
  surface: StatsSurface;
  year: string;
  limit: number;
  sortBy: SortColumn;
  order: "asc" | "desc";
}

const tours: { value: Tour; label: string }[] = [
  { value: "atp", label: "ATP" },
  { value: "wta", label: "WTA" },
];

const surfaceOptions: {
  value: StatsSurface;
  label: string;
  activeClass: string;
}[] = [
  {
    value: "all",
    label: "All",
    activeClass: "bg-primary text-primary-foreground hover:bg-primary/90",
  },
  {
    value: "hard",
    label: "Hard",
    activeClass: "bg-blue-600 text-white hover:bg-blue-700",
  },
  {
    value: "clay",
    label: "Clay",
    activeClass: "bg-amber-600 text-white hover:bg-amber-700",
  },
  {
    value: "grass",
    label: "Grass",
    activeClass: "bg-green-600 text-white hover:bg-green-700",
  },
];

const yearOptions = [
  { value: "all", label: "Career" },
  ...Array.from({ length: 10 }, (_, i) => {
    const y = String(new Date().getFullYear() - i);
    return { value: y, label: y };
  }),
];

const limitOptions = [
  { value: 50, label: "Top 50" },
  { value: 100, label: "Top 100" },
];

interface ColumnDef {
  key: SortColumn;
  label: string;
  shortLabel?: string;
}

const columns: ColumnDef[] = [
  { key: "matches", label: "Matches", shortLabel: "M" },
  { key: "aceRate", label: "Ace/Match", shortLabel: "Ace" },
  { key: "dfRate", label: "DF/Match", shortLabel: "DF" },
  { key: "firstServeInPct", label: "1st In%", shortLabel: "1st In" },
  { key: "firstServeWonPct", label: "1st Won%", shortLabel: "1st W" },
  { key: "secondServeWonPct", label: "2nd Won%", shortLabel: "2nd W" },
  { key: "bpSavedPct", label: "BP Saved%", shortLabel: "BPS" },
];

function formatStat(key: SortColumn, value: number): string {
  if (key === "matches") return String(value);
  if (key === "aceRate" || key === "dfRate") return value.toFixed(1);
  return `${value.toFixed(1)}%`;
}

export function StatsLeaderboard({
  data,
  tour,
  surface,
  year,
  limit,
  sortBy,
  order,
}: StatsLeaderboardProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  function updateParams(updates: Record<string, string>) {
    const params = new URLSearchParams(searchParams.toString());
    for (const [key, value] of Object.entries(updates)) {
      params.set(key, value);
    }
    router.push(`/stats?${params.toString()}`);
  }

  function handleSort(column: SortColumn) {
    if (sortBy === column) {
      // Toggle order
      updateParams({ order: order === "desc" ? "asc" : "desc" });
    } else {
      // New column, default desc (except matches which makes sense desc too)
      updateParams({ sort: column, order: "desc" });
    }
  }

  function SortIcon({ column }: { column: SortColumn }) {
    if (sortBy !== column) {
      return <ArrowUpDown className="ml-1 inline size-3 text-muted-foreground" />;
    }
    return order === "desc" ? (
      <ArrowDown className="ml-1 inline size-3" />
    ) : (
      <ArrowUp className="ml-1 inline size-3" />
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter controls */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Tour toggle */}
        <div className="flex gap-1">
          {tours.map((t) => (
            <Button
              key={t.value}
              variant={tour === t.value ? "default" : "outline"}
              size="sm"
              onClick={() => updateParams({ tour: t.value })}
            >
              {t.label}
            </Button>
          ))}
        </div>

        {/* Surface toggle */}
        <div className="flex gap-1">
          {surfaceOptions.map((s) => (
            <Button
              key={s.value}
              variant={surface === s.value ? "default" : "outline"}
              size="sm"
              className={surface === s.value ? s.activeClass : ""}
              onClick={() => updateParams({ surface: s.value })}
            >
              {s.label}
            </Button>
          ))}
        </div>

        {/* Year dropdown */}
        <select
          value={year}
          onChange={(e) => updateParams({ year: e.target.value })}
          className="h-8 rounded-md border border-input bg-background px-3 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring"
        >
          {yearOptions.map((y) => (
            <option key={y.value} value={y.value}>
              {y.label}
            </option>
          ))}
        </select>

        {/* Limit dropdown */}
        <select
          value={limit}
          onChange={(e) => updateParams({ limit: e.target.value })}
          className="h-8 rounded-md border border-input bg-background px-3 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring"
        >
          {limitOptions.map((l) => (
            <option key={l.value} value={l.value}>
              {l.label}
            </option>
          ))}
        </select>
      </div>

      {/* Table */}
      {data.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No stats data available for this combination.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">#</TableHead>
                <TableHead className="min-w-[140px]">Player</TableHead>
                {columns.map((col) => (
                  <TableHead
                    key={col.key}
                    className="w-24 cursor-pointer text-right select-none"
                    onClick={() => handleSort(col.key)}
                  >
                    <span className="hidden sm:inline">{col.label}</span>
                    <span className="sm:hidden">{col.shortLabel ?? col.label}</span>
                    <SortIcon column={col.key} />
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.map((row) => (
                <TableRow key={row.playerId}>
                  <TableCell className="font-mono text-muted-foreground">
                    {row.rank}
                  </TableCell>
                  <TableCell>
                    <Link
                      href={`/player/${row.slug}`}
                      className="font-medium hover:underline"
                    >
                      {row.playerName}
                    </Link>
                    {row.country && (
                      <span className="ml-2 text-xs text-muted-foreground">
                        {row.country}
                      </span>
                    )}
                  </TableCell>
                  {columns.map((col) => (
                    <TableCell
                      key={col.key}
                      className={`text-right font-mono text-sm ${
                        sortBy === col.key ? "font-semibold" : ""
                      }`}
                    >
                      {formatStat(col.key, row[col.key])}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
