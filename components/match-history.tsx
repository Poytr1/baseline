"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { MatchHistoryEntry } from "@/lib/queries/player";

interface MatchHistoryProps {
  matches: MatchHistoryEntry[];
}

const PAGE_SIZE = 20;

function formatDate(dateStr: string): string {
  if (!dateStr || dateStr.length !== 8) return dateStr;
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const d = dateStr.substring(6, 8);
  return `${y}-${m}-${d}`;
}

function getSurfaceColor(surface: string): string {
  switch (surface.toLowerCase()) {
    case "hard":
      return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
    case "clay":
      return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200";
    case "grass":
      return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    default:
      return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200";
  }
}

export function MatchHistory({ matches }: MatchHistoryProps) {
  const [surfaceFilter, setSurfaceFilter] = useState("all");
  const [yearFilter, setYearFilter] = useState("all");
  const [levelFilter, setLevelFilter] = useState("all");
  const [page, setPage] = useState(1);

  // Extract unique years and levels from matches for filter dropdowns
  const years = useMemo(() => {
    const yrs = new Set<string>();
    for (const m of matches) {
      if (m.date.length >= 4) yrs.add(m.date.substring(0, 4));
    }
    return Array.from(yrs).sort().reverse();
  }, [matches]);

  const levels = useMemo(() => {
    const lvls = new Set<string>();
    for (const m of matches) {
      if (m.tournamentLevel) lvls.add(m.tournamentLevel);
    }
    return Array.from(lvls).sort();
  }, [matches]);

  // Apply filters
  const filtered = useMemo(() => {
    return matches.filter((m) => {
      if (surfaceFilter !== "all" && m.surface.toLowerCase() !== surfaceFilter)
        return false;
      if (yearFilter !== "all" && !m.date.startsWith(yearFilter)) return false;
      if (levelFilter !== "all" && m.tournamentLevel !== levelFilter)
        return false;
      return true;
    });
  }, [matches, surfaceFilter, yearFilter, levelFilter]);

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageStart = (currentPage - 1) * PAGE_SIZE;
  const pageEnd = pageStart + PAGE_SIZE;
  const pageMatches = filtered.slice(pageStart, pageEnd);

  // Reset page when filters change
  const handleFilterChange = (
    setter: (v: string) => void,
    value: string
  ) => {
    setter(value);
    setPage(1);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Match History</h2>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <select
          className="rounded-md border bg-background px-3 py-1.5 text-sm"
          value={yearFilter}
          onChange={(e) => handleFilterChange(setYearFilter, e.target.value)}
        >
          <option value="all">All Years</option>
          {years.map((y) => (
            <option key={y} value={y}>
              {y}
            </option>
          ))}
        </select>

        <select
          className="rounded-md border bg-background px-3 py-1.5 text-sm"
          value={surfaceFilter}
          onChange={(e) =>
            handleFilterChange(setSurfaceFilter, e.target.value)
          }
        >
          <option value="all">All Surfaces</option>
          <option value="hard">Hard</option>
          <option value="clay">Clay</option>
          <option value="grass">Grass</option>
        </select>

        <select
          className="rounded-md border bg-background px-3 py-1.5 text-sm"
          value={levelFilter}
          onChange={(e) =>
            handleFilterChange(setLevelFilter, e.target.value)
          }
        >
          <option value="all">All Levels</option>
          {levels.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
      </div>

      {/* Table */}
      {pageMatches.length === 0 ? (
        <p className="text-sm text-muted-foreground py-8 text-center">
          No matches found.
        </p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Date</TableHead>
              <TableHead>Tournament</TableHead>
              <TableHead>Surface</TableHead>
              <TableHead>Round</TableHead>
              <TableHead>Opponent</TableHead>
              <TableHead>Score</TableHead>
              <TableHead>Result</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pageMatches.map((m) => (
              <TableRow key={m.id}>
                <TableCell className="text-muted-foreground">
                  {formatDate(m.date)}
                </TableCell>
                <TableCell className="font-medium">
                  {m.tournamentName}
                </TableCell>
                <TableCell>
                  <Badge
                    variant="outline"
                    className={getSurfaceColor(m.surface)}
                  >
                    {m.surface}
                  </Badge>
                </TableCell>
                <TableCell>{m.round}</TableCell>
                <TableCell>
                  <Link
                    href={`/player/${m.opponentSlug}`}
                    className="text-primary hover:underline"
                  >
                    {m.opponentName}
                  </Link>
                </TableCell>
                <TableCell className="font-mono text-xs">
                  {m.score}
                </TableCell>
                <TableCell>
                  <Badge
                    variant={m.result === "W" ? "default" : "destructive"}
                  >
                    {m.result}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            Showing {pageStart + 1}-{Math.min(pageEnd, filtered.length)} of{" "}
            {filtered.length} matches
          </span>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage <= 1}
              onClick={() => setPage(currentPage - 1)}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage >= totalPages}
              onClick={() => setPage(currentPage + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
