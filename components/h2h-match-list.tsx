"use client";

import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { H2HData } from "@/app/h2h/page";

interface H2HMatchListProps {
  data: H2HData;
}

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

export function H2HMatchList({ data }: H2HMatchListProps) {
  const { player1, matches } = data;

  if (matches.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No matches found between these players.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Match History</h2>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date</TableHead>
            <TableHead>Tournament</TableHead>
            <TableHead>Surface</TableHead>
            <TableHead>Round</TableHead>
            <TableHead>Winner</TableHead>
            <TableHead>Score</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {matches.map((m) => {
            const p1Won = m.winnerId === player1.id;
            return (
              <TableRow key={m.id}>
                <TableCell className="text-muted-foreground whitespace-nowrap">
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
                  <span
                    className={
                      p1Won
                        ? "font-semibold text-green-600 dark:text-green-400"
                        : "font-semibold text-red-600 dark:text-red-400"
                    }
                  >
                    {m.winnerName}
                  </span>
                </TableCell>
                <TableCell className="font-mono text-xs">{m.score}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
