"use client";

import * as React from "react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { TopEloData } from "@/lib/queries/home";

interface TopEloTableProps {
  atpData: TopEloData;
  wtaData: TopEloData;
}

type Tour = "atp" | "wta";

const tours: { value: Tour; label: string }[] = [
  { value: "atp", label: "ATP" },
  { value: "wta", label: "WTA" },
];

export function TopEloTable({ atpData, wtaData }: TopEloTableProps) {
  const [activeTour, setActiveTour] = React.useState<Tour>("atp");

  const data = activeTour === "atp" ? atpData : wtaData;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Top Elo Ratings</h2>
        <div className="flex gap-1">
          {tours.map((t) => (
            <Button
              key={t.value}
              variant={activeTour === t.value ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveTour(t.value)}
            >
              {t.label}
            </Button>
          ))}
        </div>
      </div>

      {data.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No Elo data available yet.
        </p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">#</TableHead>
              <TableHead>Player</TableHead>
              <TableHead className="w-16">IOC</TableHead>
              <TableHead className="w-20 text-right">Elo</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((player) => (
              <TableRow key={`${activeTour}-${player.slug}`}>
                <TableCell className="font-mono text-muted-foreground">
                  {player.rank}
                </TableCell>
                <TableCell>
                  <Link
                    href={`/player/${player.slug}`}
                    className="font-medium hover:underline"
                  >
                    {player.playerName}
                  </Link>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {player.country ?? "---"}
                </TableCell>
                <TableCell className="text-right font-mono font-semibold">
                  {player.elo}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      <div className="text-center">
        <Link
          href={`/rankings?tour=${activeTour}`}
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          View full rankings &rarr;
        </Link>
      </div>
    </div>
  );
}
