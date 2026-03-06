"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Sparkline } from "@/components/sparkline";
import type {
  RankedPlayer,
  SparklinePoint,
  Tour,
  Surface,
} from "@/lib/queries/rankings";

interface EloRankingsTableProps {
  rankings: RankedPlayer[];
  sparklines: Record<number, SparklinePoint[]>;
  tour: Tour;
  surface: Surface;
}

const tours: { value: Tour; label: string }[] = [
  { value: "atp", label: "ATP" },
  { value: "wta", label: "WTA" },
];

const surfaces: { value: Surface; label: string; activeClass: string }[] = [
  { value: "overall", label: "Overall", activeClass: "bg-primary text-primary-foreground hover:bg-primary/90" },
  { value: "hard", label: "Hard", activeClass: "bg-blue-600 text-white hover:bg-blue-700" },
  { value: "clay", label: "Clay", activeClass: "bg-amber-600 text-white hover:bg-amber-700" },
  { value: "grass", label: "Grass", activeClass: "bg-green-600 text-white hover:bg-green-700" },
];

export function EloRankingsTable({
  rankings,
  sparklines,
  tour,
  surface,
}: EloRankingsTableProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [search, setSearch] = React.useState("");

  function updateParams(key: string, value: string) {
    const params = new URLSearchParams(searchParams.toString());
    params.set(key, value);
    router.push(`/rankings?${params.toString()}`);
  }

  const filteredRankings = React.useMemo(() => {
    if (!search.trim()) return rankings;
    const q = search.toLowerCase();
    return rankings.filter(
      (r) =>
        r.playerName.toLowerCase().includes(q) ||
        (r.country && r.country.toLowerCase().includes(q))
    );
  }, [rankings, search]);

  return (
    <div className="space-y-4">
      {/* Tour toggles */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex gap-1">
          {tours.map((t) => (
            <Button
              key={t.value}
              variant={tour === t.value ? "default" : "outline"}
              size="sm"
              onClick={() => updateParams("tour", t.value)}
            >
              {t.label}
            </Button>
          ))}
        </div>

        {/* Surface toggles */}
        <div className="flex gap-1">
          {surfaces.map((s) => (
            <Button
              key={s.value}
              variant={surface === s.value ? "default" : "outline"}
              size="sm"
              className={surface === s.value ? s.activeClass : ""}
              onClick={() => updateParams("surface", s.value)}
            >
              {s.label}
            </Button>
          ))}
        </div>

        {/* Search */}
        <div className="relative ml-auto w-full sm:w-64">
          <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Filter players..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-8"
          />
        </div>
      </div>

      {/* Rankings table */}
      {filteredRankings.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No players found.
        </p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">#</TableHead>
              <TableHead>Player</TableHead>
              <TableHead className="w-20">Country</TableHead>
              <TableHead className="w-24 text-right">Elo</TableHead>
              <TableHead className="w-24 text-center">Trend</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredRankings.map((player) => (
              <TableRow key={player.playerId}>
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
                <TableCell className="text-center">
                  <div className="flex justify-center">
                    <Sparkline data={sparklines[player.playerId] ?? []} />
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
