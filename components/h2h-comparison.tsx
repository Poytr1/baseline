"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { H2HData } from "@/app/h2h/page";

interface H2HComparisonProps {
  data: H2HData;
}

function getSurfaceColor(surface: string): string {
  switch (surface.toLowerCase()) {
    case "hard":
      return "#2563eb";
    case "clay":
      return "#d97706";
    case "grass":
      return "#16a34a";
    default:
      return "#6b7280";
  }
}

function getSurfaceBadgeClass(surface: string): string {
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

function ProportionalBar({
  leftWins,
  rightWins,
}: {
  leftWins: number;
  rightWins: number;
}) {
  const total = leftWins + rightWins;
  if (total === 0) return null;
  const leftPct = (leftWins / total) * 100;
  const rightPct = (rightWins / total) * 100;

  return (
    <div className="flex h-8 w-full overflow-hidden rounded-md">
      <div
        className="flex items-center justify-center text-sm font-bold text-white transition-all"
        style={{
          width: `${leftPct}%`,
          backgroundColor: "#16a34a",
          minWidth: leftWins > 0 ? "2rem" : "0",
        }}
      >
        {leftWins > 0 && leftWins}
      </div>
      <div
        className="flex items-center justify-center text-sm font-bold text-white transition-all"
        style={{
          width: `${rightPct}%`,
          backgroundColor: "#dc2626",
          minWidth: rightWins > 0 ? "2rem" : "0",
        }}
      >
        {rightWins > 0 && rightWins}
      </div>
    </div>
  );
}

function RecordRow({
  label,
  p1Wins,
  p2Wins,
  color,
  badgeClass,
}: {
  label: string;
  p1Wins: number;
  p2Wins: number;
  color?: string;
  badgeClass?: string;
}) {
  const total = p1Wins + p2Wins;
  if (total === 0) return null;

  return (
    <div className="flex items-center gap-3">
      <Badge
        variant="outline"
        className={`w-20 justify-center text-xs ${badgeClass ?? ""}`}
        style={color ? { borderColor: color } : undefined}
      >
        {label}
      </Badge>
      <div className="flex flex-1 items-center gap-2">
        <span className="w-6 text-right text-sm font-semibold">{p1Wins}</span>
        <div className="flex h-4 flex-1 overflow-hidden rounded-sm">
          {total > 0 && (
            <>
              <div
                className="transition-all"
                style={{
                  width: `${(p1Wins / total) * 100}%`,
                  backgroundColor: "#16a34a",
                }}
              />
              <div
                className="transition-all"
                style={{
                  width: `${(p2Wins / total) * 100}%`,
                  backgroundColor: "#dc2626",
                }}
              />
            </>
          )}
        </div>
        <span className="w-6 text-left text-sm font-semibold">{p2Wins}</span>
      </div>
    </div>
  );
}

export function H2HComparison({ data }: H2HComparisonProps) {
  const { player1, player2, summary } = data;

  if (!summary) return null;

  const p1Name = `${player1.nameFirst} ${player1.nameLast}`;
  const p2Name = `${player2.nameFirst} ${player2.nameLast}`;

  // Sort surfaces in preferred order
  const surfaceOrder = ["Hard", "Clay", "Grass"];
  const surfaces = Object.keys(summary.bySurface).sort((a, b) => {
    const ai = surfaceOrder.indexOf(a);
    const bi = surfaceOrder.indexOf(b);
    if (ai === -1 && bi === -1) return a.localeCompare(b);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });

  const levels = Object.keys(summary.byLevel).sort();

  return (
    <div className="space-y-6">
      {/* Player cards side by side */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardContent className="flex flex-col items-center gap-2 pt-6">
            <Link
              href={`/player/${player1.slug}`}
              className="text-lg font-bold hover:underline md:text-xl"
            >
              {p1Name}
            </Link>
            {player1.ioc && (
              <Badge variant="outline" className="text-xs">
                {player1.ioc}
              </Badge>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex flex-col items-center gap-2 pt-6">
            <Link
              href={`/player/${player2.slug}`}
              className="text-lg font-bold hover:underline md:text-xl"
            >
              {p2Name}
            </Link>
            {player2.ioc && (
              <Badge variant="outline" className="text-xs">
                {player2.ioc}
              </Badge>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Overall Record */}
      <Card>
        <CardHeader>
          <CardTitle className="text-center text-lg">Overall Record</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between text-sm font-medium text-muted-foreground">
            <span>{p1Name}</span>
            <span>{p2Name}</span>
          </div>
          <ProportionalBar
            leftWins={summary.player1Wins}
            rightWins={summary.player2Wins}
          />
          <p className="text-center text-sm text-muted-foreground">
            {summary.player1Wins + summary.player2Wins} matches played
          </p>
        </CardContent>
      </Card>

      {/* Surface Breakdown */}
      {surfaces.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">By Surface</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {surfaces.map((surface) => {
              const record = summary.bySurface[surface];
              return (
                <RecordRow
                  key={surface}
                  label={surface}
                  p1Wins={record.wins}
                  p2Wins={record.losses}
                  color={getSurfaceColor(surface)}
                  badgeClass={getSurfaceBadgeClass(surface)}
                />
              );
            })}
          </CardContent>
        </Card>
      )}

      {/* Level Breakdown */}
      {levels.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">By Tournament Level</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {levels.map((level) => {
              const record = summary.byLevel[level];
              return (
                <RecordRow
                  key={level}
                  label={level}
                  p1Wins={record.wins}
                  p2Wins={record.losses}
                />
              );
            })}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
