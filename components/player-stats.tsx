import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { CareerStats } from "@/lib/queries/player";

interface PlayerStatsProps {
  stats: CareerStats;
}

interface SurfaceCardProps {
  title: string;
  wins: number;
  losses: number;
  pct: number;
  colorClass: string;
}

function SurfaceCard({ title, wins, losses, pct, colorClass }: SurfaceCardProps) {
  const total = wins + losses;
  return (
    <Card>
      <CardHeader>
        <CardTitle className={`text-base ${colorClass}`}>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          <div className="text-2xl font-bold">
            {wins}W - {losses}L
          </div>
          <div className="text-sm text-muted-foreground">
            {total > 0
              ? `${(pct * 100).toFixed(1)}% win rate`
              : "No matches"}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function PlayerStats({ stats }: PlayerStatsProps) {
  const { overall, bySurface } = stats;
  const overallTotal = overall.wins + overall.losses;
  const overallPct =
    overallTotal > 0 ? ((overall.wins / overallTotal) * 100).toFixed(1) : "0.0";

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Career Statistics</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Overall */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Overall</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <div className="text-2xl font-bold">
                {overall.wins}W - {overall.losses}L
              </div>
              <div className="text-sm text-muted-foreground">
                {overallTotal > 0
                  ? `${overallPct}% win rate`
                  : "No matches"}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Hard */}
        <SurfaceCard
          title="Hard"
          wins={bySurface.hard.wins}
          losses={bySurface.hard.losses}
          pct={bySurface.hard.pct}
          colorClass="text-blue-600 dark:text-blue-400"
        />

        {/* Clay */}
        <SurfaceCard
          title="Clay"
          wins={bySurface.clay.wins}
          losses={bySurface.clay.losses}
          pct={bySurface.clay.pct}
          colorClass="text-amber-600 dark:text-amber-400"
        />

        {/* Grass */}
        <SurfaceCard
          title="Grass"
          wins={bySurface.grass.wins}
          losses={bySurface.grass.losses}
          pct={bySurface.grass.pct}
          colorClass="text-green-600 dark:text-green-400"
        />
      </div>
    </div>
  );
}
