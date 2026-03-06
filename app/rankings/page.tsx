import type { Metadata } from "next";
import { Suspense } from "react";
import {
  getEloRankings,
  getPlayerEloSparkline,
  type Tour,
  type Surface,
  type SparklinePoint,
} from "@/lib/queries/rankings";
import { EloRankingsTable } from "@/components/elo-rankings-table";

export const revalidate = 3600;

interface PageProps {
  searchParams: Promise<{ tour?: string; surface?: string }>;
}

function isValidTour(value: string | undefined): value is Tour {
  return value === "atp" || value === "wta";
}

function isValidSurface(value: string | undefined): value is Surface {
  return (
    value === "overall" ||
    value === "hard" ||
    value === "clay" ||
    value === "grass"
  );
}

export async function generateMetadata({
  searchParams,
}: PageProps): Promise<Metadata> {
  const params = await searchParams;
  const tour = isValidTour(params.tour) ? params.tour : "atp";
  const surface = isValidSurface(params.surface) ? params.surface : "overall";

  const tourLabel = tour.toUpperCase();
  const surfaceLabel = surface === "overall" ? "" : ` (${surface.charAt(0).toUpperCase() + surface.slice(1)})`;

  return {
    title: `${tourLabel} Elo Rankings${surfaceLabel} — tennisconcrete`,
  };
}

async function RankingsContent({ searchParams }: PageProps) {
  const params = await searchParams;
  const tour: Tour = isValidTour(params.tour) ? params.tour : "atp";
  const surface: Surface = isValidSurface(params.surface)
    ? params.surface
    : "overall";

  const rankings = await getEloRankings(tour, surface);

  // Fetch sparkline data for top 100 players
  const top100 = rankings.slice(0, 100);
  const sparklineEntries = await Promise.all(
    top100.map(async (player) => {
      const data = await getPlayerEloSparkline(player.playerId, 10);
      return [player.playerId, data] as [number, SparklinePoint[]];
    })
  );
  const sparklines: Record<number, SparklinePoint[]> = Object.fromEntries(
    sparklineEntries
  );

  return (
    <EloRankingsTable
      rankings={rankings}
      sparklines={sparklines}
      tour={tour}
      surface={surface}
    />
  );
}

export default async function RankingsPage({ searchParams }: PageProps) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Elo Rankings</h1>
        <p className="text-muted-foreground">
          Player rankings based on Elo rating calculations
        </p>
      </div>
      <Suspense
        fallback={
          <p className="py-8 text-center text-sm text-muted-foreground">
            Loading rankings...
          </p>
        }
      >
        <RankingsContent searchParams={searchParams} />
      </Suspense>
    </div>
  );
}
