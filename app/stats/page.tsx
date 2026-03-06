import type { Metadata } from "next";
import { Suspense } from "react";
import {
  getStatsLeaderboard,
  type Tour,
  type StatsSurface,
  type SortColumn,
} from "@/lib/queries/stats";
import { StatsLeaderboard } from "@/components/stats-leaderboard";

export const revalidate = 3600;

interface PageProps {
  searchParams: Promise<{
    tour?: string;
    surface?: string;
    year?: string;
    limit?: string;
    sort?: string;
    order?: string;
  }>;
}

function isValidTour(value: string | undefined): value is Tour {
  return value === "atp" || value === "wta";
}

function isValidSurface(value: string | undefined): value is StatsSurface {
  return value === "all" || value === "hard" || value === "clay" || value === "grass";
}

const VALID_SORT_COLUMNS = new Set<SortColumn>([
  "aceRate",
  "dfRate",
  "firstServeInPct",
  "firstServeWonPct",
  "secondServeWonPct",
  "bpSavedPct",
  "matches",
]);

function isValidSortColumn(value: string | undefined): value is SortColumn {
  return VALID_SORT_COLUMNS.has(value as SortColumn);
}

function isValidOrder(value: string | undefined): value is "asc" | "desc" {
  return value === "asc" || value === "desc";
}

function isValidYear(value: string | undefined): boolean {
  if (!value || value === "all") return true;
  const num = parseInt(value, 10);
  return !isNaN(num) && num >= 1968 && num <= 2030 && String(num) === value;
}

function isValidLimit(value: string | undefined): boolean {
  return value === "50" || value === "100";
}

export async function generateMetadata({
  searchParams,
}: PageProps): Promise<Metadata> {
  const params = await searchParams;
  const tour = isValidTour(params.tour) ? params.tour : "atp";
  const tourLabel = tour.toUpperCase();

  return {
    title: `${tourLabel} Stats Leaderboards — tennisconcrete`,
  };
}

async function StatsContent({ searchParams }: PageProps) {
  const params = await searchParams;

  const tour: Tour = isValidTour(params.tour) ? params.tour : "atp";
  const surface: StatsSurface = isValidSurface(params.surface)
    ? params.surface
    : "all";
  const year = isValidYear(params.year) ? (params.year ?? "all") : "all";
  const limit = isValidLimit(params.limit) ? parseInt(params.limit!, 10) : 50;
  const sortBy: SortColumn = isValidSortColumn(params.sort)
    ? params.sort
    : "firstServeWonPct";
  const order: "asc" | "desc" = isValidOrder(params.order)
    ? params.order
    : "desc";

  const data = await getStatsLeaderboard({
    tour,
    surface,
    year,
    limit,
    sortBy,
    order,
  });

  return (
    <StatsLeaderboard
      data={data}
      tour={tour}
      surface={surface}
      year={year}
      limit={limit}
      sortBy={sortBy}
      order={order}
    />
  );
}

export default async function StatsPage({ searchParams }: PageProps) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Stats Leaderboards
        </h1>
        <p className="text-muted-foreground">
          Aggregated serve and return statistics across all matches
        </p>
      </div>
      <Suspense
        fallback={
          <p className="py-8 text-center text-sm text-muted-foreground">
            Loading stats...
          </p>
        }
      >
        <StatsContent searchParams={searchParams} />
      </Suspense>
    </div>
  );
}
