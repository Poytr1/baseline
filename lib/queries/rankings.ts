import { prisma } from "@/lib/db";

export type Surface = "overall" | "hard" | "clay" | "grass";
export type Tour = "atp" | "wta";

export interface RankedPlayer {
  rank: number;
  playerId: number;
  playerName: string;
  slug: string;
  country: string | null;
  elo: number;
}

export interface SparklinePoint {
  date: string;
  elo: number;
}

/**
 * Fetch latest Elo ratings for all players on a given tour, sorted descending.
 * For surface filtering, returns the surface-specific Elo instead of overall.
 * "Latest" means the most recent date in the elo_ratings table for that tour.
 */
export async function getEloRankings(
  tour: Tour = "atp",
  surface: Surface = "overall"
): Promise<RankedPlayer[]> {
  // Find the latest date for this tour
  const latestEntry = await prisma.eloRating.findFirst({
    where: { tour },
    orderBy: { date: "desc" },
    select: { date: true },
  });

  if (!latestEntry) {
    return [];
  }

  const latestDate = latestEntry.date;

  // Determine which Elo column to use
  const eloField = surface === "overall" ? "overall" : surface;

  // Fetch all ratings for the latest date
  const ratings = await prisma.eloRating.findMany({
    where: {
      tour,
      date: latestDate,
    },
    include: {
      player: {
        select: {
          id: true,
          nameFirst: true,
          nameLast: true,
          slug: true,
          ioc: true,
        },
      },
    },
    orderBy: { overall: "desc" },
  });

  // Map and filter: for surface-specific Elo, skip players whose surface Elo is null
  const results: RankedPlayer[] = [];
  for (const r of ratings) {
    const eloValue = r[eloField];
    if (eloValue === null || eloValue === undefined) continue;

    results.push({
      rank: 0, // will be assigned below
      playerId: r.playerId,
      playerName: `${r.player.nameFirst} ${r.player.nameLast}`,
      slug: r.player.slug,
      country: r.player.ioc,
      elo: Math.round(eloValue),
    });
  }

  // Sort by the selected Elo descending and assign ranks
  results.sort((a, b) => b.elo - a.elo);
  results.forEach((r, i) => {
    r.rank = i + 1;
  });

  return results;
}

/**
 * Fetch the last N Elo snapshots for a player (for sparkline mini-chart).
 * Returns data ordered by date ascending.
 */
export async function getPlayerEloSparkline(
  playerId: number,
  limit: number = 10
): Promise<SparklinePoint[]> {
  const ratings = await prisma.eloRating.findMany({
    where: { playerId },
    orderBy: { date: "desc" },
    take: limit,
    select: {
      date: true,
      overall: true,
    },
  });

  // Reverse so they are in chronological order
  return ratings.reverse().map((r) => ({
    date: r.date,
    elo: Math.round(r.overall),
  }));
}

export type EloRankingsResult = Awaited<ReturnType<typeof getEloRankings>>;
export type SparklineData = Awaited<ReturnType<typeof getPlayerEloSparkline>>;
