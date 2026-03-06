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
 * Fetch current Elo ratings by getting each player's most recent snapshot.
 * Only includes players whose latest snapshot is within the last 2 years
 * (to filter out retired players).
 */
export async function getEloRankings(
  tour: Tour = "atp",
  surface: Surface = "overall"
): Promise<RankedPlayer[]> {
  const eloCol = surface === "overall" ? "overall" : surface;

  // Use a subquery to get each player's latest date, then join to get the Elo
  const rows = await prisma.$queryRawUnsafe<
    {
      player_id: number;
      name_first: string;
      name_last: string;
      slug: string;
      ioc: string | null;
      elo: number;
    }[]
  >(
    `SELECT e.player_id, p.name_first, p.name_last, p.slug, p.ioc,
            e.${eloCol} as elo
     FROM elo_ratings e
     JOIN players p ON p.id = e.player_id
     WHERE e.tour = $1
       AND e.date = (
         SELECT MAX(e2.date) FROM elo_ratings e2
         WHERE e2.player_id = e.player_id AND e2.tour = $1
       )
       AND CAST(e.date AS INTEGER) >= $2
     ORDER BY e.${eloCol} DESC
     LIMIT 500`,
    tour,
    // 2 years ago as YYYYMMDD integer
    parseInt(
      new Date(Date.now() - 2 * 365 * 24 * 60 * 60 * 1000)
        .toISOString()
        .slice(0, 10)
        .replace(/-/g, ""),
      10
    )
  );

  return rows
    .filter((r) => r.elo !== null)
    .map((r, i) => ({
      rank: i + 1,
      playerId: r.player_id,
      playerName: `${r.name_first ?? ""} ${r.name_last ?? ""}`.trim(),
      slug: r.slug,
      country: r.ioc,
      elo: Math.round(Number(r.elo)),
    }));
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
