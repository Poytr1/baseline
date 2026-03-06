import { prisma } from "@/lib/db";

export interface TopEloPlayer {
  rank: number;
  playerName: string;
  slug: string;
  country: string | null;
  elo: number;
}

export interface RecentResult {
  id: number;
  winnerName: string;
  loserName: string;
  winnerSlug: string;
  loserSlug: string;
  score: string;
  tournamentName: string;
  surface: string;
  date: string;
}

/**
 * Fetch top N players by current Elo for a given tour.
 * Uses the most recent date available in the elo_ratings table.
 */
export async function getTopElo(
  tour: "atp" | "wta",
  limit: number = 10
): Promise<TopEloPlayer[]> {
  try {
    // Get each player's latest Elo snapshot, filtered to active players (last 2 years)
    const cutoff = parseInt(
      new Date(Date.now() - 2 * 365 * 24 * 60 * 60 * 1000)
        .toISOString()
        .slice(0, 10)
        .replace(/-/g, ""),
      10
    );

    const rows = await prisma.$queryRawUnsafe<
      {
        name_first: string;
        name_last: string;
        slug: string;
        ioc: string | null;
        overall: number;
      }[]
    >(
      `SELECT p.name_first, p.name_last, p.slug, p.ioc, e.overall
       FROM elo_ratings e
       JOIN players p ON p.id = e.player_id
       WHERE e.tour = $1
         AND e.date = (
           SELECT MAX(e2.date) FROM elo_ratings e2
           WHERE e2.player_id = e.player_id AND e2.tour = $1
         )
         AND CAST(e.date AS INTEGER) >= $2
       ORDER BY e.overall DESC
       LIMIT $3`,
      tour,
      cutoff,
      limit
    );

    return rows.map((r, i) => ({
      rank: i + 1,
      playerName: [r.name_first, r.name_last].filter(Boolean).join(" "),
      slug: r.slug,
      country: r.ioc,
      elo: Math.round(Number(r.overall)),
    }));
  } catch {
    // Database may be unavailable during build-time static generation
    return [];
  }
}

/**
 * Fetch the most recent match results, joined with tournament and player data.
 * Orders by tournament date descending, then match id descending.
 */
export async function getRecentResults(
  limit: number = 15
): Promise<RecentResult[]> {
  try {
    const matches = await prisma.match.findMany({
      orderBy: [
        { tournament: { date: "desc" } },
        { id: "desc" },
      ],
      take: limit,
      include: {
        winner: {
          select: {
            nameFirst: true,
            nameLast: true,
            slug: true,
          },
        },
        loser: {
          select: {
            nameFirst: true,
            nameLast: true,
            slug: true,
          },
        },
        tournament: {
          select: {
            name: true,
            surface: true,
            date: true,
          },
        },
      },
    });

    return matches.map((m) => ({
      id: m.id,
      winnerName: [m.winner.nameFirst, m.winner.nameLast]
        .filter(Boolean)
        .join(" "),
      loserName: [m.loser.nameFirst, m.loser.nameLast]
        .filter(Boolean)
        .join(" "),
      winnerSlug: m.winner.slug,
      loserSlug: m.loser.slug,
      score: m.score ?? "",
      tournamentName: m.tournament.name,
      surface: m.surface ?? m.tournament.surface ?? "Unknown",
      date: m.tournament.date,
    }));
  } catch {
    // Database may be unavailable during build-time static generation
    return [];
  }
}

export type TopEloData = Awaited<ReturnType<typeof getTopElo>>;
export type RecentResultsData = Awaited<ReturnType<typeof getRecentResults>>;
