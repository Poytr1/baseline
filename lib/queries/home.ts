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
    // Find the latest date for this tour
    const latestEntry = await prisma.eloRating.findFirst({
      where: { tour },
      orderBy: { date: "desc" },
      select: { date: true },
    });

    if (!latestEntry) {
      return [];
    }

    const ratings = await prisma.eloRating.findMany({
      where: {
        tour,
        date: latestEntry.date,
      },
      include: {
        player: {
          select: {
            nameFirst: true,
            nameLast: true,
            slug: true,
            ioc: true,
          },
        },
      },
      orderBy: { overall: "desc" },
      take: limit,
    });

    return ratings.map((r, i) => ({
      rank: i + 1,
      playerName: [r.player.nameFirst, r.player.nameLast]
        .filter(Boolean)
        .join(" "),
      slug: r.player.slug,
      country: r.player.ioc,
      elo: Math.round(r.overall),
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
