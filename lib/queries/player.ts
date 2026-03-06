import { prisma } from "@/lib/db";

/**
 * Fetch a player by their unique slug, including their latest ranking and Elo rating.
 */
export async function getPlayerBySlug(slug: string) {
  const player = await prisma.player.findUnique({
    where: { slug },
    include: {
      rankings: {
        orderBy: { date: "desc" },
        take: 1,
      },
      eloRatings: {
        orderBy: { date: "desc" },
        take: 1,
      },
    },
  });

  return player;
}

/**
 * Aggregate win/loss records from matches for a given player.
 * Returns overall and per-surface breakdowns.
 */
export async function getPlayerCareerStats(playerId: number) {
  const [wins, losses] = await Promise.all([
    prisma.match.groupBy({
      by: ["surface"],
      where: { winnerId: playerId },
      _count: { id: true },
    }),
    prisma.match.groupBy({
      by: ["surface"],
      where: { loserId: playerId },
      _count: { id: true },
    }),
  ]);

  const surfaceWins: Record<string, number> = {};
  const surfaceLosses: Record<string, number> = {};
  let totalWins = 0;
  let totalLosses = 0;

  for (const w of wins) {
    const s = (w.surface ?? "Unknown").toLowerCase();
    surfaceWins[s] = (surfaceWins[s] ?? 0) + w._count.id;
    totalWins += w._count.id;
  }

  for (const l of losses) {
    const s = (l.surface ?? "Unknown").toLowerCase();
    surfaceLosses[s] = (surfaceLosses[s] ?? 0) + l._count.id;
    totalLosses += l._count.id;
  }

  function surfaceStats(surface: string) {
    const w = surfaceWins[surface] ?? 0;
    const l = surfaceLosses[surface] ?? 0;
    const total = w + l;
    return { wins: w, losses: l, pct: total > 0 ? w / total : 0 };
  }

  return {
    overall: { wins: totalWins, losses: totalLosses },
    bySurface: {
      hard: surfaceStats("hard"),
      clay: surfaceStats("clay"),
      grass: surfaceStats("grass"),
    },
  };
}

export type CareerStats = Awaited<ReturnType<typeof getPlayerCareerStats>>;

/**
 * Fetch a player's match history with tournament info.
 * Returns an array of matches ordered by tournament date descending.
 */
export async function getPlayerMatchHistory(playerId: number) {
  const [wonMatches, lostMatches] = await Promise.all([
    prisma.match.findMany({
      where: { winnerId: playerId },
      include: {
        tournament: true,
        loser: true,
      },
      orderBy: { tournament: { date: "desc" } },
    }),
    prisma.match.findMany({
      where: { loserId: playerId },
      include: {
        tournament: true,
        winner: true,
      },
      orderBy: { tournament: { date: "desc" } },
    }),
  ]);

  const matches = [
    ...wonMatches.map((m) => ({
      id: m.id,
      date: m.tournament.date,
      tournamentName: m.tournament.name,
      tournamentLevel: m.tournament.level ?? "",
      surface: m.surface ?? m.tournament.surface ?? "Unknown",
      round: m.round ?? "",
      opponentName: `${m.loser.nameFirst} ${m.loser.nameLast}`,
      opponentSlug: m.loser.slug,
      score: m.score ?? "",
      result: "W" as const,
    })),
    ...lostMatches.map((m) => ({
      id: m.id,
      date: m.tournament.date,
      tournamentName: m.tournament.name,
      tournamentLevel: m.tournament.level ?? "",
      surface: m.surface ?? m.tournament.surface ?? "Unknown",
      round: m.round ?? "",
      opponentName: `${m.winner.nameFirst} ${m.winner.nameLast}`,
      opponentSlug: m.winner.slug,
      score: m.score ?? "",
      result: "L" as const,
    })),
  ];

  // Sort by date descending (most recent first)
  matches.sort((a, b) => b.date.localeCompare(a.date));

  return matches;
}

export type MatchHistoryEntry = Awaited<
  ReturnType<typeof getPlayerMatchHistory>
>[number];

/**
 * Fetch a player's Elo rating history for charting.
 * Ordered by date ascending.
 */
export async function getPlayerEloHistory(playerId: number) {
  const ratings = await prisma.eloRating.findMany({
    where: { playerId },
    orderBy: { date: "asc" },
    select: {
      date: true,
      overall: true,
      hard: true,
      clay: true,
      grass: true,
    },
  });

  return ratings;
}

export type EloHistoryEntry = Awaited<
  ReturnType<typeof getPlayerEloHistory>
>[number];
