import { prisma } from "@/lib/db";

/**
 * Fetch all matches between two players, with tournament info.
 * Returns matches ordered by tournament date descending (most recent first).
 */
export async function getHeadToHead(player1Id: number, player2Id: number) {
  const matches = await prisma.match.findMany({
    where: {
      OR: [
        { winnerId: player1Id, loserId: player2Id },
        { winnerId: player2Id, loserId: player1Id },
      ],
    },
    include: {
      tournament: true,
      winner: {
        select: {
          id: true,
          nameFirst: true,
          nameLast: true,
          slug: true,
          ioc: true,
        },
      },
      loser: {
        select: {
          id: true,
          nameFirst: true,
          nameLast: true,
          slug: true,
          ioc: true,
        },
      },
    },
    orderBy: { tournament: { date: "desc" } },
  });

  return matches;
}

export type H2HMatch = Awaited<ReturnType<typeof getHeadToHead>>[number];

interface SurfaceRecord {
  wins: number;
  losses: number;
}

interface LevelRecord {
  wins: number;
  losses: number;
}

export interface H2HSummary {
  player1Wins: number;
  player2Wins: number;
  bySurface: Record<string, SurfaceRecord>;
  byLevel: Record<string, LevelRecord>;
}

/**
 * Compute an H2H summary from the perspective of player1.
 * "wins" means player1 won, "losses" means player2 won.
 */
export function getH2HSummary(
  matches: H2HMatch[],
  player1Id: number
): H2HSummary {
  let player1Wins = 0;
  let player2Wins = 0;
  const bySurface: Record<string, SurfaceRecord> = {};
  const byLevel: Record<string, LevelRecord> = {};

  for (const match of matches) {
    const p1Won = match.winnerId === player1Id;
    if (p1Won) {
      player1Wins++;
    } else {
      player2Wins++;
    }

    // Surface breakdown
    const surface = (
      match.surface ??
      match.tournament.surface ??
      "Unknown"
    ).toLowerCase();
    const capitalizedSurface =
      surface.charAt(0).toUpperCase() + surface.slice(1);

    if (!bySurface[capitalizedSurface]) {
      bySurface[capitalizedSurface] = { wins: 0, losses: 0 };
    }
    if (p1Won) {
      bySurface[capitalizedSurface].wins++;
    } else {
      bySurface[capitalizedSurface].losses++;
    }

    // Level breakdown
    const level = match.tournament.level ?? "Unknown";
    if (!byLevel[level]) {
      byLevel[level] = { wins: 0, losses: 0 };
    }
    if (p1Won) {
      byLevel[level].wins++;
    } else {
      byLevel[level].losses++;
    }
  }

  return { player1Wins, player2Wins, bySurface, byLevel };
}
