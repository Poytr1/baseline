import { PrismaClient } from "@prisma/client";
import { calculateNewElo, INITIAL_ELO } from "../lib/elo";

const prisma = new PrismaClient();

// ── Types ────────────────────────────────────────────────────────────

type Surface = "hard" | "clay" | "grass";

interface PlayerElo {
  overall: number;
  hard: number;
  clay: number;
  grass: number;
}

// ── Helpers ──────────────────────────────────────────────────────────

function normalizeSurface(raw: string | null): Surface | null {
  if (!raw) return null;
  const lower = raw.toLowerCase().trim();
  if (lower === "hard") return "hard";
  if (lower === "clay") return "clay";
  if (lower === "grass") return "grass";
  // Carpet and other rare surfaces are ignored for surface-specific Elo
  return null;
}

function getOrCreate(
  map: Map<number, PlayerElo>,
  playerId: number
): PlayerElo {
  let elo = map.get(playerId);
  if (!elo) {
    elo = {
      overall: INITIAL_ELO,
      hard: INITIAL_ELO,
      clay: INITIAL_ELO,
      grass: INITIAL_ELO,
    };
    map.set(playerId, elo);
  }
  return elo;
}

// ── Main ─────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("=== Elo Computation ===\n");

  // Clear existing Elo ratings
  console.log("Clearing existing elo_ratings...");
  await prisma.eloRating.deleteMany();

  // Fetch all matches ordered by tournament date, then match_num
  console.log("Fetching all matches (ordered by tournament date, match_num)...");
  const matches = await prisma.match.findMany({
    include: {
      tournament: {
        select: { date: true },
      },
    },
    orderBy: [
      { tournament: { date: "asc" } },
      { matchNum: "asc" },
    ],
  });

  console.log(`  Total matches: ${matches.length}`);

  // In-memory Elo maps keyed by playerId
  const eloMap = new Map<number, PlayerElo>();

  // Track which players changed Elo during the current tournament
  // so we can snapshot only those players when the tournament changes
  let changedPlayerIds = new Set<number>();
  let currentTourneyId: string | null = null;
  let currentTourneyDate: string | null = null;
  let currentTour: string | null = null;

  let snapshotsInserted = 0;
  let matchesProcessed = 0;

  for (const match of matches) {
    const tourneyId = match.tourneyId;
    const tourneyDate = match.tournament.date;

    // When tournament changes, snapshot Elo for all affected players
    if (currentTourneyId !== null && tourneyId !== currentTourneyId) {
      snapshotsInserted += await insertSnapshots(
        eloMap,
        changedPlayerIds,
        currentTourneyDate!,
        currentTour!
      );
      changedPlayerIds = new Set<number>();
    }

    currentTourneyId = tourneyId;
    currentTourneyDate = tourneyDate;
    currentTour = match.tour;

    // Look up current Elo for winner and loser
    const winnerElo = getOrCreate(eloMap, match.winnerId);
    const loserElo = getOrCreate(eloMap, match.loserId);

    // Calculate new overall Elo
    const overallResult = calculateNewElo(
      winnerElo.overall,
      loserElo.overall
    );
    winnerElo.overall = overallResult.winnerNew;
    loserElo.overall = overallResult.loserNew;

    // Calculate surface-specific Elo if applicable
    const surface = normalizeSurface(match.surface);
    if (surface) {
      const surfaceResult = calculateNewElo(
        winnerElo[surface],
        loserElo[surface]
      );
      winnerElo[surface] = surfaceResult.winnerNew;
      loserElo[surface] = surfaceResult.loserNew;
    }

    changedPlayerIds.add(match.winnerId);
    changedPlayerIds.add(match.loserId);

    matchesProcessed++;
    if (matchesProcessed % 50_000 === 0) {
      console.log(`  Processed ${matchesProcessed} matches...`);
    }
  }

  // Insert final snapshots for the last tournament
  if (currentTourneyId !== null && changedPlayerIds.size > 0) {
    snapshotsInserted += await insertSnapshots(
      eloMap,
      changedPlayerIds,
      currentTourneyDate!,
      currentTour!
    );
  }

  console.log(`\n  Matches processed: ${matchesProcessed}`);
  console.log(`  Elo snapshots inserted: ${snapshotsInserted}`);
  console.log(`  Unique players tracked: ${eloMap.size}`);
  console.log("\n=== Elo computation complete ===");
}

async function insertSnapshots(
  eloMap: Map<number, PlayerElo>,
  playerIds: Set<number>,
  date: string,
  tour: string
): Promise<number> {
  const data = Array.from(playerIds).map((playerId) => {
    const elo = eloMap.get(playerId)!;
    return {
      playerId,
      date,
      overall: Math.round(elo.overall * 100) / 100,
      hard: Math.round(elo.hard * 100) / 100,
      clay: Math.round(elo.clay * 100) / 100,
      grass: Math.round(elo.grass * 100) / 100,
      tour,
    };
  });

  const BATCH = 1000;
  let inserted = 0;

  for (let i = 0; i < data.length; i += BATCH) {
    const batch = data.slice(i, i + BATCH);
    const result = await prisma.eloRating.createMany({
      data: batch,
      skipDuplicates: true,
    });
    inserted += result.count;
  }

  return inserted;
}

main()
  .catch((err) => {
    console.error("Elo computation failed:", err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
