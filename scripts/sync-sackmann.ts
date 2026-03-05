import Papa from "papaparse";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

// ── Constants ──────────────────────────────────────────────────────────

const ATP_BASE =
  "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master";
const WTA_BASE =
  "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master";

const CURRENT_YEAR = new Date().getFullYear();
const START_YEAR = 1968;

// WTA player IDs are offset by this amount so they never collide with ATP IDs
// in the shared `players` table (both tours use plain integer IDs in the CSV).
const WTA_PLAYER_ID_OFFSET = 1_000_000_000;

const RANKING_DECADES = ["70s", "80s", "90s", "00s", "10s", "20s", "current"];

// ── Helpers ────────────────────────────────────────────────────────────

function slugify(first: string, last: string, id: number): string {
  const raw = `${first}-${last}-${id}`.toLowerCase().replace(/[^a-z0-9-]/g, "");
  return raw;
}

function intOrNull(val: string | undefined): number | null {
  if (val === undefined || val === null || val === "") return null;
  const n = parseInt(val, 10);
  return Number.isNaN(n) ? null : n;
}

function floatOrNull(val: string | undefined): number | null {
  if (val === undefined || val === null || val === "") return null;
  const n = parseFloat(val);
  return Number.isNaN(n) ? null : n;
}

/** Apply the WTA offset when tour is "wta" so player IDs are unique. */
function resolvePlayerId(rawId: string | number, tour: string): number {
  const base = typeof rawId === "string" ? parseInt(rawId, 10) : rawId;
  if (Number.isNaN(base)) return -1; // sentinel for filtering
  return tour === "wta" ? base + WTA_PLAYER_ID_OFFSET : base;
}

/** Prefix tournament IDs with tour so they never collide. */
function resolveTourneyId(rawId: string, tour: string): string {
  return `${tour}-${rawId}`;
}

async function fetchCsv<T>(url: string): Promise<T[]> {
  const res = await fetch(url);
  if (!res.ok) {
    console.warn(`  [SKIP] ${url} → ${res.status}`);
    return [];
  }
  const text = await res.text();
  const parsed = Papa.parse<T>(text, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false, // we handle conversions ourselves
  });
  if (parsed.errors.length > 0) {
    console.warn(`  [CSV WARN] ${url}: ${parsed.errors.length} parse errors`);
    for (const err of parsed.errors.slice(0, 5)) {
      console.warn(`    row ${err.row}: ${err.message}`);
    }
  }
  return parsed.data;
}

// ── Players ────────────────────────────────────────────────────────────

interface RawPlayer {
  player_id: string;
  name_first: string;
  name_last: string;
  hand: string;
  dob: string;
  ioc: string;
  height: string;
  wikidata_id: string;
}

async function syncPlayers(tour: "atp" | "wta"): Promise<void> {
  const base = tour === "atp" ? ATP_BASE : WTA_BASE;
  const prefix = tour === "atp" ? "atp" : "wta";
  const url = `${base}/${prefix}_players.csv`;
  console.log(`Fetching players: ${url}`);

  const rows = await fetchCsv<RawPlayer>(url);
  console.log(`  Parsed ${rows.length} ${tour.toUpperCase()} players`);

  const BATCH = 500;
  let upserted = 0;

  for (let i = 0; i < rows.length; i += BATCH) {
    const batch = rows.slice(i, i + BATCH);
    const valid = batch.filter((r) => resolvePlayerId(r.player_id, tour) !== -1);
    const ops = valid.map((r) => {
      const id = resolvePlayerId(r.player_id, tour);
      const nameFirst = (r.name_first ?? "").trim();
      const nameLast = (r.name_last ?? "").trim();
      const slug = slugify(nameFirst, nameLast, id);

      return prisma.player.upsert({
        where: { id },
        update: {
          nameFirst,
          nameLast,
          hand: r.hand || null,
          dob: r.dob || null,
          ioc: r.ioc || null,
          height: intOrNull(r.height),
          wikidataId: r.wikidata_id || null,
          tour,
          slug,
        },
        create: {
          id,
          nameFirst,
          nameLast,
          hand: r.hand || null,
          dob: r.dob || null,
          ioc: r.ioc || null,
          height: intOrNull(r.height),
          wikidataId: r.wikidata_id || null,
          tour,
          slug,
        },
      });
    });

    await prisma.$transaction(ops);
    upserted += batch.length;
  }

  console.log(`  Upserted ${upserted} ${tour.toUpperCase()} players`);
}

// ── Tournaments & Matches ──────────────────────────────────────────────

interface RawMatch {
  tourney_id: string;
  tourney_name: string;
  surface: string;
  draw_size: string;
  tourney_level: string;
  tourney_date: string;
  match_num: string;
  winner_id: string;
  winner_seed: string;
  winner_entry: string;
  winner_name: string;
  winner_hand: string;
  winner_ht: string;
  winner_ioc: string;
  winner_age: string;
  loser_id: string;
  loser_seed: string;
  loser_entry: string;
  loser_name: string;
  loser_hand: string;
  loser_ht: string;
  loser_ioc: string;
  loser_age: string;
  score: string;
  best_of: string;
  round: string;
  minutes: string;
  w_ace: string;
  w_df: string;
  w_svpt: string;
  w_1stIn: string;
  w_1stWon: string;
  w_2ndWon: string;
  w_SvGms: string;
  w_bpSaved: string;
  w_bpFaced: string;
  l_ace: string;
  l_df: string;
  l_svpt: string;
  l_1stIn: string;
  l_1stWon: string;
  l_2ndWon: string;
  l_SvGms: string;
  l_bpSaved: string;
  l_bpFaced: string;
  winner_rank: string;
  winner_rank_points: string;
  loser_rank: string;
  loser_rank_points: string;
}

async function syncMatchesForYear(
  year: number,
  tour: "atp" | "wta"
): Promise<void> {
  const base = tour === "atp" ? ATP_BASE : WTA_BASE;
  const prefix = tour === "atp" ? "atp" : "wta";
  const url = `${base}/${prefix}_matches_${year}.csv`;
  console.log(`Fetching matches: ${url}`);

  const rows = await fetchCsv<RawMatch>(url);
  if (rows.length === 0) return;
  console.log(`  Parsed ${rows.length} matches for ${tour.toUpperCase()} ${year}`);

  // ── Extract unique tournaments ──
  const tourneyMap = new Map<
    string,
    {
      id: string;
      name: string;
      surface: string | null;
      drawSize: number | null;
      level: string | null;
      date: string;
      tour: string;
    }
  >();

  for (const r of rows) {
    const tid = resolveTourneyId(r.tourney_id, tour);
    if (!tourneyMap.has(tid)) {
      tourneyMap.set(tid, {
        id: tid,
        name: r.tourney_name ?? "",
        surface: r.surface || null,
        drawSize: intOrNull(r.draw_size),
        level: r.tourney_level || null,
        date: r.tourney_date ?? "",
        tour,
      });
    }
  }

  // Upsert tournaments
  const tourneyOps = Array.from(tourneyMap.values()).map((t) =>
    prisma.tournament.upsert({
      where: { id: t.id },
      update: {
        name: t.name,
        surface: t.surface,
        drawSize: t.drawSize,
        level: t.level,
        date: t.date,
        tour: t.tour,
      },
      create: t,
    })
  );
  await prisma.$transaction(tourneyOps);
  console.log(`  Upserted ${tourneyMap.size} tournaments`);

  // ── Upsert matches in batches ──
  const BATCH = 500;
  let upserted = 0;

  for (let i = 0; i < rows.length; i += BATCH) {
    const batch = rows.slice(i, i + BATCH);
    const valid = batch.filter((r) => {
      const wId = resolvePlayerId(r.winner_id, tour);
      const lId = resolvePlayerId(r.loser_id, tour);
      const mn = parseInt(r.match_num, 10);
      return wId !== -1 && lId !== -1 && !Number.isNaN(mn);
    });
    const ops = valid.map((r) => {
      const tourneyId = resolveTourneyId(r.tourney_id, tour);
      const matchNum = parseInt(r.match_num, 10);
      const winnerId = resolvePlayerId(r.winner_id, tour);
      const loserId = resolvePlayerId(r.loser_id, tour);

      const data = {
        tourneyId,
        matchNum,
        winnerId,
        loserId,
        score: r.score || null,
        bestOf: intOrNull(r.best_of),
        round: r.round || null,
        minutes: intOrNull(r.minutes),
        surface: r.surface || null,
        tour,
        // Winner stats
        wAce: intOrNull(r.w_ace),
        wDf: intOrNull(r.w_df),
        wSvpt: intOrNull(r.w_svpt),
        w1stIn: intOrNull(r.w_1stIn),
        w1stWon: intOrNull(r.w_1stWon),
        w2ndWon: intOrNull(r.w_2ndWon),
        wSvGms: intOrNull(r.w_SvGms),
        wBpSaved: intOrNull(r.w_bpSaved),
        wBpFaced: intOrNull(r.w_bpFaced),
        // Loser stats
        lAce: intOrNull(r.l_ace),
        lDf: intOrNull(r.l_df),
        lSvpt: intOrNull(r.l_svpt),
        l1stIn: intOrNull(r.l_1stIn),
        l1stWon: intOrNull(r.l_1stWon),
        l2ndWon: intOrNull(r.l_2ndWon),
        lSvGms: intOrNull(r.l_SvGms),
        lBpSaved: intOrNull(r.l_bpSaved),
        lBpFaced: intOrNull(r.l_bpFaced),
        // Rankings at match time
        winnerRank: intOrNull(r.winner_rank),
        winnerRankPoints: intOrNull(r.winner_rank_points),
        loserRank: intOrNull(r.loser_rank),
        loserRankPoints: intOrNull(r.loser_rank_points),
        // Ages
        winnerAge: floatOrNull(r.winner_age),
        loserAge: floatOrNull(r.loser_age),
      };

      return prisma.match.upsert({
        where: {
          tourneyId_matchNum: { tourneyId, matchNum },
        },
        update: data,
        create: data,
      });
    });

    await prisma.$transaction(ops);
    upserted += batch.length;
  }

  console.log(`  Upserted ${upserted} matches for ${tour.toUpperCase()} ${year}`);
}

// ── Rankings ───────────────────────────────────────────────────────────

interface RawRanking {
  ranking_date: string;
  rank: string;
  player: string;
  points: string;
}

async function syncRankings(tour: "atp" | "wta"): Promise<void> {
  const base = tour === "atp" ? ATP_BASE : WTA_BASE;
  const prefix = tour === "atp" ? "atp" : "wta";

  for (const decade of RANKING_DECADES) {
    const url = `${base}/${prefix}_rankings_${decade}.csv`;
    console.log(`Fetching rankings: ${url}`);

    const rows = await fetchCsv<RawRanking>(url);
    if (rows.length === 0) continue;
    console.log(`  Parsed ${rows.length} ranking rows (${decade})`);

    const BATCH = 1000;
    let created = 0;

    for (let i = 0; i < rows.length; i += BATCH) {
      const batch = rows.slice(i, i + BATCH);
      const data = batch
        .filter((r) => r.ranking_date && r.rank && r.player)
        .map((r) => ({
          date: r.ranking_date,
          rank: parseInt(r.rank, 10),
          playerId: resolvePlayerId(r.player, tour),
          points: intOrNull(r.points),
          tour,
        }));

      const result = await prisma.ranking.createMany({
        data,
        skipDuplicates: true,
      });
      created += result.count;
    }

    console.log(
      `  Created ${created} ranking rows for ${tour.toUpperCase()} ${decade}`
    );
  }
}

// ── Main ───────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("=== Sackmann CSV Sync ===\n");

  // 1. Players
  console.log("--- Syncing Players ---");
  await syncPlayers("atp");
  await syncPlayers("wta");

  // 2. Matches (and tournaments extracted from them)
  console.log("\n--- Syncing Matches ---");
  for (let year = START_YEAR; year <= CURRENT_YEAR; year++) {
    await syncMatchesForYear(year, "atp");
    await syncMatchesForYear(year, "wta");
  }

  // 3. Rankings
  console.log("\n--- Syncing Rankings ---");
  await syncRankings("atp");
  await syncRankings("wta");

  console.log("\n=== Sync complete ===");
}

main()
  .catch((err) => {
    console.error("Sync failed:", err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
