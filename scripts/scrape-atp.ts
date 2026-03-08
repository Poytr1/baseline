import { PrismaClient } from "@prisma/client";
import Fuse from "fuse.js";
import * as XLSX from "xlsx";

const prisma = new PrismaClient();

// ── Constants ──────────────────────────────────────────────────────────

const DATA_URL = "http://www.tennis-data.co.uk";
const CURRENT_YEAR = new Date().getFullYear();

// Scraped ATP player IDs live in the 900M range —
// above Sackmann (~210k max) and below WTA offset (1B).
const SCRAPED_ID_BASE = 900_000_000;

// ── Types ──────────────────────────────────────────────────────────────

interface RawRow {
  ATP: number;
  Location: string;
  Tournament: string;
  Date: number; // Excel serial
  Series: string;
  Court: string;
  Surface: string;
  Round: string;
  "Best of": number;
  Winner: string;
  Loser: string;
  WRank: number | string | undefined;
  LRank: number | string | undefined;
  WPts: number | string | undefined;
  LPts: number | string | undefined;
  W1: number | undefined;
  L1: number | undefined;
  W2: number | undefined;
  L2: number | undefined;
  W3: number | undefined;
  L3: number | undefined;
  W4: number | undefined;
  L4: number | undefined;
  W5: number | undefined;
  L5: number | undefined;
  Wsets: number | undefined;
  Lsets: number | undefined;
  Comment: string | undefined;
}

interface CachedPlayer {
  id: number;
  nameFirst: string;
  nameLast: string;
  fullName: string; // normalized lowercase
}

// ── Helpers ────────────────────────────────────────────────────────────

function excelDateToYYYYMMDD(serial: number): string {
  const d = new Date((serial - 25569) * 86400000);
  return d.toISOString().slice(0, 10).replace(/-/g, "");
}

/** Safely parse a value to integer, returning null for non-numeric values like "N/A" */
function safeInt(v: number | string | undefined): number | null {
  if (v === undefined || v === null) return null;
  if (typeof v === "number") return Number.isFinite(v) ? Math.round(v) : null;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

function slugify(first: string, last: string, id: number): string {
  return `${first}-${last}-${id}`.toLowerCase().replace(/[^a-z0-9-]/g, "");
}

function generateScrapedPlayerId(name: string): number {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) | 0;
  }
  return SCRAPED_ID_BASE + Math.abs(hash % 100_000_000);
}

const LEVEL_MAP: Record<string, string> = {
  "Grand Slam": "G",
  "Masters 1000": "M",
  "Masters Cup": "F",
  "ATP500": "A",
  "ATP250": "A",
};

function buildScore(row: RawRow): string {
  const sets: string[] = [];
  const pairs: [number | undefined, number | undefined][] = [
    [row.W1, row.L1],
    [row.W2, row.L2],
    [row.W3, row.L3],
    [row.W4, row.L4],
    [row.W5, row.L5],
  ];
  for (const [w, l] of pairs) {
    if (w === undefined || l === undefined) break;
    sets.push(`${w}-${l}`);
  }
  if (row.Comment === "Retired") {
    sets.push("RET");
  } else if (row.Comment === "Walkover") {
    return "W/O";
  }
  return sets.join(" ");
}

/**
 * Map round names to Sackmann-style codes.
 *
 * tennis-data.co.uk uses ordinal round names ("1st Round", "2nd Round", etc.)
 * that are relative to each tournament's draw size. We need to map these to
 * actual round codes (R128, R64, R32, R16, QF, SF, F).
 *
 * Draw sizes by series:
 *   Grand Slam: 128 draw → 1st=R128, 2nd=R64, 3rd=R32, 4th=R16, QF, SF, F
 *   Masters 1000: ~56-96 draw → 1st=R64, 2nd=R32, 3rd=R16, QF, SF, F
 *   ATP500: 32 draw → 1st=R32, 2nd=R16, QF, SF, F
 *   ATP250: 28-32 draw → 1st=R32, 2nd=R16, QF, SF, F
 */
function mapRound(round: string, series: string): string {
  // QF/SF/F/RR are the same regardless of draw size
  if (round === "Quarterfinals") return "QF";
  if (round === "Semifinals") return "SF";
  if (round === "The Final") return "F";
  if (round === "Round Robin") return "RR";

  if (series === "Grand Slam") {
    if (round === "1st Round") return "R128";
    if (round === "2nd Round") return "R64";
    if (round === "3rd Round") return "R32";
    if (round === "4th Round") return "R16";
  } else if (series === "Masters 1000") {
    if (round === "1st Round") return "R64";
    if (round === "2nd Round") return "R32";
    if (round === "3rd Round") return "R16";
  } else {
    // ATP500, ATP250, and others — 32-draw
    if (round === "1st Round") return "R32";
    if (round === "2nd Round") return "R16";
  }

  return round;
}

// ── Player matching ────────────────────────────────────────────────────

let playerFuse: Fuse<CachedPlayer>;
let playerCache: Map<string, CachedPlayer>;
const resolvedCache = new Map<string, number>(); // name -> id for this session

async function buildPlayerCache(): Promise<void> {
  const allPlayers = await prisma.player.findMany({
    where: { tour: "atp" },
    select: { id: true, nameFirst: true, nameLast: true },
  });

  const items: CachedPlayer[] = allPlayers.map((p) => ({
    id: p.id,
    nameFirst: p.nameFirst,
    nameLast: p.nameLast,
    fullName: `${p.nameFirst} ${p.nameLast}`.trim().toLowerCase(),
  }));

  playerCache = new Map(items.map((p) => [p.fullName, p]));
  playerFuse = new Fuse(items, {
    keys: ["fullName"],
    threshold: 0.3,
    includeScore: true,
  });
  console.log(`  Player cache: ${items.length} ATP players`);
}

/**
 * Match a tennis-data.co.uk name (e.g. "Sinner J.") to a DB player ID.
 *
 * tennis-data uses "LastName Initial." format. We expand this and do:
 * 1. Session cache lookup
 * 2. DB exact match on last name + first initial
 * 3. Fuse.js fuzzy match
 * 4. Create new player if no match
 */
async function resolvePlayerId(tdName: string): Promise<number> {
  const normalized = tdName.trim();
  if (resolvedCache.has(normalized)) return resolvedCache.get(normalized)!;

  // Parse "LastName F." or "LastName F.M." or "LastName Zh." or "LastName Dar." format
  const parts = normalized.match(/^(.+?)\s+([A-Z][a-z]{0,2})\.?\s*([A-Z]\.?)?$/);
  let searchLast = "";
  let searchInitial = "";
  if (parts) {
    searchLast = parts[1].trim();
    searchInitial = parts[2]; // First initial/abbreviation (e.g. "J", "Zh", "Dar")
  } else {
    // Fallback: just use the full name
    searchLast = normalized;
  }

  // 1. Try exact match: last name + first initial
  if (searchInitial) {
    // Build name variants: handle apostrophes and hyphens
    // "O Connell" -> also try "O'Connell"
    // "Auger-Aliassime" -> also try "Auger Aliassime"
    const lastNameVariants = [searchLast];
    if (searchLast.match(/^[A-Z] [A-Z]/)) {
      lastNameVariants.push(searchLast.replace(/^([A-Z]) /, "$1'"));
    }
    if (searchLast.includes("-")) {
      lastNameVariants.push(searchLast.replace(/-/g, " "));
    }
    if (searchLast.includes(" ") && !searchLast.match(/^[A-Z] [A-Z]/)) {
      lastNameVariants.push(searchLast.replace(/ /g, "-"));
    }

    for (const lastVariant of lastNameVariants) {
      const exactMatches = await prisma.player.findMany({
        where: {
          tour: "atp",
          nameLast: { equals: lastVariant, mode: "insensitive" },
          nameFirst: { startsWith: searchInitial, mode: "insensitive" },
        },
        select: { id: true, nameFirst: true, nameLast: true },
      });

      if (exactMatches.length === 1) {
        resolvedCache.set(normalized, exactMatches[0].id);
        return exactMatches[0].id;
      }

      if (exactMatches.length > 1) {
        // Prefer the one with most recent match data
        for (const m of exactMatches) {
          const hasMatches = await prisma.match.findFirst({
            where: { OR: [{ winnerId: m.id }, { loserId: m.id }] },
            select: { id: true },
          });
          if (hasMatches) {
            resolvedCache.set(normalized, m.id);
            return m.id;
          }
        }
        resolvedCache.set(normalized, exactMatches[0].id);
        return exactMatches[0].id;
      }
    }

    // Try partial last name match — handles truncated names
    // e.g. "Mpetshi G." where full name is "Mpetshi Perricard"
    const partialMatches = await prisma.player.findMany({
      where: {
        tour: "atp",
        nameLast: { startsWith: searchLast, mode: "insensitive" },
        nameFirst: { startsWith: searchInitial, mode: "insensitive" },
      },
      select: { id: true, nameFirst: true, nameLast: true },
    });

    if (partialMatches.length === 1) {
      console.log(
        `  [PARTIAL] "${normalized}" -> "${partialMatches[0].nameFirst} ${partialMatches[0].nameLast}"`
      );
      resolvedCache.set(normalized, partialMatches[0].id);
      return partialMatches[0].id;
    }

    if (partialMatches.length > 1) {
      // Prefer the one with match data
      for (const m of partialMatches) {
        const hasMatches = await prisma.match.findFirst({
          where: { OR: [{ winnerId: m.id }, { loserId: m.id }] },
          select: { id: true },
        });
        if (hasMatches) {
          console.log(
            `  [PARTIAL] "${normalized}" -> "${m.nameFirst} ${m.nameLast}"`
          );
          resolvedCache.set(normalized, m.id);
          return m.id;
        }
      }
    }
  }

  // 2. Fuzzy match
  const fuzzyQuery = searchInitial ? `${searchInitial} ${searchLast}` : searchLast;
  const fuzzyResults = playerFuse.search(fuzzyQuery);
  if (fuzzyResults.length > 0 && (fuzzyResults[0].score ?? 1) < 0.35) {
    const match = fuzzyResults[0].item;
    // Validate: fuzzy match must share the same last name (case-insensitive)
    // to avoid false positives like "Bu Y." -> "Guy Burdick"
    // Normalize hyphens/spaces for comparison (Auger-Aliassime vs Auger Aliassime)
    const norm = (s: string) => s.toLowerCase().replace(/[-']/g, " ");
    const matchLast = norm(match.nameLast);
    const queryLast = norm(searchLast);
    if (matchLast === queryLast || matchLast.startsWith(queryLast) || queryLast.startsWith(matchLast)) {
      console.log(
        `  [FUZZY] "${normalized}" -> "${match.nameFirst} ${match.nameLast}" (score: ${fuzzyResults[0].score?.toFixed(3)})`
      );
      resolvedCache.set(normalized, match.id);
      return match.id;
    }
  }

  // 3. Create new player
  const nameFirst = searchInitial || "";
  const nameLast = searchLast || normalized;
  const id = generateScrapedPlayerId(normalized);
  const slug = slugify(nameFirst, nameLast, id);

  console.log(`  [NEW] "${normalized}" -> id=${id}`);

  await prisma.player.upsert({
    where: { id },
    update: { nameFirst, nameLast, tour: "atp", slug },
    create: { id, nameFirst, nameLast, tour: "atp", slug },
  });

  resolvedCache.set(normalized, id);
  return id;
}

// ── Tournament ID resolution ───────────────────────────────────────────

const tourneyIdCache = new Map<string, string>();

async function resolveTourneyId(
  atpNum: number,
  year: number
): Promise<string> {
  const key = `${atpNum}-${year}`;
  if (tourneyIdCache.has(key)) return tourneyIdCache.get(key)!;

  const paddedNum = String(atpNum).padStart(4, "0");

  // Try to find existing tournament with same numeric ID from any year
  const existing = await prisma.tournament.findFirst({
    where: {
      tour: "atp",
      OR: [
        { id: { endsWith: `-${atpNum}` } },
        { id: { endsWith: `-${paddedNum}` } },
      ],
    },
    select: { id: true },
    orderBy: { date: "desc" },
  });

  let tourneyId: string;
  if (existing) {
    // Extract the numeric portion format from existing ID
    const existingParts = existing.id.split("-");
    const existingNum = existingParts[existingParts.length - 1];
    tourneyId = `atp-${year}-${existingNum}`;
  } else {
    tourneyId = `atp-${year}-${paddedNum}`;
  }

  tourneyIdCache.set(key, tourneyId);
  return tourneyId;
}

// ── Surface resolution from DB ─────────────────────────────────────────

async function resolveSurfaceFromDb(
  atpNum: number,
  fallback: string
): Promise<string> {
  const paddedNum = String(atpNum).padStart(4, "0");
  const existing = await prisma.tournament.findFirst({
    where: {
      tour: "atp",
      OR: [
        { id: { endsWith: `-${atpNum}` } },
        { id: { endsWith: `-${paddedNum}` } },
      ],
      surface: { not: null },
    },
    select: { surface: true },
    orderBy: { date: "desc" },
  });
  return existing?.surface ?? fallback;
}

// ── Fetch and parse Excel ──────────────────────────────────────────────

async function fetchExcel(year: number): Promise<RawRow[]> {
  const url = `${DATA_URL}/${year}/${year}.xlsx`;
  console.log(`Fetching: ${url}`);

  const res = await fetch(url, { signal: AbortSignal.timeout(30_000) });
  if (!res.ok) {
    console.warn(`  [SKIP] ${url} -> ${res.status}`);
    return [];
  }

  const buffer = await res.arrayBuffer();
  const wb = XLSX.read(new Uint8Array(buffer), { type: "array" });
  const ws = wb.Sheets[wb.SheetNames[0]];
  const data = XLSX.utils.sheet_to_json<RawRow>(ws);
  console.log(`  Parsed ${data.length} matches`);
  return data;
}

// ── Main sync logic ────────────────────────────────────────────────────

async function syncYear(year: number): Promise<void> {
  const rows = await fetchExcel(year);
  if (rows.length === 0) return;

  // Filter to only completed matches after our last data date
  const lastAtpDate = await prisma.$queryRaw<{ max_date: string }[]>`
    SELECT MAX(t.date) as max_date
    FROM matches m JOIN tournaments t ON m.tourney_id = t.id
    WHERE m.tour = 'atp'
  `;
  const cutoffDate = lastAtpDate[0]?.max_date ?? "00000000";
  console.log(`  Last ATP match date in DB: ${cutoffDate}`);

  const newRows = rows.filter((r) => {
    if (!r.Date || !r.Winner || !r.Loser) return false;
    const dateStr = excelDateToYYYYMMDD(r.Date);
    return dateStr > cutoffDate;
  });

  console.log(`  New matches to import: ${newRows.length}`);
  if (newRows.length === 0) return;

  // Group by tournament ATP number for batch processing
  const byTourney = new Map<number, RawRow[]>();
  for (const row of newRows) {
    const num = row.ATP;
    if (!byTourney.has(num)) byTourney.set(num, []);
    byTourney.get(num)!.push(row);
  }

  let totalUpserted = 0;

  for (const [atpNum, matches] of byTourney) {
    const first = matches[0];
    const tourneyId = await resolveTourneyId(atpNum, year);
    const surface = first.Surface || (await resolveSurfaceFromDb(atpNum, "Hard"));
    const dateStr = excelDateToYYYYMMDD(first.Date);

    // Upsert tournament
    await prisma.tournament.upsert({
      where: { id: tourneyId },
      update: {
        name: first.Tournament,
        surface,
        level: LEVEL_MAP[first.Series] ?? null,
        date: dateStr,
        tour: "atp",
      },
      create: {
        id: tourneyId,
        name: first.Tournament,
        surface,
        level: LEVEL_MAP[first.Series] ?? null,
        date: dateStr,
        tour: "atp",
      },
    });

    // Find the highest existing matchNum for this tournament
    const maxMatch = await prisma.match.aggregate({
      where: { tourneyId },
      _max: { matchNum: true },
    });
    let matchNum = (maxMatch._max.matchNum ?? 0) + 1;

    // Upsert matches
    for (const row of matches) {
      const winnerId = await resolvePlayerId(row.Winner);
      const loserId = await resolvePlayerId(row.Loser);
      if (winnerId === loserId) continue; // skip if same player (shouldn't happen)

      const score = buildScore(row);
      const matchDate = excelDateToYYYYMMDD(row.Date);
      const round = mapRound(row.Round, first.Series);

      // Check if this match already exists (by tournament + players + round)
      const existing = await prisma.match.findFirst({
        where: {
          tourneyId,
          winnerId,
          loserId,
          round,
        },
        select: { id: true },
      });

      if (existing) continue; // already imported

      await prisma.match.create({
        data: {
          tourneyId,
          matchNum,
          winnerId,
          loserId,
          score,
          bestOf: row["Best of"] ?? 3,
          round,
          surface,
          tour: "atp",
          winnerRank: safeInt(row.WRank),
          winnerRankPoints: safeInt(row.WPts),
          loserRank: safeInt(row.LRank),
          loserRankPoints: safeInt(row.LPts),
        },
      });

      matchNum++;
      totalUpserted++;
    }

    console.log(
      `  ${first.Tournament}: ${matches.length} matches (${totalUpserted} total new)`
    );
  }

  console.log(`  Imported ${totalUpserted} new matches for ${year}`);
}

// ── Main ───────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("=== ATP Data Backfill (tennis-data.co.uk) ===\n");

  // Build player cache
  console.log("Building player cache...");
  await buildPlayerCache();

  // Determine years to fetch
  const startYear = parseInt(process.argv[2] || "2025", 10);
  const endYear = parseInt(process.argv[3] || String(CURRENT_YEAR), 10);

  for (let year = startYear; year <= endYear; year++) {
    console.log(`\n--- ${year} ---`);
    await syncYear(year);
  }

  console.log("\n=== Backfill complete ===");
}

main()
  .catch((err) => {
    console.error("Backfill failed:", err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
