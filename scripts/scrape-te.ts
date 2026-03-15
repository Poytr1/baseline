/**
 * Near-real-time ATP data backfill from TennisExplorer.com
 *
 * Scrapes server-rendered HTML to get match results within hours of completion.
 * Supplements the existing tennis-data.co.uk Excel scraper (scrape-atp.ts)
 * which has ~1 week lag.
 *
 * Usage:
 *   npm run scrape-te              # scrape last 7 days
 *   npm run scrape-te -- --days 14 # scrape last 14 days
 */

import { PrismaClient } from "@prisma/client";
import * as cheerio from "cheerio";
import Fuse from "fuse.js";

const prisma = new PrismaClient();

// ── Constants ──────────────────────────────────────────────────────────

const BASE_URL = "https://www.tennisexplorer.com";
const RATE_LIMIT_MS = 1500;
const USER_AGENT = "baseline-data-sync/1.0";
const SCRAPED_ID_BASE = 900_000_000;

// Skip challenger, futures, and exhibition tournaments
const SKIP_PATTERNS = [
  /challenger/i,
  /futures/i,
  /utr pro/i,
  /itf/i,
  /davis cup/i,
  /laver cup/i,
  /united cup/i,
  /exhibition/i,
  /next gen/i,
];

// Map TennisExplorer round codes to Sackmann-style
const TE_ROUND_MAP: Record<string, string> = {
  F: "F",
  SF: "SF",
  QF: "QF",
  R16: "R16",
  R32: "R32",
  R64: "R64",
  R128: "R128",
  "1R": "R128",
  "2R": "R32",
  "3R": "R32",
  RR: "RR",
};

// Map TennisExplorer surface to our format
const SURFACE_MAP: Record<string, string> = {
  hard: "Hard",
  clay: "Clay",
  grass: "Grass",
  carpet: "Carpet",
  indoor: "Hard",
};

// ── Types ──────────────────────────────────────────────────────────────

interface SetScore {
  games: number;
  tiebreak?: number; // tiebreak points lost by this player (from <sup>)
}

interface ScrapedMatch {
  round: string;
  player1Name: string;
  player1Slug: string;
  player1Sets: number;
  player2Name: string;
  player2Slug: string;
  player2Sets: number;
  sets: [SetScore, SetScore][]; // [player1, player2] per set
  isWalkover: boolean;
  matchDetailId: string;
  dateStr: string; // DD.MM. or DD.MM.YYYY format
}

interface TournamentInfo {
  name: string;
  slug: string;
  surface: string;
  matches: ScrapedMatch[];
}

interface CachedPlayer {
  id: number;
  nameFirst: string;
  nameLast: string;
  slug: string;
  fullName: string;
}

// ── Helpers ────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchPage(url: string): Promise<string> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(url, {
        headers: { "User-Agent": USER_AGENT },
        signal: AbortSignal.timeout(30_000),
      });
      if (!res.ok) {
        console.warn(`  [HTTP ${res.status}] ${url}`);
        if (attempt < 2) {
          await sleep(2000 * (attempt + 1));
          continue;
        }
        return "";
      }
      return await res.text();
    } catch (err) {
      if (attempt < 2) {
        await sleep(2000 * (attempt + 1));
        continue;
      }
      console.warn(`  [FETCH ERROR] ${url}: ${err}`);
      return "";
    }
  }
  return "";
}

function generateScrapedPlayerId(name: string): number {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) | 0;
  }
  return SCRAPED_ID_BASE + Math.abs(hash % 100_000_000);
}

function slugify(first: string, last: string, id: number): string {
  return `${first}-${last}-${id}`.toLowerCase().replace(/[^a-z0-9-]/g, "");
}

function dateStrToYYYYMMDD(dateStr: string, year: number): string {
  // "08.03.2026" or "08.03." format
  const parts = dateStr.trim().match(/(\d{1,2})\.(\d{1,2})\.(\d{4})?/);
  if (!parts) return "";
  const day = parts[1].padStart(2, "0");
  const month = parts[2].padStart(2, "0");
  const y = parts[3] || String(year);
  return `${y}${month}${day}`;
}

// ── Player matching (reuses same logic as scrape-atp.ts) ────────────

let playerFuse: Fuse<CachedPlayer>;
const resolvedCache = new Map<string, number>();
let currentTour = "atp";

async function buildPlayerCache(tour: string = "atp"): Promise<void> {
  currentTour = tour;
  resolvedCache.clear();

  const allPlayers = await prisma.player.findMany({
    where: { tour },
    select: { id: true, nameFirst: true, nameLast: true, slug: true },
  });

  const items: CachedPlayer[] = allPlayers.map((p) => ({
    id: p.id,
    nameFirst: p.nameFirst,
    nameLast: p.nameLast,
    slug: p.slug,
    fullName: `${p.nameFirst} ${p.nameLast}`.trim().toLowerCase(),
  }));

  playerFuse = new Fuse(items, {
    keys: ["fullName", "nameLast"],
    threshold: 0.3,
    includeScore: true,
  });
  console.log(`  Player cache: ${items.length} ${tour.toUpperCase()} players`);
}

/**
 * Resolve a TennisExplorer player name to a DB player ID.
 *
 * TE uses "LastName" on tournament pages and "LastName FirstName" on
 * match detail pages. The slug (from href) gives us a unique identifier.
 */
async function resolvePlayerId(
  teName: string,
  teSlug: string,
): Promise<number> {
  const cacheKey = teSlug || teName;
  if (resolvedCache.has(cacheKey)) return resolvedCache.get(cacheKey)!;

  // TE names on match detail pages are "Sinner Jannik" (LastName FirstName)
  // On tournament pages just "Sinner"
  // Multi-word last names need special handling: "Auger Aliassime Felix"
  // Try progressively longer last name combinations
  const parts = teName.trim().split(/\s+/);
  let searchLast = parts[0];
  let searchFirst = parts.slice(1).join(" ") || "";

  // Try multi-word last names first (e.g. "Auger Aliassime Felix" -> Last="Auger Aliassime", First="Felix")
  if (parts.length >= 3) {
    for (let splitAt = parts.length - 1; splitAt >= 2; splitAt--) {
      const tryLast = parts.slice(0, splitAt).join(" ");
      const tryFirst = parts.slice(splitAt).join(" ");
      const multiMatch = await prisma.player.findMany({
        where: {
          tour: currentTour,
          nameLast: { equals: tryLast, mode: "insensitive" },
          nameFirst: { equals: tryFirst, mode: "insensitive" },
        },
        select: { id: true, nameFirst: true, nameLast: true },
      });
      if (multiMatch.length === 1) {
        resolvedCache.set(cacheKey, multiMatch[0].id);
        return multiMatch[0].id;
      }
    }
  }

  // 1. Try exact match by last name + first name
  if (searchFirst) {
    const exactMatches = await prisma.player.findMany({
      where: {
        tour: currentTour,
        nameLast: { equals: searchLast, mode: "insensitive" },
        nameFirst: { equals: searchFirst, mode: "insensitive" },
      },
      select: { id: true, nameFirst: true, nameLast: true },
    });

    if (exactMatches.length === 1) {
      resolvedCache.set(cacheKey, exactMatches[0].id);
      return exactMatches[0].id;
    }

    if (exactMatches.length > 1) {
      for (const m of exactMatches) {
        const hasMatches = await prisma.match.findFirst({
          where: { OR: [{ winnerId: m.id }, { loserId: m.id }] },
          select: { id: true },
        });
        if (hasMatches) {
          resolvedCache.set(cacheKey, m.id);
          return m.id;
        }
      }
      resolvedCache.set(cacheKey, exactMatches[0].id);
      return exactMatches[0].id;
    }
  }

  // 2. Try match by last name only (common on tournament pages)
  const lastNameMatches = await prisma.player.findMany({
    where: {
      tour: currentTour,
      nameLast: { equals: searchLast, mode: "insensitive" },
    },
    select: { id: true, nameFirst: true, nameLast: true },
  });

  if (lastNameMatches.length === 1) {
    resolvedCache.set(cacheKey, lastNameMatches[0].id);
    return lastNameMatches[0].id;
  }

  // Multiple matches by last name — prefer one with match history
  if (lastNameMatches.length > 1) {
    // If we have a first name initial from the slug, use it
    // TE slugs are like "sinner-8b8e8" or "alcaraz-5ab70"
    if (searchFirst) {
      const initial = searchFirst[0].toUpperCase();
      const filtered = lastNameMatches.filter(
        (m) => m.nameFirst && m.nameFirst[0].toUpperCase() === initial,
      );
      if (filtered.length === 1) {
        resolvedCache.set(cacheKey, filtered[0].id);
        return filtered[0].id;
      }
    }

    // Pick the one with most match data
    for (const m of lastNameMatches) {
      const matchCount = await prisma.match.count({
        where: { OR: [{ winnerId: m.id }, { loserId: m.id }] },
      });
      if (matchCount > 0) {
        resolvedCache.set(cacheKey, m.id);
        return m.id;
      }
    }
    resolvedCache.set(cacheKey, lastNameMatches[0].id);
    return lastNameMatches[0].id;
  }

  // 3. Try partial last name match (handles hyphenated names etc.)
  const norm = (s: string) => s.toLowerCase().replace(/[-']/g, " ");
  const partialMatches = await prisma.player.findMany({
    where: {
      tour: currentTour,
      nameLast: { startsWith: searchLast, mode: "insensitive" },
    },
    select: { id: true, nameFirst: true, nameLast: true },
    take: 10,
  });

  if (partialMatches.length === 1) {
    console.log(
      `  [PARTIAL] "${teName}" -> "${partialMatches[0].nameFirst} ${partialMatches[0].nameLast}"`,
    );
    resolvedCache.set(cacheKey, partialMatches[0].id);
    return partialMatches[0].id;
  }

  // 4. Fuzzy match
  const fuzzyQuery = searchFirst
    ? `${searchFirst} ${searchLast}`
    : searchLast;
  const fuzzyResults = playerFuse.search(fuzzyQuery);
  if (fuzzyResults.length > 0 && (fuzzyResults[0].score ?? 1) < 0.35) {
    const match = fuzzyResults[0].item;
    const matchLast = norm(match.nameLast);
    const queryLast = norm(searchLast);
    if (
      matchLast === queryLast ||
      matchLast.startsWith(queryLast) ||
      queryLast.startsWith(matchLast)
    ) {
      console.log(
        `  [FUZZY] "${teName}" -> "${match.nameFirst} ${match.nameLast}" (score: ${fuzzyResults[0].score?.toFixed(3)})`,
      );
      resolvedCache.set(cacheKey, match.id);
      return match.id;
    }
  }

  // 5. Create new player
  const nameFirst = searchFirst || "";
  const nameLast = searchLast || teName;
  const id = generateScrapedPlayerId(cacheKey);
  const slug = slugify(nameFirst, nameLast, id);

  console.log(`  [NEW] "${teName}" -> id=${id}`);

  await prisma.player.upsert({
    where: { id },
    update: { nameFirst, nameLast, tour: currentTour, slug },
    create: { id, nameFirst, nameLast, tour: currentTour, slug },
  });

  resolvedCache.set(cacheKey, id);
  return id;
}

// ── Scraping ──────────────────────────────────────────────────────────

/**
 * Discover tournament slugs from the daily results page.
 */
async function discoverTournaments(
  year: number,
  month: number,
  day: number,
): Promise<string[]> {
  const url = `${BASE_URL}/results/?type=atp-single&year=${year}&month=${String(month).padStart(2, "0")}&day=${String(day).padStart(2, "0")}`;
  const html = await fetchPage(url);
  if (!html) return [];

  const $ = cheerio.load(html);
  const slugs = new Set<string>();

  $('tr.head.flags a[href*="/atp-men/"]').each((_, el) => {
    const href = $(el).attr("href");
    if (!href) return;
    // Extract tournament slug: /indian-wells/2026/atp-men/ -> indian-wells
    const match = href.match(/^\/(.+?)\/\d{4}\/atp-men\//);
    if (match) {
      const slug = match[1];
      // Skip challengers, futures, exhibitions
      const name = $(el).text().trim();
      if (SKIP_PATTERNS.some((p) => p.test(name) || p.test(slug))) return;
      slugs.add(slug);
    }
  });

  return [...slugs];
}

/**
 * Scrape a tournament page for surface + completed match results.
 */
async function scrapeTournament(
  slug: string,
  year: number,
): Promise<TournamentInfo | null> {
  const url = `${BASE_URL}/${slug}/${year}/atp-men/`;
  const html = await fetchPage(url);
  if (!html) return null;

  const $ = cheerio.load(html);

  // Extract tournament name from h1
  const name = $("h1.bg").text().replace(/\s*\d{4}\s*\(.*?\)\s*$/, "").trim();
  if (!name) return null;

  // Extract surface from the info line: "(9,415,725 $, hard, men)"
  let surface = "Hard";
  const infoText = $("div.box.boxBasic.lGray").first().text();
  const surfaceMatch = infoText.match(
    /\b(hard|clay|grass|carpet|indoor)\b/i,
  );
  if (surfaceMatch) {
    surface = SURFACE_MAP[surfaceMatch[1].toLowerCase()] || "Hard";
  }

  // Find the completed matches table (second table.result on the page)
  // First table is "Next matches", second is completed results
  const tables = $("table.result");
  const matches: ScrapedMatch[] = [];

  tables.each((_, table) => {
    const $table = $(table);
    // Skip if this is the "Next matches" table (has no score columns with results)
    const hasResults = $table.find("td.result").length > 0;
    if (!hasResults) return;

    const rows = $table.find("tbody tr");
    let currentRound = "";

    for (let i = 0; i < rows.length; i++) {
      const $row = $(rows[i]);

      // First player row has time + round column
      const timeCell = $row.find("td.first.time");
      if (timeCell.length === 0) continue; // This is the second player row

      const roundCell = $row.find('td[class*="round"], td[title]');
      let round = "";
      if (roundCell.length > 0) {
        round = roundCell.text().trim();
        if (!round) {
          round = roundCell.attr("title") || "";
        }
        if (round) currentRound = round;
      }
      if (!round) round = currentRound;

      // Map round to Sackmann style
      const mappedRound =
        TE_ROUND_MAP[round] ||
        round.replace(/\. round/, "R").replace("round of ", "R");

      // Player 1 (this row)
      const p1Link = $row.find("td.t-name a");
      const p1Name = p1Link.text().replace(/\s*\(\d+\)/, "").trim();
      const p1Href = p1Link.attr("href") || "";
      const p1Slug = p1Href.replace(/^\/player\//, "").replace(/\/$/, "");
      const p1ResultText = $row.find("td.result").text().trim();
      const p1Result = parseInt(p1ResultText, 10);

      // Set scores for player 1
      const p1Scores: SetScore[] = [];
      $row.find("td.score").each((_, td) => {
        const $td = $(td);
        const sup = $td.find("sup").text().trim();
        // Remove <sup> to get just the game score
        const $clone = $td.clone();
        $clone.find("sup").remove();
        const gamesText = $clone.text().trim();
        if (gamesText && gamesText !== "\u00a0") {
          const score: SetScore = { games: parseInt(gamesText, 10) };
          if (sup) score.tiebreak = parseInt(sup, 10);
          p1Scores.push(score);
        }
      });

      // Player 2 (next row)
      const $nextRow = $(rows[i + 1]);
      if (!$nextRow.length) continue;

      const p2Link = $nextRow.find("td.t-name a");
      const p2Name = p2Link.text().replace(/\s*\(\d+\)/, "").trim();
      const p2Href = p2Link.attr("href") || "";
      const p2Slug = p2Href.replace(/^\/player\//, "").replace(/\/$/, "");
      const p2ResultText = $nextRow.find("td.result").text().trim();
      const p2Result = parseInt(p2ResultText, 10);

      const p2Scores: SetScore[] = [];
      $nextRow.find("td.score").each((_, td) => {
        const $td = $(td);
        const sup = $td.find("sup").text().trim();
        const $clone = $td.clone();
        $clone.find("sup").remove();
        const gamesText = $clone.text().trim();
        if (gamesText && gamesText !== "\u00a0") {
          const score: SetScore = { games: parseInt(gamesText, 10) };
          if (sup) score.tiebreak = parseInt(sup, 10);
          p2Scores.push(score);
        }
      });

      if (!p1Name || !p2Name) continue;

      // Detect walkovers: result text is "w/o" or similar
      const isWalkover =
        (isNaN(p1Result) && /w\/o/i.test(p1ResultText)) ||
        (isNaN(p2Result) && /w\/o/i.test(p2ResultText));

      // Skip matches where results are truly unparseable (not a walkover)
      if (!isWalkover && (isNaN(p1Result) || isNaN(p2Result))) continue;

      // Build sets array
      const sets: [SetScore, SetScore][] = [];
      for (
        let s = 0;
        s < Math.min(p1Scores.length, p2Scores.length);
        s++
      ) {
        sets.push([p1Scores[s], p2Scores[s]]);
      }

      // Extract match detail ID for dedup
      const infoLink =
        $row.find('a[href*="match-detail"]').attr("href") ||
        $nextRow.find('a[href*="match-detail"]').attr("href") ||
        "";
      const detailMatch = infoLink.match(/id=(\d+)/);
      const matchDetailId = detailMatch ? detailMatch[1] : "";

      // Extract date from the time cell
      const timeCellHtml = timeCell.html() || "";
      let dateStr = "";
      const dateMatch = timeCellHtml.match(/(\d{1,2}\.\d{1,2}\.)/);
      if (dateMatch) {
        dateStr = dateMatch[1];
      }

      matches.push({
        round: mappedRound || "R32",
        player1Name: p1Name,
        player1Slug: p1Slug,
        player1Sets: isWalkover ? (isNaN(p1Result) ? 0 : p1Result) : p1Result,
        player2Name: p2Name,
        player2Slug: p2Slug,
        player2Sets: isWalkover ? (isNaN(p2Result) ? 0 : p2Result) : p2Result,
        sets,
        isWalkover,
        matchDetailId,
        dateStr,
      });

      i++; // Skip the second player row
    }
  });

  return { name, slug, surface, matches };
}

/**
 * Detect if a match was a retirement based on the score.
 * A retirement occurs when the winner didn't win enough sets
 * (2 in bo3, 3 in bo5) — meaning the match ended before completion.
 */
function isRetirement(
  sets: [SetScore, SetScore][],
  winnerSetsWon: number,
  bestOf: number,
): boolean {
  if (sets.length === 0) return false;
  const setsNeeded = bestOf === 5 ? 3 : 2;
  if (winnerSetsWon >= setsNeeded) return false;

  // Winner didn't win enough sets — it's a retirement
  return true;
}

/**
 * Format a set score, e.g. 7-6(4) or 6-3
 * The tiebreak points are always on the losing side (the player with 6 games).
 */
function formatSet(winner: SetScore, loser: SetScore): string {
  if (loser.tiebreak !== undefined) {
    return `${winner.games}-${loser.games}(${loser.tiebreak})`;
  }
  if (winner.tiebreak !== undefined) {
    // Winner lost this set in a tiebreak
    return `${winner.games}(${winner.tiebreak})-${loser.games}`;
  }
  return `${winner.games}-${loser.games}`;
}

/**
 * Build score string from sets array.
 * Winner is always player1 if player1Sets > player2Sets.
 * Appends RET for retirements and returns W/O for walkovers.
 */
function buildScoreFromSets(
  sets: [SetScore, SetScore][],
  p1Won: boolean,
  isWalkover: boolean,
  winnerSetsWon: number,
  bestOf: number,
): string {
  if (isWalkover || sets.length === 0) return "W/O";
  const ordered = p1Won
    ? sets.map(([a, b]) => formatSet(a, b))
    : sets.map(([a, b]) => formatSet(b, a));
  const score = ordered.join(" ");
  if (isRetirement(sets, winnerSetsWon, bestOf)) {
    return `${score} RET`;
  }
  return score;
}

// ── Tournament ID resolution ────────────────────────────────────────

const tourneyIdCache = new Map<string, string>();

async function resolveTourneyId(
  teName: string,
  teSlug: string,
  year: number,
): Promise<string> {
  const key = `${teSlug}-${year}`;
  if (tourneyIdCache.has(key)) return tourneyIdCache.get(key)!;

  // Try to find existing tournament by name match
  const existing = await prisma.tournament.findFirst({
    where: {
      tour: "atp",
      name: { contains: teName, mode: "insensitive" },
      id: { startsWith: `atp-${year}` },
    },
    select: { id: true },
  });

  if (existing) {
    tourneyIdCache.set(key, existing.id);
    return existing.id;
  }

  // Try prior year with same name to get the numeric ID
  const priorYear = await prisma.tournament.findFirst({
    where: {
      tour: "atp",
      name: { contains: teName, mode: "insensitive" },
    },
    select: { id: true },
    orderBy: { date: "desc" },
  });

  let tourneyId: string;
  if (priorYear) {
    const parts = priorYear.id.split("-");
    const numId = parts[parts.length - 1];
    tourneyId = `atp-${year}-${numId}`;
  } else {
    // Generate a new ID from the slug
    const numericSlug = teSlug
      .split("")
      .reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 0);
    const paddedNum = String(Math.abs(numericSlug % 10000)).padStart(4, "0");
    tourneyId = `atp-${year}-${paddedNum}`;
  }

  tourneyIdCache.set(key, tourneyId);
  return tourneyId;
}

// ── Level detection ─────────────────────────────────────────────────

const MASTERS_NAMES = [
  "indian wells",
  "miami",
  "monte carlo",
  "madrid",
  "rome",
  "canadian",
  "rogers cup",
  "cincinnati",
  "western & southern",
  "shanghai",
  "paris",
  "bnp paribas",
  "internazionali",
  "mutua madrid",
];

const GRAND_SLAM_NAMES = [
  "australian open",
  "french open",
  "roland garros",
  "wimbledon",
  "us open",
];

function detectLevel(name: string): string | null {
  const lower = name.toLowerCase();
  if (GRAND_SLAM_NAMES.some((gs) => lower.includes(gs))) return "G";
  if (MASTERS_NAMES.some((m) => lower.includes(m))) return "M";
  if (lower.includes("masters cup") || lower.includes("atp finals"))
    return "F";
  return "A";
}

// ── Main sync logic ────────────────────────────────────────────────

async function syncDay(year: number, month: number, day: number): Promise<number> {
  const dateLabel = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;

  // Discover tournaments with results on this day
  const tournamentSlugs = await discoverTournaments(year, month, day);
  if (tournamentSlugs.length === 0) return 0;

  await sleep(RATE_LIMIT_MS);

  let totalImported = 0;

  for (const slug of tournamentSlugs) {
    // Scrape tournament page
    const tournament = await scrapeTournament(slug, year);
    if (!tournament || tournament.matches.length === 0) {
      await sleep(RATE_LIMIT_MS);
      continue;
    }

    // Resolve tournament ID
    const tourneyId = await resolveTourneyId(
      tournament.name,
      slug,
      year,
    );
    const level = detectLevel(tournament.name);

    // Compute YYYYMMDD date for the tournament
    const dateYMD = `${year}${String(month).padStart(2, "0")}${String(day).padStart(2, "0")}`;

    // Upsert tournament
    await prisma.tournament.upsert({
      where: { id: tourneyId },
      update: {
        name: tournament.name,
        surface: tournament.surface,
        level,
        date: dateYMD,
        tour: "atp",
      },
      create: {
        id: tourneyId,
        name: tournament.name,
        surface: tournament.surface,
        level,
        date: dateYMD,
        tour: "atp",
      },
    });

    // Get existing max match number
    const maxMatch = await prisma.match.aggregate({
      where: { tourneyId },
      _max: { matchNum: true },
    });
    let matchNum = (maxMatch._max.matchNum ?? 0) + 1;

    for (const m of tournament.matches) {
      // Determine winner/loser
      const p1Won = m.player1Sets > m.player2Sets;
      const winnerName = p1Won ? m.player1Name : m.player2Name;
      const winnerSlug = p1Won ? m.player1Slug : m.player2Slug;
      const loserName = p1Won ? m.player2Name : m.player1Name;
      const loserSlug = p1Won ? m.player2Slug : m.player1Slug;

      const winnerId = await resolvePlayerId(winnerName, winnerSlug);
      const loserId = await resolvePlayerId(loserName, loserSlug);
      if (winnerId === loserId) continue;

      const bestOf = level === "G" ? 5 : 3;
      const winnerSetsWon = p1Won ? m.player1Sets : m.player2Sets;
      const score = buildScoreFromSets(m.sets, p1Won, m.isWalkover, winnerSetsWon, bestOf);

      // Check for existing match (by tournament + winner + loser + round)
      const existing = await prisma.match.findFirst({
        where: { tourneyId, winnerId, loserId, round: m.round },
        select: { id: true },
      });

      if (existing) continue;

      // Also check by match detail ID to avoid duplicates
      if (m.matchDetailId) {
        const byDetail = await prisma.match.findFirst({
          where: {
            tourneyId,
            OR: [
              { winnerId, loserId },
              { winnerId: loserId, loserId: winnerId },
            ],
          },
          select: { id: true },
        });
        if (byDetail) continue;
      }

      await prisma.match.create({
        data: {
          tourneyId,
          matchNum,
          winnerId,
          loserId,
          score,
          bestOf,
          round: m.round,
          surface: tournament.surface,
          tour: "atp",
          winnerRank: null,
          winnerRankPoints: null,
          loserRank: null,
          loserRankPoints: null,
        },
      });

      matchNum++;
      totalImported++;
    }

    if (tournament.matches.length > 0) {
      console.log(
        `  ${dateLabel} | ${tournament.name}: ${tournament.matches.length} results, ${totalImported} new`,
      );
    }

    await sleep(RATE_LIMIT_MS);
  }

  return totalImported;
}

// ── Rankings Sync ─────────────────────────────────────────────────────

/**
 * Scrape current ATP/WTA rankings from TennisExplorer.
 * Fetches top 200 (4 pages of 50) for each tour.
 */
async function syncRankings(): Promise<void> {
  console.log("\n--- Syncing Rankings ---");

  const today = new Date();
  const dateStr = `${today.getFullYear()}${String(today.getMonth() + 1).padStart(2, "0")}${String(today.getDate()).padStart(2, "0")}`;

  for (const tour of [
    { key: "atp", url: "atp-men", label: "ATP" },
    { key: "wta", url: "wta-women", label: "WTA" },
  ]) {
    // Rebuild player cache for this tour
    await buildPlayerCache(tour.key);
    let totalUpserted = 0;

    for (let page = 1; page <= 4; page++) {
      const url =
        page === 1
          ? `${BASE_URL}/ranking/${tour.url}/`
          : `${BASE_URL}/ranking/${tour.url}/?page=${page}`;

      const html = await fetchPage(url);
      if (!html) {
        await sleep(RATE_LIMIT_MS);
        continue;
      }

      const $ = cheerio.load(html);
      const rows = $("table.result tbody tr");

      for (let i = 0; i < rows.length; i++) {
        const $row = $(rows[i]);
        const rankText = $row.find("td.rank").text().trim().replace(".", "");
        const rank = parseInt(rankText, 10);
        if (isNaN(rank)) continue;

        const nameLink = $row.find("td.t-name a");
        const name = nameLink.text().trim();
        const href = nameLink.attr("href") || "";
        const slug = href.replace(/^\/player\//, "").replace(/\/$/, "");
        if (!name) continue;

        const pointsText = $row.find("td.long-point").text().trim();
        const points = parseInt(pointsText, 10);
        if (isNaN(points)) continue;

        // Resolve player ID using existing matching logic
        const playerId = await resolvePlayerId(
          name,
          slug,
        );

        await prisma.ranking.upsert({
          where: {
            date_playerId_tour: {
              date: dateStr,
              playerId,
              tour: tour.key,
            },
          },
          update: { rank, points },
          create: {
            date: dateStr,
            rank,
            playerId,
            points,
            tour: tour.key,
          },
        });

        totalUpserted++;
      }

      await sleep(RATE_LIMIT_MS);
    }

    console.log(`  ${tour.label}: ${totalUpserted} rankings synced`);
  }
}

// ── Main ──────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("=== ATP Data Sync (TennisExplorer) ===\n");

  // Parse arguments
  const daysArg = process.argv.indexOf("--days");
  const days = daysArg !== -1 ? parseInt(process.argv[daysArg + 1], 10) : 7;

  console.log(`Syncing last ${days} days...\n`);

  // Build player cache
  console.log("Building player cache...");
  await buildPlayerCache();

  let totalImported = 0;

  // Iterate backwards from today
  for (let d = 0; d < days; d++) {
    const date = new Date();
    date.setDate(date.getDate() - d);
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();

    const imported = await syncDay(year, month, day);
    totalImported += imported;
  }

  console.log(`\n=== Match sync complete: ${totalImported} new matches ===`);

  // Sync rankings
  await syncRankings();

  console.log("\n=== All sync complete ===");
}

main()
  .catch((err) => {
    console.error("Sync failed:", err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
