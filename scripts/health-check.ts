/**
 * Post-sync health check — verifies data pipeline produced reasonable results.
 *
 * Exits with code 1 if any check fails, making the GitHub Action fail visibly.
 *
 * Usage:
 *   npm run health-check
 */

import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

interface CheckResult {
  name: string;
  passed: boolean;
  detail: string;
}

const results: CheckResult[] = [];

function check(name: string, passed: boolean, detail: string): void {
  results.push({ name, passed, detail });
  const icon = passed ? "PASS" : "FAIL";
  console.log(`  [${icon}] ${name}: ${detail}`);
}

async function main(): Promise<void> {
  console.log("=== Data Pipeline Health Check ===\n");

  const now = new Date();
  const todayYMD = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}`;

  // How many days back counts as "recent"
  const recentDate = new Date();
  recentDate.setDate(recentDate.getDate() - 7);
  const recentYMD = `${recentDate.getFullYear()}${String(recentDate.getMonth() + 1).padStart(2, "0")}${String(recentDate.getDate()).padStart(2, "0")}`;

  // ── Check 1: Total match count is sane ──────────────────────────────

  const totalMatches = await prisma.match.count();
  check(
    "Total matches",
    totalMatches > 300_000,
    `${totalMatches.toLocaleString()} matches (expect >300k)`,
  );

  // ── Check 2: Recent ATP matches exist (last 7 days) ─────────────────

  const recentAtpMatches = await prisma.match.count({
    where: {
      tour: "atp",
      tournament: { date: { gte: recentYMD } },
    },
  });
  check(
    "Recent ATP matches (7d)",
    recentAtpMatches > 0,
    `${recentAtpMatches} matches`,
  );

  // ── Check 3: ATP rankings are current (within last 7 days) ──────────

  const latestAtpRanking = await prisma.ranking.findFirst({
    where: { tour: "atp" },
    orderBy: { date: "desc" },
    select: { date: true },
  });
  const rankingAge = latestAtpRanking
    ? Math.floor(
        (now.getTime() -
          new Date(
            parseInt(latestAtpRanking.date.slice(0, 4)),
            parseInt(latestAtpRanking.date.slice(4, 6)) - 1,
            parseInt(latestAtpRanking.date.slice(6, 8)),
          ).getTime()) /
          (1000 * 60 * 60 * 24),
      )
    : 999;
  check(
    "ATP rankings freshness",
    rankingAge <= 7,
    latestAtpRanking
      ? `last updated ${latestAtpRanking.date} (${rankingAge}d ago)`
      : "no rankings found",
  );

  // ── Check 4: WTA rankings are current ───────────────────────────────

  const latestWtaRanking = await prisma.ranking.findFirst({
    where: { tour: "wta" },
    orderBy: { date: "desc" },
    select: { date: true },
  });
  const wtaRankingAge = latestWtaRanking
    ? Math.floor(
        (now.getTime() -
          new Date(
            parseInt(latestWtaRanking.date.slice(0, 4)),
            parseInt(latestWtaRanking.date.slice(4, 6)) - 1,
            parseInt(latestWtaRanking.date.slice(6, 8)),
          ).getTime()) /
          (1000 * 60 * 60 * 24),
      )
    : 999;
  check(
    "WTA rankings freshness",
    wtaRankingAge <= 7,
    latestWtaRanking
      ? `last updated ${latestWtaRanking.date} (${wtaRankingAge}d ago)`
      : "no rankings found",
  );

  // ── Check 5: Elo ratings exist and are current ──────────────────────

  const latestElo = await prisma.eloRating.findFirst({
    orderBy: { date: "desc" },
    select: { date: true },
  });
  const eloAge = latestElo
    ? Math.floor(
        (now.getTime() -
          new Date(
            parseInt(latestElo.date.slice(0, 4)),
            parseInt(latestElo.date.slice(4, 6)) - 1,
            parseInt(latestElo.date.slice(6, 8)),
          ).getTime()) /
          (1000 * 60 * 60 * 24),
      )
    : 999;
  check(
    "Elo ratings freshness",
    eloAge <= 14,
    latestElo
      ? `last updated ${latestElo.date} (${eloAge}d ago)`
      : "no Elo ratings found",
  );

  // ── Check 6: Top players have sane Elo values ──────────────────────

  const topElo = await prisma.eloRating.findFirst({
    orderBy: { overall: "desc" },
    include: { player: { select: { nameFirst: true, nameLast: true } } },
  });
  const topEloSane =
    topElo !== null && topElo.overall > 2000 && topElo.overall < 3000;
  check(
    "Top Elo sanity",
    topEloSane,
    topElo
      ? `#1 ${topElo.player.nameFirst} ${topElo.player.nameLast} = ${Math.round(topElo.overall)}`
      : "no Elo data",
  );

  // ── Check 7: Player count is sane ───────────────────────────────────

  const playerCount = await prisma.player.count();
  check(
    "Player count",
    playerCount > 10_000,
    `${playerCount.toLocaleString()} players (expect >10k)`,
  );

  // ── Summary ─────────────────────────────────────────────────────────

  const failed = results.filter((r) => !r.passed);
  console.log(
    `\n=== ${results.length - failed.length}/${results.length} checks passed ===`,
  );

  if (failed.length > 0) {
    console.error(
      `\nFAILED CHECKS:\n${failed.map((f) => `  - ${f.name}: ${f.detail}`).join("\n")}`,
    );
    process.exit(1);
  }
}

main()
  .catch((err) => {
    console.error("Health check failed:", err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
