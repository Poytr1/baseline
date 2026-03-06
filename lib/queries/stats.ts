import { prisma } from "@/lib/db";
import { Prisma } from "@prisma/client";

export type Tour = "atp" | "wta";
export type StatsSurface = "all" | "hard" | "clay" | "grass";
export type SortColumn =
  | "aceRate"
  | "dfRate"
  | "firstServeInPct"
  | "firstServeWonPct"
  | "secondServeWonPct"
  | "bpSavedPct"
  | "matches";

export interface StatsLeaderboardOptions {
  tour: Tour;
  surface: StatsSurface;
  year: string; // "all" or a specific year like "2024"
  limit: number;
  sortBy: SortColumn;
  order: "asc" | "desc";
}

export interface PlayerStatRow {
  rank: number;
  playerId: number;
  playerName: string;
  slug: string;
  country: string | null;
  matches: number;
  aceRate: number; // aces per match
  dfRate: number; // double faults per match
  firstServeInPct: number; // 1st serve in %
  firstServeWonPct: number; // 1st serve points won %
  secondServeWonPct: number; // 2nd serve points won %
  bpSavedPct: number; // break points saved %
}

const VALID_SORT_COLUMNS: Record<SortColumn, string> = {
  aceRate: "ace_rate",
  dfRate: "df_rate",
  firstServeInPct: "first_serve_in_pct",
  firstServeWonPct: "first_serve_won_pct",
  secondServeWonPct: "second_serve_won_pct",
  bpSavedPct: "bp_saved_pct",
  matches: "match_count",
};

const MIN_MATCHES = 20;

/**
 * Aggregate player serve statistics from the matches table.
 *
 * Uses a UNION ALL approach: for each match, a player contributes stats as
 * either the winner or the loser. We union both sides, group by player, and
 * compute aggregate percentages.
 */
export async function getStatsLeaderboard(
  options: StatsLeaderboardOptions
): Promise<PlayerStatRow[]> {
  const { tour, surface, year, limit, sortBy, order } = options;

  // Validate sort column to prevent injection
  const sqlSort = VALID_SORT_COLUMNS[sortBy];
  if (!sqlSort) {
    throw new Error(`Invalid sort column: ${sortBy}`);
  }

  const sqlOrder = order === "asc" ? "ASC" : "DESC";

  // Build WHERE conditions for the matches/tournaments join
  // We always filter by tour. Surface and year are optional.
  const conditions: string[] = ["m.tour = $1"];
  const params: (string | number)[] = [tour];

  if (surface !== "all") {
    params.push(surface);
    conditions.push(`LOWER(COALESCE(m.surface, t.surface)) = LOWER($${params.length})`);
  }

  if (year !== "all") {
    params.push(`${year}%`);
    conditions.push(`t.date LIKE $${params.length}`);
  }

  const whereClause = conditions.join(" AND ");

  // Push limit param
  params.push(limit);
  const limitParamIdx = params.length;

  const sql = `
    WITH player_stats AS (
      SELECT
        player_id,
        COUNT(*) AS match_count,
        SUM(aces) AS total_aces,
        SUM(dfs) AS total_dfs,
        SUM(svpt) AS total_svpt,
        SUM(first_in) AS total_first_in,
        SUM(first_won) AS total_first_won,
        SUM(second_won) AS total_second_won,
        SUM(bp_saved) AS total_bp_saved,
        SUM(bp_faced) AS total_bp_faced
      FROM (
        -- Stats when player is winner
        SELECT
          m.winner_id AS player_id,
          m.w_ace AS aces,
          m.w_df AS dfs,
          m.w_svpt AS svpt,
          m.w_1st_in AS first_in,
          m.w_1st_won AS first_won,
          m.w_2nd_won AS second_won,
          m.w_bp_saved AS bp_saved,
          m.w_bp_faced AS bp_faced
        FROM matches m
        JOIN tournaments t ON m.tourney_id = t.id
        WHERE ${whereClause}
          AND m.w_svpt IS NOT NULL
          AND m.w_svpt > 0

        UNION ALL

        -- Stats when player is loser
        SELECT
          m.loser_id AS player_id,
          m.l_ace AS aces,
          m.l_df AS dfs,
          m.l_svpt AS svpt,
          m.l_1st_in AS first_in,
          m.l_1st_won AS first_won,
          m.l_2nd_won AS second_won,
          m.l_bp_saved AS bp_saved,
          m.l_bp_faced AS bp_faced
        FROM matches m
        JOIN tournaments t ON m.tourney_id = t.id
        WHERE ${whereClause}
          AND m.l_svpt IS NOT NULL
          AND m.l_svpt > 0
      ) AS combined
      GROUP BY player_id
      HAVING COUNT(*) >= ${MIN_MATCHES}
    ),
    computed AS (
      SELECT
        ps.player_id,
        ps.match_count,
        CASE WHEN ps.match_count > 0
          THEN ROUND(ps.total_aces::numeric / ps.match_count, 2)
          ELSE 0 END AS ace_rate,
        CASE WHEN ps.match_count > 0
          THEN ROUND(ps.total_dfs::numeric / ps.match_count, 2)
          ELSE 0 END AS df_rate,
        CASE WHEN ps.total_svpt > 0
          THEN ROUND(ps.total_first_in::numeric / ps.total_svpt * 100, 1)
          ELSE 0 END AS first_serve_in_pct,
        CASE WHEN ps.total_first_in > 0
          THEN ROUND(ps.total_first_won::numeric / ps.total_first_in * 100, 1)
          ELSE 0 END AS first_serve_won_pct,
        CASE WHEN (ps.total_svpt - ps.total_first_in) > 0
          THEN ROUND(ps.total_second_won::numeric / (ps.total_svpt - ps.total_first_in) * 100, 1)
          ELSE 0 END AS second_serve_won_pct,
        CASE WHEN ps.total_bp_faced > 0
          THEN ROUND(ps.total_bp_saved::numeric / ps.total_bp_faced * 100, 1)
          ELSE 0 END AS bp_saved_pct,
        p.id AS p_id,
        CONCAT(p.name_first, ' ', p.name_last) AS player_name,
        p.slug,
        p.ioc
      FROM player_stats ps
      JOIN players p ON ps.player_id = p.id
    )
    SELECT *
    FROM computed
    ORDER BY ${sqlSort} ${sqlOrder}
    LIMIT $${limitParamIdx}
  `;

  const rows: Array<{
    player_id: number;
    match_count: bigint;
    ace_rate: Prisma.Decimal | number;
    df_rate: Prisma.Decimal | number;
    first_serve_in_pct: Prisma.Decimal | number;
    first_serve_won_pct: Prisma.Decimal | number;
    second_serve_won_pct: Prisma.Decimal | number;
    bp_saved_pct: Prisma.Decimal | number;
    p_id: number;
    player_name: string;
    slug: string;
    ioc: string | null;
  }> = await prisma.$queryRawUnsafe(sql, ...params);

  return rows.map((row, index) => ({
    rank: index + 1,
    playerId: row.player_id,
    playerName: row.player_name,
    slug: row.slug,
    country: row.ioc,
    matches: Number(row.match_count),
    aceRate: Number(row.ace_rate),
    dfRate: Number(row.df_rate),
    firstServeInPct: Number(row.first_serve_in_pct),
    firstServeWonPct: Number(row.first_serve_won_pct),
    secondServeWonPct: Number(row.second_serve_won_pct),
    bpSavedPct: Number(row.bp_saved_pct),
  }));
}

export type StatsLeaderboardResult = Awaited<
  ReturnType<typeof getStatsLeaderboard>
>;
