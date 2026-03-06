import { NextRequest, NextResponse } from "next/server";
import {
  getStatsLeaderboard,
  type Tour,
  type StatsSurface,
  type SortColumn,
} from "@/lib/queries/stats";

const VALID_TOURS = new Set(["atp", "wta"]);
const VALID_SURFACES = new Set(["all", "hard", "clay", "grass"]);
const VALID_SORT_COLUMNS = new Set<SortColumn>([
  "aceRate",
  "dfRate",
  "firstServeInPct",
  "firstServeWonPct",
  "secondServeWonPct",
  "bpSavedPct",
  "matches",
]);
const VALID_ORDERS = new Set(["asc", "desc"]);
const VALID_LIMITS = new Set([50, 100]);

function isValidYear(value: string): boolean {
  if (value === "all") return true;
  const num = parseInt(value, 10);
  return !isNaN(num) && num >= 1968 && num <= 2030 && String(num) === value;
}

export async function GET(request: NextRequest) {
  try {
    const sp = request.nextUrl.searchParams;

    const tour = (sp.get("tour") ?? "atp") as Tour;
    const surface = (sp.get("surface") ?? "all") as StatsSurface;
    const year = sp.get("year") ?? "all";
    const limitStr = sp.get("limit") ?? "50";
    const sort = (sp.get("sort") ?? "firstServeWonPct") as SortColumn;
    const order = (sp.get("order") ?? "desc") as "asc" | "desc";

    // Validate
    if (!VALID_TOURS.has(tour)) {
      return NextResponse.json(
        { error: "Invalid tour. Must be 'atp' or 'wta'." },
        { status: 400 }
      );
    }

    if (!VALID_SURFACES.has(surface)) {
      return NextResponse.json(
        { error: "Invalid surface. Must be 'all', 'hard', 'clay', or 'grass'." },
        { status: 400 }
      );
    }

    if (!isValidYear(year)) {
      return NextResponse.json(
        { error: "Invalid year. Must be 'all' or a year between 1968 and 2030." },
        { status: 400 }
      );
    }

    const limit = parseInt(limitStr, 10);
    if (!VALID_LIMITS.has(limit)) {
      return NextResponse.json(
        { error: "Invalid limit. Must be 50 or 100." },
        { status: 400 }
      );
    }

    if (!VALID_SORT_COLUMNS.has(sort)) {
      return NextResponse.json(
        { error: `Invalid sort column: ${sort}` },
        { status: 400 }
      );
    }

    if (!VALID_ORDERS.has(order)) {
      return NextResponse.json(
        { error: "Invalid order. Must be 'asc' or 'desc'." },
        { status: 400 }
      );
    }

    const data = await getStatsLeaderboard({
      tour,
      surface,
      year,
      limit,
      sortBy: sort,
      order,
    });

    return NextResponse.json({ data });
  } catch (error) {
    console.error("Stats leaderboard query failed:", error);
    return NextResponse.json(
      { error: "Failed to fetch stats leaderboard" },
      { status: 500 }
    );
  }
}
