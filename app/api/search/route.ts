import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET(request: NextRequest) {
  const query = request.nextUrl.searchParams.get("q");
  if (!query || query.length < 2 || query.length > 100) {
    return NextResponse.json([]);
  }

  try {
    const players = await prisma.player.findMany({
      where: {
        OR: [
          { nameFirst: { contains: query, mode: "insensitive" } },
          { nameLast: { contains: query, mode: "insensitive" } },
        ],
      },
      select: {
        id: true,
        nameFirst: true,
        nameLast: true,
        ioc: true,
        slug: true,
        tour: true,
      },
      take: 10,
      orderBy: { nameLast: "asc" },
    });

    return NextResponse.json(players);
  } catch (error) {
    console.error("Search query failed:", error);
    return NextResponse.json({ error: "Search failed" }, { status: 500 });
  }
}
