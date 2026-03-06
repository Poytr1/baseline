import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getHeadToHead, getH2HSummary } from "@/lib/queries/h2h";

export async function GET(request: NextRequest) {
  const player1Slug = request.nextUrl.searchParams.get("player1");
  const player2Slug = request.nextUrl.searchParams.get("player2");

  if (!player1Slug || !player2Slug) {
    return NextResponse.json(
      { error: "Both player1 and player2 query params are required" },
      { status: 400 }
    );
  }

  if (player1Slug === player2Slug) {
    return NextResponse.json(
      { error: "Cannot compare a player with themselves" },
      { status: 400 }
    );
  }

  try {
    // Resolve slugs to players
    const [player1, player2] = await Promise.all([
      prisma.player.findUnique({
        where: { slug: player1Slug },
        select: {
          id: true,
          nameFirst: true,
          nameLast: true,
          ioc: true,
          slug: true,
          tour: true,
        },
      }),
      prisma.player.findUnique({
        where: { slug: player2Slug },
        select: {
          id: true,
          nameFirst: true,
          nameLast: true,
          ioc: true,
          slug: true,
          tour: true,
        },
      }),
    ]);

    if (!player1) {
      return NextResponse.json(
        { error: `Player not found: ${player1Slug}` },
        { status: 404 }
      );
    }

    if (!player2) {
      return NextResponse.json(
        { error: `Player not found: ${player2Slug}` },
        { status: 404 }
      );
    }

    // Check if players are from the same tour
    if (player1.tour !== player2.tour) {
      return NextResponse.json({
        player1,
        player2,
        crossTour: true,
        summary: null,
        matches: [],
      });
    }

    const matches = await getHeadToHead(player1.id, player2.id);
    const summary = getH2HSummary(matches, player1.id);

    // Serialize matches for the client
    const serializedMatches = matches.map((m) => ({
      id: m.id,
      date: m.tournament.date,
      tournamentName: m.tournament.name,
      tournamentLevel: m.tournament.level ?? "",
      surface: m.surface ?? m.tournament.surface ?? "Unknown",
      round: m.round ?? "",
      score: m.score ?? "",
      winnerId: m.winnerId,
      winnerName: `${m.winner.nameFirst} ${m.winner.nameLast}`,
      winnerSlug: m.winner.slug,
      loserId: m.loserId,
      loserName: `${m.loser.nameFirst} ${m.loser.nameLast}`,
      loserSlug: m.loser.slug,
    }));

    return NextResponse.json({
      player1,
      player2,
      crossTour: false,
      summary,
      matches: serializedMatches,
    });
  } catch (error) {
    console.error("H2H query failed:", error);
    return NextResponse.json(
      { error: "Failed to fetch head-to-head data" },
      { status: 500 }
    );
  }
}
