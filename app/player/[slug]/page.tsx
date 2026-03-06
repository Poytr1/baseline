import { notFound } from "next/navigation";
import type { Metadata } from "next";
import {
  getPlayerBySlug,
  getPlayerCareerStats,
  getPlayerMatchHistory,
  getPlayerEloHistory,
} from "@/lib/queries/player";
import { PlayerHero } from "@/components/player-hero";
import { PlayerStats } from "@/components/player-stats";
import { EloChart } from "@/components/elo-chart";
import { MatchHistory } from "@/components/match-history";

export const revalidate = 3600;

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const player = await getPlayerBySlug(slug);

  if (!player) {
    return { title: "Player Not Found | tennisconcrete" };
  }

  return {
    title: `${player.nameFirst} ${player.nameLast} | tennisconcrete`,
  };
}

export default async function PlayerPage({ params }: PageProps) {
  const { slug } = await params;
  const player = await getPlayerBySlug(slug);

  if (!player) {
    notFound();
  }

  const [careerStats, matchHistory, eloHistory] = await Promise.all([
    getPlayerCareerStats(player.id),
    getPlayerMatchHistory(player.id),
    getPlayerEloHistory(player.id),
  ]);

  const latestRanking = player.rankings[0] ?? null;
  const latestElo = player.eloRatings[0] ?? null;

  return (
    <div className="space-y-8">
      <PlayerHero
        player={player}
        latestRanking={latestRanking}
        latestElo={latestElo}
        careerRecord={careerStats.overall}
      />
      <PlayerStats stats={careerStats} />
      <EloChart history={eloHistory} />
      <MatchHistory matches={matchHistory} />
    </div>
  );
}
