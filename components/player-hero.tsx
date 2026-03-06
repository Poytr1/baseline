import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface PlayerHeroProps {
  player: {
    nameFirst: string;
    nameLast: string;
    ioc: string | null;
    dob: string | null;
    hand: string | null;
    height: number | null;
  };
  latestRanking: { rank: number; points: number | null } | null;
  latestElo: { overall: number } | null;
  careerRecord: { wins: number; losses: number };
}

function computeAge(dob: string | null): number | null {
  if (!dob || dob.length !== 8) return null;
  const year = parseInt(dob.substring(0, 4), 10);
  const month = parseInt(dob.substring(4, 6), 10);
  const day = parseInt(dob.substring(6, 8), 10);
  if (isNaN(year) || isNaN(month) || isNaN(day)) return null;
  const birthDate = new Date(year, month - 1, day);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  if (
    monthDiff < 0 ||
    (monthDiff === 0 && today.getDate() < birthDate.getDate())
  ) {
    age--;
  }
  return age;
}

function formatHand(hand: string | null): string {
  if (!hand) return "Unknown";
  switch (hand.toUpperCase()) {
    case "R":
      return "Right-handed";
    case "L":
      return "Left-handed";
    case "U":
      return "Unknown";
    default:
      return hand;
  }
}

export function PlayerHero({
  player,
  latestRanking,
  latestElo,
  careerRecord,
}: PlayerHeroProps) {
  const age = computeAge(player.dob);
  const fullName = `${player.nameFirst} ${player.nameLast}`;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-2xl md:text-3xl">{fullName}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
          {/* Player info */}
          <div className="space-y-2 text-sm">
            {player.ioc && (
              <div>
                <span className="text-muted-foreground">Country:</span>{" "}
                <span className="font-medium">{player.ioc}</span>
              </div>
            )}
            {age !== null && (
              <div>
                <span className="text-muted-foreground">Age:</span>{" "}
                <span className="font-medium">{age}</span>
              </div>
            )}
            <div>
              <span className="text-muted-foreground">Hand:</span>{" "}
              <span className="font-medium">{formatHand(player.hand)}</span>
            </div>
            {player.height && (
              <div>
                <span className="text-muted-foreground">Height:</span>{" "}
                <span className="font-medium">{player.height} cm</span>
              </div>
            )}
          </div>

          {/* Rankings and stats */}
          <div className="flex flex-wrap gap-4">
            {latestRanking && (
              <div className="flex flex-col items-center gap-1">
                <span className="text-xs text-muted-foreground">Ranking</span>
                <Badge variant="default" className="text-lg px-3 py-1">
                  #{latestRanking.rank}
                </Badge>
                {latestRanking.points !== null && (
                  <span className="text-xs text-muted-foreground">
                    {latestRanking.points} pts
                  </span>
                )}
              </div>
            )}
            {latestElo && (
              <div className="flex flex-col items-center gap-1">
                <span className="text-xs text-muted-foreground">Elo Rating</span>
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  {Math.round(latestElo.overall)}
                </Badge>
              </div>
            )}
            <div className="flex flex-col items-center gap-1">
              <span className="text-xs text-muted-foreground">Career Record</span>
              <Badge variant="outline" className="text-lg px-3 py-1">
                {careerRecord.wins}W - {careerRecord.losses}L
              </Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
