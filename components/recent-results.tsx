import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import type { RecentResultsData } from "@/lib/queries/home";

interface RecentResultsProps {
  results: RecentResultsData;
}

function formatDate(dateStr: string): string {
  if (!dateStr || dateStr.length !== 8) return dateStr;
  const y = dateStr.substring(0, 4);
  const m = dateStr.substring(4, 6);
  const d = dateStr.substring(6, 8);
  return `${y}-${m}-${d}`;
}

function getSurfaceBadgeClass(surface: string): string {
  switch (surface.toLowerCase()) {
    case "hard":
      return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
    case "clay":
      return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200";
    case "grass":
      return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    default:
      return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200";
  }
}

export function RecentResults({ results }: RecentResultsProps) {
  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold">Recent Results</h2>

      {results.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No recent matches available yet.
        </p>
      ) : (
        <div className="space-y-2">
          {results.map((match) => (
            <div
              key={match.id}
              className="rounded-lg border p-3 transition-colors hover:bg-muted/50"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <p className="text-sm">
                    <Link
                      href={`/player/${match.winnerSlug}`}
                      className="font-semibold hover:underline"
                    >
                      {match.winnerName}
                    </Link>
                    <span className="text-muted-foreground"> d. </span>
                    <Link
                      href={`/player/${match.loserSlug}`}
                      className="hover:underline"
                    >
                      {match.loserName}
                    </Link>
                  </p>
                  {match.score && (
                    <p className="mt-0.5 font-mono text-xs text-muted-foreground">
                      {match.score}
                    </p>
                  )}
                </div>
                <Badge
                  variant="outline"
                  className={getSurfaceBadgeClass(match.surface)}
                >
                  {match.surface}
                </Badge>
              </div>
              <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                <span>{match.tournamentName}</span>
                <span>&middot;</span>
                <span>{formatDate(match.date)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
