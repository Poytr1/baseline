import type { Metadata } from "next";
import { Suspense } from "react";
import { Search } from "lucide-react";
import { getTopElo, getRecentResults } from "@/lib/queries/home";
import { TopEloTable } from "@/components/top-elo-table";
import { RecentResults } from "@/components/recent-results";

export const revalidate = 1800; // ISR: 30 minutes

export const metadata: Metadata = {
  title: "tennisconcrete — Modern Tennis Analytics",
  description:
    "Modern tennis analytics powered by data. Elo ratings, head-to-head comparisons, and statistical leaderboards.",
};

async function TopEloSection() {
  const [atpTop, wtaTop] = await Promise.all([
    getTopElo("atp", 10),
    getTopElo("wta", 10),
  ]);

  return <TopEloTable atpData={atpTop} wtaData={wtaTop} />;
}

async function RecentResultsSection() {
  const recentResults = await getRecentResults(15);

  return <RecentResults results={recentResults} />;
}

export default function HomePage() {
  return (
    <div className="space-y-10">
      {/* Hero */}
      <section className="flex flex-col items-center gap-4 pt-8 pb-4 text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          tennisconcrete
        </h1>
        <p className="max-w-lg text-lg text-muted-foreground">
          Modern tennis analytics powered by data
        </p>
        <div className="mt-2">
          <kbd className="inline-flex items-center gap-1.5 rounded-lg border bg-muted px-4 py-2 text-sm text-muted-foreground">
            <Search className="size-4" />
            <span>Search players...</span>
            <span className="ml-2 rounded border bg-background px-1.5 py-0.5 font-mono text-xs">
              <span>&#8984;</span>K
            </span>
          </kbd>
        </div>
      </section>

      {/* Two-column layout */}
      <section className="grid gap-8 lg:grid-cols-2">
        <Suspense
          fallback={
            <p className="py-8 text-center text-sm text-muted-foreground">
              Loading rankings...
            </p>
          }
        >
          <TopEloSection />
        </Suspense>
        <Suspense
          fallback={
            <p className="py-8 text-center text-sm text-muted-foreground">
              Loading recent results...
            </p>
          }
        >
          <RecentResultsSection />
        </Suspense>
      </section>

      {/* Attribution */}
      <footer className="border-t pt-6 pb-4 text-center text-xs text-muted-foreground">
        Data from Jeff Sackmann&apos;s{" "}
        <a
          href="https://github.com/JeffSackmann/tennis_atp"
          className="underline hover:text-foreground"
          target="_blank"
          rel="noopener noreferrer"
        >
          Tennis Abstract
        </a>{" "}
        datasets. Licensed CC BY-NC-SA 4.0.
      </footer>
    </div>
  );
}
