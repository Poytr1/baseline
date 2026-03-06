"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";

import { Input } from "@/components/ui/input";
import { H2HComparison } from "@/components/h2h-comparison";
import { H2HMatchList } from "@/components/h2h-match-list";
import type { H2HSummary } from "@/lib/queries/h2h";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface PlayerOption {
  id: number;
  nameFirst: string;
  nameLast: string;
  ioc: string | null;
  slug: string;
  tour: string;
}

interface H2HMatchEntry {
  id: number;
  date: string;
  tournamentName: string;
  tournamentLevel: string;
  surface: string;
  round: string;
  score: string;
  winnerId: number;
  winnerName: string;
  winnerSlug: string;
  loserId: number;
  loserName: string;
  loserSlug: string;
}

export interface H2HData {
  player1: PlayerOption;
  player2: PlayerOption;
  crossTour: boolean;
  summary: H2HSummary | null;
  matches: H2HMatchEntry[];
}

/* ------------------------------------------------------------------ */
/*  Player Autocomplete Input                                          */
/* ------------------------------------------------------------------ */

function PlayerAutocomplete({
  label,
  value,
  onSelect,
}: {
  label: string;
  value: PlayerOption | null;
  onSelect: (player: PlayerOption) => void;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PlayerOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Display selected player name or allow typing
  const displayValue = value
    ? `${value.nameFirst} ${value.nameLast}`
    : query;

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Debounced search
  useEffect(() => {
    if (query.length < 2) {
      setResults([]);
      return;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(
          `/api/search?q=${encodeURIComponent(query)}`,
          { signal: controller.signal }
        );
        if (res.ok) {
          const data = await res.json();
          setResults(data);
          setOpen(true);
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === "AbortError")) {
          console.error("Search failed:", err);
        }
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => {
      clearTimeout(timeoutId);
      controller.abort();
    };
  }, [query]);

  function handleInputChange(val: string) {
    setQuery(val);
    // If user starts typing again after selecting, clear the selection
    if (value) {
      onSelect(null as unknown as PlayerOption); // signal parent to clear
    }
  }

  function handleSelect(player: PlayerOption) {
    onSelect(player);
    setQuery("");
    setOpen(false);
    setResults([]);
  }

  return (
    <div ref={containerRef} className="relative flex-1">
      <label className="mb-1 block text-sm font-medium text-muted-foreground">
        {label}
      </label>
      <div className="relative">
        <Input
          placeholder="Search player..."
          value={displayValue}
          onChange={(e) => handleInputChange(e.target.value)}
          onFocus={() => {
            if (value) {
              // Clear selection so user can re-search
              onSelect(null as unknown as PlayerOption);
              setQuery("");
            }
            if (results.length > 0) {
              setOpen(true);
            }
          }}
        />
        {loading && (
          <Loader2 className="absolute right-3 top-1/2 size-4 -translate-y-1/2 animate-spin text-muted-foreground" />
        )}
      </div>
      {open && results.length > 0 && (
        <div className="absolute z-50 mt-1 w-full rounded-md border bg-popover p-1 shadow-md">
          {results.map((player) => (
            <button
              key={player.id}
              type="button"
              className="flex w-full items-center gap-2 rounded-sm px-3 py-2 text-left text-sm hover:bg-accent hover:text-accent-foreground"
              onClick={() => handleSelect(player)}
            >
              <span className="flex-1">
                {player.nameFirst} {player.nameLast}
              </span>
              {player.ioc && (
                <span className="rounded border px-1.5 py-0.5 text-xs text-muted-foreground">
                  {player.ioc}
                </span>
              )}
              <span className="rounded bg-secondary px-1.5 py-0.5 text-xs uppercase text-secondary-foreground">
                {player.tour}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main H2H Page Content                                              */
/* ------------------------------------------------------------------ */

function H2HPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [player1, setPlayer1] = useState<PlayerOption | null>(null);
  const [player2, setPlayer2] = useState<PlayerOption | null>(null);
  const [h2hData, setH2hData] = useState<H2HData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initialLoadDone = useRef(false);

  // Load players from URL params on mount
  useEffect(() => {
    if (initialLoadDone.current) return;
    initialLoadDone.current = true;

    const p1Slug = searchParams.get("p1");
    const p2Slug = searchParams.get("p2");

    if (p1Slug && p2Slug) {
      // Fetch both players by slug via the search API approach
      // We'll resolve them from the H2H API which returns player info
      (async () => {
        setLoading(true);
        try {
          const res = await fetch(
            `/api/h2h?player1=${encodeURIComponent(p1Slug)}&player2=${encodeURIComponent(p2Slug)}`
          );
          if (res.ok) {
            const data: H2HData = await res.json();
            setPlayer1(data.player1);
            setPlayer2(data.player2);
            setH2hData(data);
            setError(null);
          } else {
            const body = await res.json();
            setError(body.error ?? "Failed to load head-to-head data");
          }
        } catch {
          setError("Failed to load head-to-head data");
        } finally {
          setLoading(false);
        }
      })();
    }
  }, [searchParams]);

  // Fetch H2H data when both players are selected
  const fetchH2H = useCallback(
    async (p1: PlayerOption, p2: PlayerOption) => {
      if (p1.slug === p2.slug) {
        setError("Please select two different players");
        setH2hData(null);
        return;
      }

      // Update URL
      const params = new URLSearchParams();
      params.set("p1", p1.slug);
      params.set("p2", p2.slug);
      router.replace(`/h2h?${params.toString()}`, { scroll: false });

      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `/api/h2h?player1=${encodeURIComponent(p1.slug)}&player2=${encodeURIComponent(p2.slug)}`
        );
        if (res.ok) {
          const data: H2HData = await res.json();
          setH2hData(data);
        } else {
          const body = await res.json();
          setError(body.error ?? "Failed to load head-to-head data");
          setH2hData(null);
        }
      } catch {
        setError("Failed to load head-to-head data");
        setH2hData(null);
      } finally {
        setLoading(false);
      }
    },
    [router]
  );

  function handlePlayer1Select(player: PlayerOption) {
    setPlayer1(player);
    if (player && player2) {
      fetchH2H(player, player2);
    } else {
      setH2hData(null);
      setError(null);
    }
  }

  function handlePlayer2Select(player: PlayerOption) {
    setPlayer2(player);
    if (player1 && player) {
      fetchH2H(player1, player);
    } else {
      setH2hData(null);
      setError(null);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Head to Head</h1>
        <p className="text-muted-foreground">
          Compare two players and view their head-to-head record
        </p>
      </div>

      {/* Player selection inputs */}
      <div className="flex flex-col gap-4 sm:flex-row">
        <PlayerAutocomplete
          label="Player 1"
          value={player1}
          onSelect={handlePlayer1Select}
        />
        <div className="flex items-end justify-center pb-1 text-lg font-bold text-muted-foreground">
          vs
        </div>
        <PlayerAutocomplete
          label="Player 2"
          value={player2}
          onSelect={handlePlayer2Select}
        />
      </div>

      {/* Error state */}
      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
        </div>
      )}

      {/* Cross-tour message */}
      {h2hData && h2hData.crossTour && (
        <div className="rounded-md border bg-muted/50 px-4 py-3 text-sm text-muted-foreground">
          Cross-tour head-to-head is not available.{" "}
          {h2hData.player1.nameFirst} {h2hData.player1.nameLast} plays on the{" "}
          {h2hData.player1.tour.toUpperCase()} tour, while{" "}
          {h2hData.player2.nameFirst} {h2hData.player2.nameLast} plays on the{" "}
          {h2hData.player2.tour.toUpperCase()} tour.
        </div>
      )}

      {/* H2H results */}
      {h2hData && !h2hData.crossTour && !loading && (
        <div className="space-y-8">
          <H2HComparison data={h2hData} />
          <H2HMatchList data={h2hData} />
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Page Export (wrapped in Suspense for useSearchParams)              */
/* ------------------------------------------------------------------ */

import { Suspense } from "react";

export default function H2HPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-12">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <H2HPageContent />
    </Suspense>
  );
}
