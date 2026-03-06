"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";

interface PlayerResult {
  id: number;
  nameFirst: string;
  nameLast: string;
  ioc: string | null;
  slug: string;
  tour: string;
}

interface PlayerSearchProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

export function PlayerSearch({ open, setOpen }: PlayerSearchProps) {
  const router = useRouter();
  const [query, setQuery] = React.useState("");
  const [results, setResults] = React.useState<PlayerResult[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [hasSearched, setHasSearched] = React.useState(false);

  // Global Cmd+K keyboard shortcut
  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen(!open);
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, [open, setOpen]);

  // Debounced search
  React.useEffect(() => {
    if (query.length < 2) {
      setResults([]);
      setHasSearched(false);
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
          setHasSearched(true);
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

  // Reset state when dialog closes
  React.useEffect(() => {
    if (!open) {
      setQuery("");
      setResults([]);
      setHasSearched(false);
      setLoading(false);
    }
  }, [open]);

  const handleSelect = (slug: string) => {
    setOpen(false);
    router.push(`/player/${slug}`);
  };

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Search Players"
      description="Search for a tennis player by name"
    >
      <CommandInput
        placeholder="Search players..."
        value={query}
        onValueChange={setQuery}
      />
      <CommandList>
        {loading && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          </div>
        )}
        {!loading && hasSearched && results.length === 0 && (
          <CommandEmpty>No results found.</CommandEmpty>
        )}
        {!loading && results.length > 0 && (
          <CommandGroup heading="Players">
            {results.map((player) => (
              <CommandItem
                key={player.id}
                value={`${player.nameFirst} ${player.nameLast}`}
                onSelect={() => handleSelect(player.slug)}
              >
                <span className="flex-1">
                  {player.nameFirst} {player.nameLast}
                </span>
                {player.ioc && (
                  <Badge variant="outline" className="ml-2 text-xs">
                    {player.ioc}
                  </Badge>
                )}
                <Badge variant="secondary" className="ml-1 text-xs uppercase">
                  {player.tour}
                </Badge>
              </CommandItem>
            ))}
          </CommandGroup>
        )}
      </CommandList>
    </CommandDialog>
  );
}
