# tennisconcrete Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modern web app alternative to Tennis Abstract with player profiles, Elo ratings, head-to-head comparisons, and stats leaderboards.

**Architecture:** Next.js 15 App Router with ISR, PostgreSQL (Neon) via Prisma, Tailwind + shadcn/ui. Data pipeline syncs Sackmann GitHub CSVs daily into Postgres, computes Elo ratings, and supplements with scraping for same-day results.

**Tech Stack:** Next.js 15, TypeScript, Prisma, PostgreSQL (Neon), Tailwind CSS, shadcn/ui, Recharts, Fuse.js, PapaParse, Vitest, Playwright

---

## Task 1: Project Scaffolding

**Files:**
- Create: `package.json`, `tsconfig.json`, `next.config.ts`, `tailwind.config.ts`, `postcss.config.mjs`, `app/layout.tsx`, `app/page.tsx`, `app/globals.css`, `.env.example`, `.gitignore`

**Step 1: Initialize Next.js project**

Run:
```bash
npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir=false --import-alias="@/*" --turbopack
```

Expected: Next.js project scaffolded in current directory with App Router, TypeScript, Tailwind, ESLint.

**Step 2: Install core dependencies**

Run:
```bash
npm install prisma @prisma/client recharts fuse.js papaparse
npm install -D @types/papaparse vitest @vitejs/plugin-react jsdom @testing-library/react @testing-library/jest-dom
```

**Step 3: Create `.env.example`**

```env
DATABASE_URL="postgresql://user:pass@host/dbname?sslmode=require"
```

Add `DATABASE_URL` with your actual Neon connection string to `.env.local` (gitignored).

**Step 4: Verify dev server starts**

Run: `npm run dev`
Expected: App running on http://localhost:3000

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: scaffold Next.js project with dependencies"
```

---

## Task 2: Database Schema with Prisma

**Files:**
- Create: `prisma/schema.prisma`
- Create: `lib/db.ts`

**Step 1: Initialize Prisma**

Run:
```bash
npx prisma init
```

**Step 2: Write the schema**

Replace `prisma/schema.prisma` with:

```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model Player {
  id         Int      @id
  nameFirst  String   @map("name_first")
  nameLast   String   @map("name_last")
  hand       String?
  dob        String?
  ioc        String?
  height     Int?
  wikidataId String?  @map("wikidata_id")
  tour       String   // "atp" or "wta"
  slug       String   @unique

  wonMatches  Match[] @relation("MatchWinner")
  lostMatches Match[] @relation("MatchLoser")
  rankings    Ranking[]
  eloRatings  EloRating[]

  @@map("players")
}

model Tournament {
  id        String  @id // e.g. "2024-0339"
  name      String
  surface   String?
  drawSize  Int?    @map("draw_size")
  level     String?
  date      String  // YYYYMMDD
  tour      String  // "atp" or "wta"

  matches   Match[]

  @@map("tournaments")
}

model Match {
  id           Int     @id @default(autoincrement())
  tourneyId    String  @map("tourney_id")
  matchNum     Int     @map("match_num")
  winnerId     Int     @map("winner_id")
  loserId      Int     @map("loser_id")
  score        String?
  bestOf       Int?    @map("best_of")
  round        String?
  minutes      Int?
  surface      String?
  tour         String  // "atp" or "wta"

  // Winner stats
  wAce     Int? @map("w_ace")
  wDf      Int? @map("w_df")
  wSvpt    Int? @map("w_svpt")
  w1stIn   Int? @map("w_1st_in")
  w1stWon  Int? @map("w_1st_won")
  w2ndWon  Int? @map("w_2nd_won")
  wSvGms   Int? @map("w_sv_gms")
  wBpSaved Int? @map("w_bp_saved")
  wBpFaced Int? @map("w_bp_faced")

  // Loser stats
  lAce     Int? @map("l_ace")
  lDf      Int? @map("l_df")
  lSvpt    Int? @map("l_svpt")
  l1stIn   Int? @map("l_1st_in")
  l1stWon  Int? @map("l_1st_won")
  l2ndWon  Int? @map("l_2nd_won")
  lSvGms   Int? @map("l_sv_gms")
  lBpSaved Int? @map("l_bp_saved")
  lBpFaced Int? @map("l_bp_faced")

  // Rankings at time of match
  winnerRank       Int? @map("winner_rank")
  winnerRankPoints Int? @map("winner_rank_points")
  loserRank        Int? @map("loser_rank")
  loserRankPoints  Int? @map("loser_rank_points")

  // Winner/Loser metadata at match time
  winnerAge Float? @map("winner_age")
  loserAge  Float? @map("loser_age")

  tournament Tournament @relation(fields: [tourneyId], references: [id])
  winner     Player     @relation("MatchWinner", fields: [winnerId], references: [id])
  loser      Player     @relation("MatchLoser", fields: [loserId], references: [id])

  @@unique([tourneyId, matchNum])
  @@index([winnerId])
  @@index([loserId])
  @@index([surface])
  @@index([tour])
  @@map("matches")
}

model Ranking {
  id       Int    @id @default(autoincrement())
  date     String // YYYYMMDD
  rank     Int
  playerId Int    @map("player_id")
  points   Int?
  tour     String // "atp" or "wta"

  player Player @relation(fields: [playerId], references: [id])

  @@unique([date, playerId, tour])
  @@index([playerId])
  @@index([date])
  @@map("rankings")
}

model EloRating {
  id        Int    @id @default(autoincrement())
  playerId  Int    @map("player_id")
  date      String // YYYYMMDD
  overall   Float
  hard      Float?
  clay      Float?
  grass     Float?
  tour      String // "atp" or "wta"

  player Player @relation(fields: [playerId], references: [id])

  @@unique([playerId, date, tour])
  @@index([playerId])
  @@index([date])
  @@index([overall])
  @@map("elo_ratings")
}
```

**Step 3: Create the Prisma client helper**

Create `lib/db.ts`:

```typescript
import { PrismaClient } from "@prisma/client";

const globalForPrisma = globalThis as unknown as { prisma: PrismaClient };

export const prisma = globalForPrisma.prisma || new PrismaClient();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
```

**Step 4: Generate the Prisma client and push schema**

Run:
```bash
npx prisma generate
npx prisma db push
```

Expected: Schema synced to Neon database, Prisma client generated.

**Step 5: Commit**

```bash
git add prisma/schema.prisma lib/db.ts
git commit -m "feat: add Prisma schema for players, matches, tournaments, rankings, elo"
```

---

## Task 3: Data Pipeline — CSV Sync Script

**Files:**
- Create: `scripts/sync-sackmann.ts`
- Create: `scripts/tsconfig.json`
- Modify: `package.json` (add sync script)

**Step 1: Create scripts tsconfig**

Create `scripts/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "esModuleInterop": true,
    "strict": true,
    "outDir": "./dist",
    "rootDir": "."
  },
  "include": ["./**/*.ts"]
}
```

**Step 2: Write the sync script**

Create `scripts/sync-sackmann.ts`. This script:
1. Downloads CSV files from `JeffSackmann/tennis_atp` and `JeffSackmann/tennis_wta` GitHub repos (raw URLs)
2. Parses them with PapaParse
3. Upserts players, tournaments, and matches into PostgreSQL via Prisma
4. Processes years 1968–current for matches, all ranking files, players file

Key logic:
- Fetch `atp_players.csv` and `wta_players.csv` → upsert into `players` table with `tour` field
- Generate `slug` from `nameFirst-nameLast-id` (lowercase, hyphens)
- Fetch `atp_matches_YYYY.csv` for each year → extract unique tournaments → upsert tournaments, then upsert matches
- Fetch ranking files → upsert rankings
- Use batch operations (`createMany` with `skipDuplicates`) for performance
- Log progress to stdout

The script should be ~200-300 lines. Map CSV column names to Prisma field names carefully:
- CSV `winner_id` → `winnerId`
- CSV `w_1stIn` → `w1stIn`
- CSV `tourney_id` → `tourneyId`
- etc.

**Step 3: Add npm script**

Add to `package.json` scripts:
```json
"sync": "npx tsx scripts/sync-sackmann.ts"
```

**Step 4: Test the sync with a single year**

Run: `npm run sync -- --year 2024`
Expected: Players, tournaments, matches for 2024 inserted into database.

**Step 5: Run full sync**

Run: `npm run sync`
Expected: All historical data loaded. This may take several minutes.

**Step 6: Commit**

```bash
git add scripts/ package.json
git commit -m "feat: add Sackmann CSV sync pipeline"
```

---

## Task 4: Elo Computation Script

**Files:**
- Create: `scripts/compute-elo.ts`
- Create: `lib/elo.ts`
- Create: `lib/__tests__/elo.test.ts`
- Create: `vitest.config.ts`

**Step 1: Create Vitest config**

Create `vitest.config.ts`:

```typescript
import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    environment: "node",
    globals: true,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});
```

**Step 2: Write the failing test for Elo calculation**

Create `lib/__tests__/elo.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { calculateNewElo, INITIAL_ELO } from "@/lib/elo";

describe("calculateNewElo", () => {
  it("returns higher rating for winner and lower for loser", () => {
    const result = calculateNewElo(1500, 1500);
    expect(result.winnerNew).toBeGreaterThan(1500);
    expect(result.loserNew).toBeLessThan(1500);
  });

  it("awards more points for an upset", () => {
    const upset = calculateNewElo(1300, 1700);
    const expected = calculateNewElo(1700, 1300);
    expect(upset.winnerNew - 1300).toBeGreaterThan(expected.winnerNew - 1700);
  });

  it("uses K-factor of 32 by default", () => {
    const result = calculateNewElo(1500, 1500);
    // When ratings are equal, expected score is 0.5, so change = K * 0.5 = 16
    expect(result.winnerNew).toBeCloseTo(1516, 0);
    expect(result.loserNew).toBeCloseTo(1484, 0);
  });

  it("initial Elo is 1500", () => {
    expect(INITIAL_ELO).toBe(1500);
  });
});
```

**Step 3: Run test to verify it fails**

Run: `npx vitest run lib/__tests__/elo.test.ts`
Expected: FAIL — module not found

**Step 4: Write the Elo calculation library**

Create `lib/elo.ts`:

```typescript
export const INITIAL_ELO = 1500;
const DEFAULT_K = 32;

export function calculateNewElo(
  winnerElo: number,
  loserElo: number,
  k: number = DEFAULT_K
): { winnerNew: number; loserNew: number } {
  const expectedWinner = 1 / (1 + Math.pow(10, (loserElo - winnerElo) / 400));
  const expectedLoser = 1 - expectedWinner;

  return {
    winnerNew: winnerElo + k * (1 - expectedWinner),
    loserNew: loserElo + k * (0 - expectedLoser),
  };
}
```

**Step 5: Run test to verify it passes**

Run: `npx vitest run lib/__tests__/elo.test.ts`
Expected: All 4 tests PASS

**Step 6: Write the compute-elo script**

Create `scripts/compute-elo.ts`. This script:
1. Fetches all matches from DB ordered by tournament date, then match_num
2. Maintains in-memory Elo maps: `overall`, `hard`, `clay`, `grass` per player
3. For each match, calculates new Elo for winner and loser (overall + surface-specific)
4. Snapshots Elo ratings into the `elo_ratings` table at regular intervals (e.g. after each tournament's matches)
5. Uses `INITIAL_ELO` (1500) for new players

**Step 7: Add npm script**

Add to `package.json` scripts:
```json
"compute-elo": "npx tsx scripts/compute-elo.ts"
```

**Step 8: Run Elo computation**

Run: `npm run compute-elo`
Expected: Elo ratings computed and stored. Top players should have ratings ~2000+.

**Step 9: Commit**

```bash
git add lib/elo.ts lib/__tests__/elo.test.ts scripts/compute-elo.ts vitest.config.ts package.json
git commit -m "feat: add Elo rating computation with tests"
```

---

## Task 5: shadcn/ui Setup and Layout Shell

**Files:**
- Modify: `app/layout.tsx`
- Modify: `app/globals.css`
- Create: `components/ui/` (shadcn components)
- Create: `components/navbar.tsx`
- Create: `components/theme-provider.tsx`
- Create: `lib/utils.ts`

**Step 1: Initialize shadcn/ui**

Run:
```bash
npx shadcn@latest init
```

Select: New York style, Zinc base color, CSS variables for colors.

**Step 2: Add needed shadcn components**

Run:
```bash
npx shadcn@latest add button input table card badge command dialog dropdown-menu
```

**Step 3: Install theme provider dependency**

Run:
```bash
npm install next-themes
```

**Step 4: Create theme provider**

Create `components/theme-provider.tsx`:

```tsx
"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";

export function ThemeProvider({
  children,
  ...props
}: React.ComponentProps<typeof NextThemesProvider>) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>;
}
```

**Step 5: Create the navbar**

Create `components/navbar.tsx` with:
- Site logo/name "tennisconcrete" linking to `/`
- Navigation links: Rankings, H2H, Stats
- Player search with Command palette (Cmd+K)
- Theme toggle (sun/moon icon)
- Responsive: hamburger menu on mobile

**Step 6: Update root layout**

Modify `app/layout.tsx`:
- Wrap children in `ThemeProvider`
- Add `Navbar` component
- Set metadata: title "tennisconcrete", description
- Import Inter font from `next/font/google`

**Step 7: Verify the shell renders**

Run: `npm run dev`
Expected: Navbar visible with links, theme toggle works, search opens command palette.

**Step 8: Commit**

```bash
git add app/ components/ lib/utils.ts
git commit -m "feat: add layout shell with navbar, theme toggle, search"
```

---

## Task 6: Player Search API and Autocomplete

**Files:**
- Create: `app/api/search/route.ts`
- Create: `components/player-search.tsx`

**Step 1: Write the search API route**

Create `app/api/search/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function GET(request: NextRequest) {
  const query = request.nextUrl.searchParams.get("q");
  if (!query || query.length < 2) {
    return NextResponse.json([]);
  }

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
}
```

**Step 2: Build the autocomplete component**

Create `components/player-search.tsx`:
- Uses the Command component from shadcn/ui
- Debounced input (300ms) calls `/api/search?q=...`
- Displays player name + country flag
- On select, navigates to `/player/[slug]`
- Keyboard accessible (arrow keys, enter to select)
- Opens via Cmd+K global shortcut

**Step 3: Integrate into navbar**

Replace the placeholder search in `components/navbar.tsx` with the `PlayerSearch` component.

**Step 4: Verify search works**

Run: `npm run dev`
Search for "Djokovic" — should show Novak Djokovic in results.

**Step 5: Commit**

```bash
git add app/api/search/ components/player-search.tsx components/navbar.tsx
git commit -m "feat: add player search API and autocomplete"
```

---

## Task 7: Player Profile Page

**Files:**
- Create: `app/player/[slug]/page.tsx`
- Create: `components/player-hero.tsx`
- Create: `components/player-stats.tsx`
- Create: `components/match-history.tsx`
- Create: `components/elo-chart.tsx`
- Create: `lib/queries/player.ts`

**Step 1: Write the data query functions**

Create `lib/queries/player.ts` with functions:
- `getPlayerBySlug(slug)` — fetch player + latest ranking + latest Elo
- `getPlayerCareerStats(playerId)` — aggregate win/loss by surface from matches
- `getPlayerMatchHistory(playerId, filters)` — paginated match history with optional surface/year/opponent filters
- `getPlayerEloHistory(playerId)` — Elo rating snapshots for charting

**Step 2: Build the player hero component**

Create `components/player-hero.tsx`:
- Name, country flag (use ioc code), age (computed from dob), hand, height
- Current ranking badge, current Elo rating
- Career wins/losses, titles count

**Step 3: Build the stats breakdown component**

Create `components/player-stats.tsx`:
- Cards for each surface (Hard, Clay, Grass) showing W-L record and win %
- Overall career stats: ace %, 1st serve %, break points saved %, etc.
- Use shadcn Card components

**Step 4: Build the match history table**

Create `components/match-history.tsx`:
- Sortable table: date, tournament, surface, round, opponent, score, W/L
- Filter dropdowns: year, surface, tournament level
- Paginated (20 per page)
- Client component with state for filters

**Step 5: Build the Elo chart**

Create `components/elo-chart.tsx`:
- Recharts LineChart showing Elo over time
- Toggle lines for overall, hard, clay, grass
- Tooltip showing date and rating value
- Responsive container

**Step 6: Assemble the player page**

Create `app/player/[slug]/page.tsx`:
- Server component with ISR (`revalidate: 3600`)
- `generateMetadata` for SEO (player name in title)
- Fetch all data via query functions
- Render: Hero → Stats → Elo Chart → Match History
- 404 if slug not found

**Step 7: Verify a player page**

Run: `npm run dev`, navigate to `/player/novak-djokovic-104925`
Expected: Full player profile with stats, chart, match history.

**Step 8: Commit**

```bash
git add app/player/ components/player-hero.tsx components/player-stats.tsx components/match-history.tsx components/elo-chart.tsx lib/queries/
git commit -m "feat: add player profile page with stats, Elo chart, match history"
```

---

## Task 8: Elo Rankings Page

**Files:**
- Create: `app/rankings/elo/page.tsx`
- Create: `components/elo-rankings-table.tsx`
- Create: `components/sparkline.tsx`
- Create: `lib/queries/rankings.ts`

**Step 1: Write ranking query functions**

Create `lib/queries/rankings.ts`:
- `getEloRankings(tour, surface)` — fetch latest Elo ratings for all players, sorted desc, with player name/country
- `getPlayerEloSparkline(playerId, days)` — last N Elo snapshots for mini-chart

**Step 2: Build the sparkline component**

Create `components/sparkline.tsx`:
- Tiny Recharts LineChart (80x24px), no axes/labels
- Shows last ~10 Elo data points as a trend line
- Green if trending up, red if trending down

**Step 3: Build the rankings table**

Create `components/elo-rankings-table.tsx`:
- Columns: Rank, Player (link to profile), Country, Elo Rating, Trend (sparkline)
- Client component with state for: tour toggle (ATP/WTA), surface toggle (Overall/Hard/Clay/Grass)
- Search filter input to find players within the table
- shadcn Table with sticky header

**Step 4: Assemble the rankings page**

Create `app/rankings/elo/page.tsx`:
- Server component with ISR (`revalidate: 3600`)
- Fetch default rankings (ATP, Overall)
- Pass to client table component
- Meta title: "Elo Rankings — tennisconcrete"

**Step 5: Verify rankings page**

Run: `npm run dev`, navigate to `/rankings/elo`
Expected: Table of players ranked by Elo, toggleable by tour and surface.

**Step 6: Commit**

```bash
git add app/rankings/ components/elo-rankings-table.tsx components/sparkline.tsx lib/queries/rankings.ts
git commit -m "feat: add Elo rankings page with sparklines"
```

---

## Task 9: Head-to-Head Page

**Files:**
- Create: `app/h2h/page.tsx`
- Create: `app/api/h2h/route.ts`
- Create: `components/h2h-comparison.tsx`
- Create: `components/h2h-match-list.tsx`
- Create: `lib/queries/h2h.ts`

**Step 1: Write H2H query functions**

Create `lib/queries/h2h.ts`:
- `getHeadToHead(player1Id, player2Id)` — all matches between two players
- `getH2HSummary(matches)` — compute overall record, record by surface, by tournament level, by year range

**Step 2: Write the H2H API route**

Create `app/api/h2h/route.ts`:
- Accepts `player1` and `player2` query params (player IDs)
- Returns full H2H data: summary + match list

**Step 3: Build the comparison component**

Create `components/h2h-comparison.tsx`:
- Side-by-side player cards with photos/flags
- Overall record displayed as a bar (e.g. green|red proportional bar)
- Breakdown tables: by surface, by tournament level
- Filter controls for year range

**Step 4: Build the match list component**

Create `components/h2h-match-list.tsx`:
- Table of all H2H matches: date, tournament, surface, round, score, winner highlighted

**Step 5: Assemble the H2H page**

Create `app/h2h/page.tsx`:
- Client component (dynamic, not ISR)
- Two player search inputs using the existing PlayerSearch component
- On both players selected, fetch H2H data from API route
- Render comparison + match list
- URL params: `?p1=slug1&p2=slug2` for shareable links

**Step 6: Verify H2H**

Run: `npm run dev`, navigate to `/h2h`, search for Djokovic and Nadal
Expected: Head-to-head record with breakdown and match list.

**Step 7: Commit**

```bash
git add app/h2h/ app/api/h2h/ components/h2h-comparison.tsx components/h2h-match-list.tsx lib/queries/h2h.ts
git commit -m "feat: add head-to-head comparison page"
```

---

## Task 10: Stats Leaderboards Page

**Files:**
- Create: `app/stats/page.tsx`
- Create: `app/api/stats/route.ts`
- Create: `components/stats-leaderboard.tsx`
- Create: `lib/queries/stats.ts`

**Step 1: Write stats query functions**

Create `lib/queries/stats.ts`:
- `getStatsLeaderboard(tour, surface, timePeriod, limit)` — aggregate player stats from matches table
- Computed stats include: ace %, double fault %, 1st serve in %, 1st serve won %, 2nd serve won %, serve points won %, break points saved %, return points won %, total points won %
- Filter by tour, surface, time period (last 52 weeks, career, specific year)
- Minimum match threshold to qualify (e.g. 20 matches)

**Step 2: Write the stats API route**

Create `app/api/stats/route.ts`:
- Accepts query params: `tour`, `surface`, `period`, `limit`, `sort`, `order`
- Returns sorted leaderboard data

**Step 3: Build the leaderboard component**

Create `components/stats-leaderboard.tsx`:
- Client component with filter controls: tour, surface, time period, player count (Top 50/100/All)
- Large sortable table with all stat columns
- Click any column header to sort
- Player names link to profiles
- Responsive: horizontal scroll on mobile, or column visibility toggle

**Step 4: Assemble the stats page**

Create `app/stats/page.tsx`:
- Server component with ISR (`revalidate: 3600`)
- Fetch default leaderboard (ATP, All surfaces, Last 52 weeks, Top 50)
- Pass to client component for interactive filtering
- Meta title: "Stats Leaderboards — tennisconcrete"

**Step 5: Verify stats page**

Run: `npm run dev`, navigate to `/stats`
Expected: Sortable stats table, filters work, players link to profiles.

**Step 6: Commit**

```bash
git add app/stats/ app/api/stats/ components/stats-leaderboard.tsx lib/queries/stats.ts
git commit -m "feat: add stats leaderboards page"
```

---

## Task 11: Homepage

**Files:**
- Modify: `app/page.tsx`
- Create: `components/top-elo-table.tsx`
- Create: `components/recent-results.tsx`
- Create: `lib/queries/home.ts`

**Step 1: Write homepage query functions**

Create `lib/queries/home.ts`:
- `getTopElo(tour, limit)` — top N players by current Elo
- `getRecentResults(limit)` — most recent match results from the matches table

**Step 2: Build the top Elo component**

Create `components/top-elo-table.tsx`:
- Compact table showing rank, player, country, Elo
- Tabs for ATP and WTA
- Player names link to profiles
- Shows top 10 by default

**Step 3: Build recent results component**

Create `components/recent-results.tsx`:
- List of recent matches: winner beat loser, score, tournament, surface
- Links to player profiles

**Step 4: Assemble the homepage**

Modify `app/page.tsx`:
- Server component with ISR (`revalidate: 1800`)
- Hero section with site tagline and search bar
- Two-column layout below: Top Elo (left), Recent Results (right)
- Attribution footer: "Data from Jeff Sackmann's Tennis Abstract datasets. Licensed CC BY-NC-SA 4.0."

**Step 5: Verify homepage**

Run: `npm run dev`, navigate to `/`
Expected: Homepage with search, top Elo tables for ATP/WTA, recent results.

**Step 6: Commit**

```bash
git add app/page.tsx components/top-elo-table.tsx components/recent-results.tsx lib/queries/home.ts
git commit -m "feat: add homepage with top Elo and recent results"
```

---

## Task 12: GitHub Actions Data Pipeline

**Files:**
- Create: `.github/workflows/sync-data.yml`

**Step 1: Write the GitHub Action**

Create `.github/workflows/sync-data.yml`:

```yaml
name: Sync Tennis Data

on:
  schedule:
    - cron: "0 6 * * *" # daily at 6am UTC
  workflow_dispatch: # manual trigger

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci
      - run: npx prisma generate
      - run: npm run sync
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
      - run: npm run compute-elo
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

**Step 2: Commit**

```bash
git add .github/
git commit -m "feat: add daily data sync GitHub Action"
```

---

## Task 13: E2E Tests with Playwright

**Files:**
- Create: `playwright.config.ts`
- Create: `e2e/homepage.spec.ts`
- Create: `e2e/player.spec.ts`
- Create: `e2e/rankings.spec.ts`
- Create: `e2e/h2h.spec.ts`

**Step 1: Install Playwright**

Run:
```bash
npm install -D @playwright/test
npx playwright install
```

**Step 2: Create Playwright config**

Create `playwright.config.ts`:

```typescript
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  use: {
    baseURL: "http://localhost:3000",
  },
  webServer: {
    command: "npm run dev",
    port: 3000,
    reuseExistingServer: !process.env.CI,
  },
});
```

**Step 3: Write E2E tests**

`e2e/homepage.spec.ts`:
- Page loads with title "tennisconcrete"
- Search bar is visible
- Top Elo tables render with player data
- Navigation links work

`e2e/player.spec.ts`:
- Player page renders with player name in heading
- Stats cards are visible
- Elo chart renders
- Match history table has rows

`e2e/rankings.spec.ts`:
- Rankings table renders with data
- Tour toggle switches between ATP/WTA
- Surface toggle changes displayed ratings

`e2e/h2h.spec.ts`:
- Two search inputs are visible
- Selecting two players shows comparison data

**Step 4: Run E2E tests**

Run: `npx playwright test`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add playwright.config.ts e2e/
git commit -m "test: add Playwright e2e tests for core pages"
```

---

## Task 14: Polish and Deploy

**Files:**
- Modify: `app/layout.tsx` (add favicon, OG meta)
- Create: `public/favicon.ico`
- Modify: `next.config.ts` (image domains if needed)

**Step 1: Add metadata and SEO**

Update `app/layout.tsx` metadata:
- Title template: `%s | tennisconcrete`
- Default description
- Open Graph tags

**Step 2: Deploy to Vercel**

Run:
```bash
npx vercel
```

Follow prompts to link project and deploy. Set `DATABASE_URL` environment variable in Vercel dashboard.

**Step 3: Verify production deployment**

Visit the Vercel URL, check all pages load correctly with data.

**Step 4: Add `DATABASE_URL` secret to GitHub repo**

In GitHub repo settings → Secrets → add `DATABASE_URL` so the GitHub Action can run.

**Step 5: Commit any remaining changes**

```bash
git add -A
git commit -m "chore: add metadata, SEO, deployment config"
```

---

## Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | Project Scaffolding | — |
| 2 | Database Schema (Prisma) | 1 |
| 3 | CSV Sync Script | 2 |
| 4 | Elo Computation | 2, 3 |
| 5 | Layout Shell (shadcn/ui, navbar, theme) | 1 |
| 6 | Player Search API + Autocomplete | 2, 5 |
| 7 | Player Profile Page | 4, 6 |
| 8 | Elo Rankings Page | 4, 6 |
| 9 | Head-to-Head Page | 6, 7 |
| 10 | Stats Leaderboards Page | 3, 6 |
| 11 | Homepage | 7, 8 |
| 12 | GitHub Actions Pipeline | 3, 4 |
| 13 | E2E Tests | 7, 8, 9, 10, 11 |
| 14 | Polish and Deploy | 13 |
