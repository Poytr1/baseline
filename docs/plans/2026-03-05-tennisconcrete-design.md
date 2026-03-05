# tennisconcrete — Design Document

A modern web app alternative to Tennis Abstract, providing the same rich tennis analytics data with a clean, responsive, mobile-friendly interface.

## Goals

- Rebuild Tennis Abstract's core features with modern UI/UX
- Use Jeff Sackmann's open datasets as the primary data source, supplemented with scraping for same-day results
- Deliver fast, static-first pages with ISR for near-real-time freshness
- Support both ATP and WTA tours

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | Next.js 15 (App Router, TypeScript) |
| Styling | Tailwind CSS + shadcn/ui (Radix primitives) |
| Database | PostgreSQL (Neon, serverless) |
| ORM | Prisma |
| Charts | Recharts |
| Search | Fuse.js (client-side fuzzy matching) |
| CSV Parsing | PapaParse |
| Testing | Vitest (unit), Playwright (e2e) |
| Hosting | Vercel (frontend), Neon (database), GitHub Actions (data pipeline) |

## Data Layer

### Data Sources

- **Primary:** Jeff Sackmann's GitHub repos (`tennis_atp`, `tennis_wta`) — CSV files with matches (1968–present), rankings, and player metadata. Licensed CC BY-NC-SA 4.0.
- **Supplement:** Lightweight scraper hitting ATP/WTA official sites for same-day results not yet reflected in the CSVs.

### Database Schema

- `players` — id, name, country, birth_date, hand, height, wikidata_id
- `matches` — id, tourney_id, surface, round, winner_id, loser_id, score, detailed stat columns (aces, double faults, serve points, break points, etc.), date
- `tournaments` — id, name, surface, level, draw_size, date
- `rankings` — player_id, rank, points, date
- `elo_ratings` — player_id, elo_overall, elo_hard, elo_clay, elo_grass, date

### Data Pipeline

- GitHub Action runs daily on a cron schedule
- Pulls latest CSVs from Sackmann repos, diffs against existing data, upserts new rows into PostgreSQL
- Supplemental scraper runs every few hours for in-progress tournament results
- Elo ratings are computed from match data using the standard Elo formula (with surface-specific variants), recalculated on each data sync

## Core Features (MVP)

### Player Profiles (`/player/[slug]`)

- Hero section: name, country flag, age, hand, height, current ranking, current Elo
- Career summary: titles, win/loss record, career high ranking
- Stats breakdown by surface (hard, clay, grass) with win percentages
- Match history table with filtering (year, surface, tournament level, opponent)
- Elo rating chart over time (line graph)

### Elo Rankings (`/rankings/elo`)

- Sortable table of all players by Elo rating
- Toggle between overall, hard, clay, grass Elo
- Toggle between ATP and WTA
- Search/filter within the table
- Sparkline mini-charts showing recent Elo trend per player

### Head-to-Head (`/h2h`)

- Two player search inputs with autocomplete
- Overall record, then broken down by surface, tournament level, year range
- List of all matches between the two players with scores
- Visual comparison of key stats side-by-side

### Stats Leaderboards (`/stats`)

- ~60 sortable statistics (ace %, 1st serve %, break points saved, etc.)
- Filter by tour (ATP/WTA), time period, surface
- Top 50 / Top 100 / all players toggle
- Click column headers to sort; click player names to navigate to profiles

### Global

- Homepage with search bar, current top-10 Elo, and recent notable results
- Global player search with autocomplete in the navbar on every page
- Responsive design for desktop and mobile
- Dark/light theme toggle

## Architecture

### App Router Structure

```
app/
  page.tsx                # Homepage
  player/[slug]/page.tsx  # Player profile (ISR, revalidate: 3600)
  rankings/elo/page.tsx   # Elo rankings (ISR, revalidate: 3600)
  h2h/page.tsx            # Head-to-head (client-side dynamic)
  stats/page.tsx          # Stats leaderboards (ISR, revalidate: 3600)
  api/
    search/route.ts       # Player autocomplete search
    h2h/route.ts          # H2H query endpoint
    stats/route.ts        # Filtered stats endpoint
```

### Rendering Strategy (Static-First with ISR)

- Player profiles, Elo rankings, stats leaderboards: statically generated, revalidate every 1 hour
- Homepage: revalidate every 30 minutes
- H2H and filtered queries: dynamic via API routes
- Server components fetch data directly from the database (no API layer for static pages)
- API routes only for interactive/dynamic features

### Data Pipeline Scripts

```
scripts/
  sync-sackmann.ts    # Pull CSVs from GitHub, parse, upsert to DB
  compute-elo.ts      # Recalculate Elo ratings from match history
  scrape-recent.ts    # Supplement with same-day results
```

## UI/UX Design

### Design Philosophy

- Clean, data-dense but not cluttered
- Typography-driven hierarchy: monospace for numbers/stats, sans-serif for labels
- High-contrast tables with alternating rows and sticky headers

### Key UI Patterns

- **Data tables** — sortable, filterable, paginated, column visibility toggles on mobile
- **Autocomplete search** — debounced, fuzzy-matching, keyboard accessible
- **Stat cards** — compact label + value + trend indicator
- **Comparison layout** — side-by-side panels for H2H
- **Responsive** — tables collapse to cards on mobile, charts resize fluidly

### Color Palette

- Neutral base: slate/zinc grays
- Accent: teal/green (tennis court inspired)
- Surface indicators: blue (hard), orange/brown (clay), green (grass)
- Full dark mode support (system preference or manual toggle)

## Deployment & Operations

- **Frontend:** Vercel (free tier to start)
- **Database:** Neon PostgreSQL (free tier, 0.5 GB)
- **Pipeline:** GitHub Actions (daily cron)
- **Monitoring:** Vercel Analytics, GitHub Actions logs, Neon dashboard
- **Scaling:** ISR + edge CDN means database is only hit on revalidation. Upgrade to Vercel Pro / Neon Pro if traffic grows.

## License Considerations

- Sackmann data is CC BY-NC-SA 4.0: attribution required, non-commercial use only
- tennisconcrete must include visible attribution to Jeff Sackmann and the data source
- Monetization would require a separate license or independent dataset
