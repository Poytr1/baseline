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
    expect(result.winnerNew).toBeCloseTo(1516, 0);
    expect(result.loserNew).toBeCloseTo(1484, 0);
  });

  it("initial Elo is 1500", () => {
    expect(INITIAL_ELO).toBe(1500);
  });
});
