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
