import { test, expect } from "@playwright/test";

test.describe("Player Page", () => {
  test("returns 404 for nonexistent player", async ({ page }) => {
    const response = await page.goto("/player/nonexistent-player-0");
    expect(response?.status()).toBe(404);
  });

  test("404 page has meaningful content", async ({ page }) => {
    await page.goto("/player/nonexistent-player-0");
    // Next.js shows a "Not Found" message for 404 pages
    const body = page.locator("body");
    await expect(body).toContainText(/not found/i);
  });
});
