import { test, expect } from "@playwright/test";

test.describe("Homepage", () => {
  test("page loads successfully", async ({ page }) => {
    const response = await page.goto("/");
    expect(response?.status()).toBe(200);
  });

  test("title contains tennisconcrete", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/tennisconcrete/i);
  });

  test("hero section is visible with site name", async ({ page }) => {
    await page.goto("/");
    const hero = page.locator("h1", { hasText: "tennisconcrete" });
    await expect(hero).toBeVisible();
  });

  test("Cmd+K search hint is visible", async ({ page }) => {
    await page.goto("/");
    const kbd = page.locator("kbd", { hasText: "K" });
    await expect(kbd.first()).toBeVisible();
  });

  test("navigation links are present", async ({ page }) => {
    await page.goto("/");
    // Desktop nav links in header
    const rankings = page.locator("header a", { hasText: "Rankings" });
    const h2h = page.locator("header a", { hasText: "H2H" });
    const stats = page.locator("header a", { hasText: "Stats" });

    await expect(rankings.first()).toBeAttached();
    await expect(h2h.first()).toBeAttached();
    await expect(stats.first()).toBeAttached();
  });

  test("attribution footer is visible", async ({ page }) => {
    await page.goto("/");
    const footer = page.locator("footer", { hasText: "Jeff Sackmann" });
    await expect(footer).toBeVisible();
  });
});
