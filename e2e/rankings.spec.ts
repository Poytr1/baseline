import { test, expect } from "@playwright/test";

test.describe("Rankings Page", () => {
  test("page loads successfully", async ({ page }) => {
    const response = await page.goto("/rankings");
    expect(response?.status()).toBe(200);
  });

  test("has Elo Rankings heading", async ({ page }) => {
    await page.goto("/rankings");
    const heading = page.locator("h1", { hasText: "Elo Rankings" });
    await expect(heading).toBeVisible();
  });

  test("tour toggle buttons are visible", async ({ page }) => {
    await page.goto("/rankings");
    const atpButton = page.getByRole("button", { name: "ATP" });
    const wtaButton = page.getByRole("button", { name: "WTA" });
    await expect(atpButton).toBeVisible();
    await expect(wtaButton).toBeVisible();
  });

  test("surface toggle buttons are visible", async ({ page }) => {
    await page.goto("/rankings");
    const overall = page.getByRole("button", { name: "Overall" });
    const hard = page.getByRole("button", { name: "Hard" });
    const clay = page.getByRole("button", { name: "Clay" });
    const grass = page.getByRole("button", { name: "Grass" });
    await expect(overall).toBeVisible();
    await expect(hard).toBeVisible();
    await expect(clay).toBeVisible();
    await expect(grass).toBeVisible();
  });

  test("rankings table structure is present", async ({ page }) => {
    await page.goto("/rankings");
    // The table should be present (might be empty if no data, but structure exists)
    const table = page.locator("table");
    await expect(table).toBeAttached();
  });
});
