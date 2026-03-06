import { test, expect } from "@playwright/test";

test.describe("Head to Head Page", () => {
  test("page loads successfully", async ({ page }) => {
    const response = await page.goto("/h2h");
    expect(response?.status()).toBe(200);
  });

  test("has Head to Head heading", async ({ page }) => {
    await page.goto("/h2h");
    const heading = page.locator("h1", { hasText: "Head to Head" });
    await expect(heading).toBeVisible();
  });

  test("two player search inputs are visible", async ({ page }) => {
    await page.goto("/h2h");
    const inputs = page.locator("input[placeholder='Search player...']");
    await expect(inputs).toHaveCount(2);
    await expect(inputs.first()).toBeVisible();
    await expect(inputs.last()).toBeVisible();
  });

  test("vs separator text is present", async ({ page }) => {
    await page.goto("/h2h");
    const vs = page.locator("text=vs");
    await expect(vs.first()).toBeVisible();
  });
});
