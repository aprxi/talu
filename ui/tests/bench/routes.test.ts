import { describe, expect, test } from "bun:test";
import { inspectBenchPage } from "../../src/bench/app.ts";

describe("bench route hierarchy", () => {
  test("db vector alias resolves to canonical vectors page", () => {
    const page = inspectBenchPage("/bench/db/vector/");
    expect(page.status).toBe("ok");
    if (page.status !== "ok") return;
    expect(page.slug).toBe("db/vectors");
    expect(page.scenarioIds.length).toBeGreaterThan(0);
  });

  test("responses evals page exposes suite subpages from real harness", () => {
    const page = inspectBenchPage("/bench/responses/evals/");
    expect(page.status).toBe("ok");
    if (page.status !== "ok") return;

    expect(page.childSlugs).toEqual([
      "responses/evals/mmlu",
      "responses/evals/gpqa",
      "responses/evals/ifeval",
      "responses/evals/bfcl",
      "responses/evals/mmmu",
    ]);
  });

  test("each eval team page has scoped scenarios", () => {
    const suites = [
      "/bench/responses/evals/mmlu/",
      "/bench/responses/evals/gpqa/",
      "/bench/responses/evals/ifeval/",
      "/bench/responses/evals/bfcl/",
      "/bench/responses/evals/mmmu/",
    ];

    for (const suite of suites) {
      const page = inspectBenchPage(suite);
      expect(page.status).toBe("ok");
      if (page.status !== "ok") continue;
      expect(page.scenarioIds.length).toBeGreaterThan(0);
    }
  });

  test("results section has responses and db history pages", () => {
    const page = inspectBenchPage("/bench/results/");
    expect(page.status).toBe("ok");
    if (page.status !== "ok") return;
    expect(page.childSlugs).toEqual(["results/responses", "results/db"]);
  });

  test("responses perf page exposes canonical pp/tg scenario matrix", () => {
    const page = inspectBenchPage("/bench/responses/perf/");
    expect(page.status).toBe("ok");
    if (page.status !== "ok") return;

    const required = [
      "responses/perf/pp512",
      "responses/perf/pp1024",
      "responses/perf/pp2048",
      "responses/perf/pp4096",
      "responses/perf/tg128",
      "responses/perf/tg256",
      "responses/perf/tg512",
    ];
    for (const id of required) {
      expect(page.scenarioIds).toContain(id);
    }
  });

  test("db vectors page includes extended vector perf operations", () => {
    const page = inspectBenchPage("/bench/db/vectors/");
    expect(page.status).toBe("ok");
    if (page.status !== "ok") return;

    const required = [
      "vector_create_collection",
      "vector_append",
      "vector_query",
      "vector_fetch",
      "vector_upsert",
      "vector_delete",
      "vector_stats",
      "vector_changes",
      "vector_compact",
      "vector_indexes_build",
    ];
    for (const id of required) {
      expect(page.scenarioIds).toContain(id);
    }
  });
});
