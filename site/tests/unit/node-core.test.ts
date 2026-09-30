// The core must run unbundled in plain Node (type stripping only: erasable syntax,
// `.ts` import specifiers, no DOM). Vitest transforms sources itself, so this runs
// every core module through `node --experimental-strip-types` in a child process.

import { execFileSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const site = fileURLToPath(new URL("../..", import.meta.url));

describe("src/core under node --experimental-strip-types", () => {
  it("imports every module and runs a round trip", () => {
    const modules = readdirSync(`${site}src/core`).filter((f) => f.endsWith(".ts"));
    const script = `
      for (const f of ${JSON.stringify(modules)}) await import("./src/core/" + f);
      const core = await import("./src/core/index.ts");
      const bytes = core.buildRssb({ q: { data: [0, 0.5, 1], shape: [3], dtype: "f32", enc: "q16d" } }, { frame0: 0 });
      console.log(JSON.stringify(Array.from(core.parseRssb(bytes).arrays.q.data)));
    `;
    const out = execFileSync(process.execPath, ["--experimental-strip-types", "--no-warnings", "--input-type=module", "-e", script], {
      cwd: site,
      encoding: "utf8",
    });
    expect(modules.length).toBeGreaterThanOrEqual(13);
    expect(JSON.parse(out)).toEqual([0, Math.fround(32768 / 65535), 1]);
  });
});
