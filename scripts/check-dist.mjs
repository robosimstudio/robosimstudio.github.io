// Gate a built site before it is pushed to GitHub Pages:
//   - total <= 900 MB (Pages: 1 GB per site), every file <= 95 MB (git: 100 MB);
//   - every manifest's files exist; every demo file but manifest.json has a content-hashed name
//     (Pages caches for 10 minutes, so an unhashed name would serve stale data after a redeploy);
//   - no LFS pointers; .nojekyll present.
// Usage: node scripts/check-dist.mjs site/dist
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const root = process.argv[2] ?? "site/dist";
const MAX_TOTAL = 900 * 1024 * 1024;
const MAX_FILE = 95 * 1024 * 1024;
let total = 0;
const problems = [];
const walk = (d) => {
  for (const n of readdirSync(d)) {
    const p = join(d, n);
    const s = statSync(p);
    if (s.isDirectory()) walk(p);
    else {
      total += s.size;
      if (s.size > MAX_FILE) problems.push(`${relative(root, p)} is ${(s.size / 1e6).toFixed(1)} MB (> 95 MB)`);
      if (s.size < 200 && readFileSync(p, "utf8").startsWith("version https://git-lfs")) problems.push(`${relative(root, p)} is an LFS pointer`);
    }
  }
};
walk(root);
if (total > MAX_TOTAL) problems.push(`site is ${(total / 1e6).toFixed(0)} MB (> 900 MB)`);
if (!existsSync(join(root, ".nojekyll"))) problems.push(".nojekyll missing");
const idx = join(root, "data", "demos.json");
if (!existsSync(idx)) problems.push("data/demos.json missing");
else {
  for (const d of JSON.parse(readFileSync(idx, "utf8")).demos) {
    const dir = join(root, "data", "demos", d.id);
    const m = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    const need = [];
    for (const v of Object.values(m.files)) (Array.isArray(v) ? v : [v]).forEach((f) => f && need.push(f));
    for (const f of m.stills?.files ?? []) need.push(f);
    for (const f of m.stills?.clean ?? []) if (f) need.push(f);
    for (const f of need) if (!existsSync(join(dir, f))) problems.push(`${d.id}: missing ${f}`);
    const walkNames = (sub) => {
      for (const n of readdirSync(join(dir, sub))) {
        const rel = sub ? `${sub}/${n}` : n;
        if (statSync(join(dir, rel)).isDirectory()) walkNames(rel);
        else if (rel !== "manifest.json" && !/\.[0-9a-f]{10}\.[a-z0-9.]+$/.test(n)) problems.push(`${d.id}: unhashed ${rel}`);
      }
    };
    walkNames("");
  }
}
console.log(`dist: ${(total / 1e6).toFixed(1)} MB`);
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("check-dist: ok");
