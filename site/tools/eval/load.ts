// Load a built demo's data exactly as the page does, in Node: a quick check that
// the pipeline's files and the core agree. Usage:
//   node --experimental-strip-types tools/eval/load.ts ../build/site-data/demos/wipe

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { parseRssb } from "../../src/core/rssb.ts";
import { FrameStore } from "../../src/core/store.ts";
import { Matcher } from "../../src/core/matcher.ts";
import { Kinematics } from "../../src/core/kinematics.ts";
import { Camera } from "../../src/core/camera.ts";
import { DragMapping } from "../../src/core/drag.ts";

const dir = process.argv[2].replace(/\/$/, "") + "/";
const m = JSON.parse(readFileSync(dir + "manifest.json", "utf8"));
const rssb = (f: string) => parseRssb(new Uint8Array(gunzipSync(readFileSync(dir + f))));

const t0 = performance.now();
const chunk0 = rssb(m.files.db[0]);
console.log("chunk0 arrays", Object.fromEntries(Object.entries(chunk0.arrays).map(([k, v]) => [k, v.shape])), "meta", chunk0.meta);
const store = new FrameStore(chunk0);
console.log("store n", store.n, "takes", store.takes.length, "dims", store.dims);
const matcher = new Matcher(store, m.features.groups);
for (let i = 1; i < m.files.db.length; i++) {
  store.append(rssb(m.files.db[i]));
  matcher.sync();
}
console.log("all chunks", store.n, "frames in", (performance.now() - t0).toFixed(0), "ms");
const kin = new Kinematics(JSON.parse(readFileSync(dir + m.files.kin, "utf8")));
console.log("fk selfTest", kin.selfTest());
// the tool centre from FK against the recorded hand
const poses = new Map();
let worst = 0;
const nq = store.dims.q as number;
for (let f = 0; f < store.n; f += 997) {
  kin.fk(store.q.subarray(f * nq, (f + 1) * nq), poses);
  const t = kin.tcp(m.deform_arm ?? "right", poses);
  const h = store.hand.subarray(3 * f, 3 * f + 3);
  worst = Math.max(worst, Math.hypot(t.p[0] - h[0], t.p[1] - h[1], t.p[2] - h[2]));
}
console.log("tcp(FK) vs recorded hand, worst [mm]", (worst * 1e3).toFixed(3));
const cam = new Camera(m.camera);
const drag = new DragMapping(m.drag);
const hover = drag.toWorld(m.drag.hover[0], m.drag.hover[1]);
console.log("hover world", hover.map((v) => v.toFixed(3)), "projects to", cam.project(hover).map((v) => v.toFixed(1)));
const deform = rssb(m.files.deform);
console.log("deform", Object.fromEntries(Object.entries(deform.arrays).map(([k, v]) => [k, v.shape])), deform.meta);
