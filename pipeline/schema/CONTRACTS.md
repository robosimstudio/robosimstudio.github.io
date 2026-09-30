# Data contracts: pipeline (Python, `rsw`) → site (TypeScript)

Every demo is a directory `demos/<id>/` in the built site. Every file name except
`manifest.json` carries a content hash (`scene.3fa9c1d2.glb`), because GitHub Pages
serves everything with `max-age=600`. World frame: Newton's, **Z up**, metres.
Quaternions are **xyzw**. Poses are 7-vectors `[px,py,pz,qx,qy,qz,qw]`.

## 1. RSSB container (`*.rssb.gz`)

A whole file is gzip-compressed (Pages does not gzip binary; the page decompresses
with `DecompressionStream('gzip')`, fflate as fallback). Inside:

```
offset 0   'RSSB'            4 bytes ASCII
offset 4   version = 1       u32 LE
offset 8   json_len          u32 LE
offset 12  JSON              json_len bytes UTF-8, then 0x20 padding to a 16-byte boundary
D = align16(12 + json_len)   start of the data section
```

JSON: `{"arrays": {name: Array}, "meta": {...}}`, `Array = {"dtype", "shape", "offset", "bytes", "enc", "lo"?, "hi"?}`

* `offset` is relative to `D`, a multiple of 16. `bytes` is the stored size.
* `dtype` ∈ `f64 f32 f16 i32 u32 i16 u16 i8 u8`. Little endian, row-major.
* `enc = "raw"`: the bytes are the typed array.
* `enc = "q16d"` (logical dtype f32, shape `[N, C]`; a 1-D array is `C = 1`): each value is
  quantized to u16 over `[lo[c], hi[c]]`, delta-encoded along axis 0
  (`d[n,c] = (v[n,c] - v[n-1,c]) mod 65536`, `v[-1,c] = 0`), and stored as **byte planes**:
  the `N*C` low bytes, then the `N*C` high bytes. Decode: `v = cumsum(d) mod 65536` per
  channel, `x = lo[c] + (hi[c] - lo[c]) * v / 65535`.

## 2. `kinematics.<h>.json` (free-drag demos)

```jsonc
{
  "q_index": [3, 4, ...],            // which Newton joint_q entries the DB stores per frame, in order
  "q_full": [ ... ],                 // the full joint_q at the canonical frame (entries not in q_index keep these)
  "joints": [                        // topological order; only joints whose child is a robot body
    {"type": "revolute"|"prismatic"|"fixed"|"free"|"ball", "parent": -1 | body, "child": body,
     "X_p": [7], "X_c": [7], "axis": [3], "q_start": int}
  ],
  "bodies": [{"index": int, "label": str}],       // bodies FK produces
  "roots": {"<body>": [7]},                        // world poses of parents FK does not produce (-1 → identity)
  "tcp": {"<arm>": {"body": int, "offset": [7]}},  // tool centre = pose[body] ∘ offset
  "tests": [{"q": [len(q_index)], "poses": {"<body>": [7]}}]
}
```

FK (Newton semantics): for each joint in order,
`pose[child] = pose[parent] ∘ X_p ∘ X_j(q) ∘ X_c⁻¹`, with
`X_j` = rotation about `axis` by `q[q_start]` (revolute), translation `axis * q[q_start]`
(prismatic), identity (fixed), `q[q_start..+7]` as a pose (free), `q[q_start..+4]` xyzw (ball).

## 3. Free-drag database `db<k>.<h>.rssb.gz`

Chunk 0 alone is enough to start (it holds the attract take and a farthest-point subset of
the rest); later chunks append. Frame indices are global across chunks (`meta.frame0`).

| name | enc/dtype | shape | meaning |
|---|---|---|---|
| `takes` | raw u32 | [T,3] | `(start, len, kind)`; kind 0 = visitor, 1 = attract. Frames never interpolate or continue across a take |
| `q` | q16d | [N, nq] | robot coordinates, order `kinematics.q_index` |
| `hand` | raw f32 | [N,3] | tool centre position (towel: midpoint of the two) |
| `setpoint` | raw f32 | [N,3] | the servo's own setpoint, to seed the browser's OscSetpoint on handover |
| `cursor` | raw f32 | [N,2] | the virtual visitor's cursor `(a,b)` in drag coordinates |
| `deform` | q16d | [N,K] | PCA coefficients (or `[N, 3V]` raw positions when `deform.meta.mode == "raw"`) |
| `feat` | raw f16 | [N,D] | standardized features `(x - μ)/σ_group`, group layout in the manifest |
| `ink` | raw u8 | [N,cells] | wipe: removal rate, `rate = r0 * (exp(v/255 * ln(1 + rmax/r0)) - 1)` [1/s] |
| `ink_state` | raw u8 | [N_attract, cells] | wipe: ink intensity × 255 for attract frames (takeover) |
| `broken` | raw u16 | [N] | peel: bonds broken |
| `front` | raw u8 | [N,C] | peel: per-column peel front (monotone keys) |
| `still` | raw i32 | [N] | index into `manifest.stills.files`, or -1 |

Feature groups (manifest `features.groups`, in this column order):
`fd` 9 = hand displacement at +6/+12/+20 control steps; `hp` 3 = hand position; `hv` 3 =
hand velocity [m/s]; `dm` m = first PCA coefficients; `dv` m = their velocities; `pg` 1 =
peel progress (peel only). Query-time weight per group in the manifest; the stored values are
standardized only.

## 4. `deform.<h>.rssb.gz`

Arrays: `tris` u32 [M,3], `uv` f32 [V,2], `mean` f32 [V,3], `basis` f32 [K,V,3] (absent in raw
mode). `meta`: `{"name", "mode": "pca"|"raw", "frame": "world"|"tcp"|"held"|"mid", "material": {...}}`.
A point is `frame_pose ∘ (mean + Σ c_k basis_k)`. `frame = "tcp"` uses the demo's first arm TCP
from FK; `"world"` is identity.

## 5. `manifest.json`

```jsonc
{
  "id": "wipe", "kind": "free" | "track", "fps": 30, "frames": N,
  "camera": {"width": 1920, "height": 1080, "fx", "fy", "cx", "cy", "near", "far",
             "eye": [3], "look_at": [3], "up": [3], "cam_to_world_cv": [16]},  // column-major 4x4, +x right +y down +z forward
  "roi": [x0, y0, w, h],                       // RTX px: where the action is (phone portrait crop)
  "files": {"glb", "kin"?, "deform"?, "db"?: [..], "state"?, "paths"?, "plate", "video", "tile", "poster",
            "stills"?: [..], "ink_clean"?: [..], "inkmap"?, "attract"?},
  "nodes": {"body:<i>": i},                    // GLB pose nodes the page drives
  "moving": [body...], "static": [body...],
  "look": {"exposure": 1.0, "tonemap": "linear", "key": {"pos": [3], "target": [3], "intensity", "color": [3]},
           "env_intensity", "ao": {...}},
  "ctrl": {"control_dt", "pos_speed", "pos_lead", "rot_speed", "rot_lead", "pre_limit"?: m/s,
           "plant": {"alpha", "beta"}, "hand_quat": [4]},
  "drag": {"origin": [3], "A": [3], "B": [3], "a": [lo, hi], "b": [lo, hi], "hover": [a, b],
           "coast_tau": 0.3, "v_max": 1.5},    // world = origin + a*A + b*B, clamped in (a, b)
  "features": {"groups": [{"name", "dims", "w"}], "horizons": [6, 12, 20]},
  "match": {...}, "settle": {...},
  "stills": {"files": [..], "frames": [db frame per still]},
  "attract": {"video", "frames": [db frame per video frame], "cursor": [[a, b]...]},  // or a file
  "ink"?: {"cells": [[u, angle]...], "pitch", "radius", "reach", "taint": [3], "clear": [3],
           "bbox": [x0, y0, w, h], "r0", "rmax"},
  "track"?: {"handles": [{"name", "body", "offset": [7]}]}
}
```

## 6. Track demos

`state.<h>.rssb.gz`: `bodies` u32 [nb] (which Newton bodies), `body_q` q16d [N, nb*7] (xyzw,
renormalize after decode), optional `particles` q16d [N, 3P] + `tris`.
`paths.<h>.rssb.gz`: `ee_world` f32 [N,H,3], `ee_px` f32 [N,H,2] (RTX px), `active` u8 [N,H],
`arc` f32 [N,H].
The rollout video has exactly N frames at 30 fps CFR, pts from 0: frame i is shown for
`t ∈ [i/30, (i+1)/30)`; seek to `(i + 0.5)/30`.
