"""``rsw db``: recorded takes -> the free-drag database the browser matches over.

* 60 Hz steps -> 30 Hz frames (even steps). Hand velocity and future displacement
  are measured at 60 Hz first; per-frame ink rates are the mean of the two steps
  (the browser applies ``rate / 30`` per frame, the same total removal).
* The deformable is stored in its natural frame (the sponge in the TCP frame) and
  PCA-compressed: the smallest k whose max per-point error is <= 2 mm on held-out
  frames (99.9th percentile reported).
* Features (CONTRACTS.md §3): fd 9 | hp 3 | hv 3 | dm m | dv m [| pg 1], standardized per group.
* The arm's response to its setpoint is fitted as a 2nd-order lag for the browser's
  predictor: x+ = x + alpha (s - x) + beta (x - x-).
* The still set: settled frames (hand < 3 mm/s and deformable < 5 mm/s for 0.5 s),
  spread by farthest-point sampling over hand position and deformable descriptor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths, rssb
from .record import load

FD_STEPS = (6, 12, 20)  # control steps at 60 Hz: +0.1/+0.2/+0.33 s
DT60 = 1.0 / 60.0


def _quat_matrix_batch(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def take_dirs(demo: str, rounds) -> list[Path]:
    out = []
    for r in rounds:
        out += sorted(p for p in (paths.DATA / demo / "takes" / r).glob("s*/take_*") if (p / "meta.json").exists())
    return out


def _usable(take: Path) -> tuple[dict, int]:
    """(meta, usable 60 Hz steps): a gated take is cut 2 s before its failure."""
    meta = json.loads((take / "meta.json").read_text())
    n = int(meta["steps"])
    if meta.get("gate"):
        n = max(0, n - 120)
    return meta, n


MIN_RUN_STEPS = 4 * 60
"""A run cut out of a take shorter than this is dropped: the matcher keeps 1 s of lead-in and 0.33 s
of runway off every take, so a shorter run offers it almost nothing."""


def fit_plant(hand: list[np.ndarray], setp: list[np.ndarray]) -> dict:
    rows, rhs = [], []
    for x, s in zip(hand, setp):
        if len(x) < 3:
            continue
        a = (s[1:-1] - x[1:-1]).reshape(-1)
        b = (x[1:-1] - x[:-2]).reshape(-1)
        rows.append(np.stack([a, b], 1))
        rhs.append((x[2:] - x[1:-1]).reshape(-1))
    A, y = np.concatenate(rows), np.concatenate(rhs)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, beta = float(sol[0]), float(sol[1])
    if beta >= 0.99:
        # The browser iterates this 20 steps ahead: beta >= 1 grows the velocity (1.11**20 ~ 8x).
        # The whip's joint lead (q + tau dq) puts its hand ahead of the setpoint, which a free fit
        # reads as beta 1.11. Keep the best stable model: beta searched below 0.99, alpha refit.
        best = None
        for bb in np.linspace(0.0, 0.99, 100):
            r = y - bb * A[:, 1]
            aa = float(A[:, 0] @ r / max(1e-12, float(A[:, 0] @ A[:, 0])))
            e = float(np.mean((r - aa * A[:, 0]) ** 2))
            if best is None or e < best[0]:
                best = (e, aa, float(bb))
        alpha, beta = best[1], best[2]
    resid = y - A @ np.array([alpha, beta])
    return {"alpha": alpha, "beta": beta, "rms_step_mm": float(np.sqrt(np.mean(resid**2)) * 1e3)}


def pca_fit(X: np.ndarray, p999_max: float = 0.0015, tail_mm: float = 0.004, tail_frac: float = 0.005,
            k_max: int = 128, seed: int = 0) -> dict:
    """PCA on [N, V*3] rows of the DRAWN particles.

    A max-error bound is not reachable for SuperDex foam (measured on the wipe:
    9 mm at k=64, contact wrinkles are local), so k is the smallest whose
    held-out 99.9th percentile point error is <= 1.5 mm AND whose frames with any
    point off by more than 4 mm are <= 0.5 %."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    fit, val = X[idx[: min(len(X), 40000)]], X[idx[-min(len(X) // 5, 8000):]]
    mean = fit.mean(axis=0)
    _, s, vt = np.linalg.svd((fit - mean).astype(np.float64), full_matrices=False)
    V = X.shape[1] // 3
    chosen, report = min(k_max, len(vt)), []
    for k in (8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256):
        if k > min(k_max, len(vt)):
            break
        B = vt[:k]
        rec = (val - mean) @ B.T @ B + mean
        e = np.linalg.norm((rec - val).reshape(len(val), V, 3), axis=2)
        mx, p999, tail = float(e.max()), float(np.percentile(e, 99.9)), float(np.mean(e.max(axis=1) > tail_mm))
        report.append((k, mx, p999, tail))
        if p999 <= p999_max and tail <= tail_frac:
            chosen = k
            break
    B = vt[:chosen].astype(np.float32)
    return {"mean": mean.astype(np.float32), "basis": B, "k": chosen, "curve": report,
            "energy": float((s[:chosen] ** 2).sum() / (s**2).sum())}


def fps(points: np.ndarray, n: int, seed_index: int = 0) -> tuple[np.ndarray, float]:
    """Greedy farthest-point sampling; returns indices and the covering radius."""
    n = min(n, len(points))
    chosen = [seed_index]
    d = np.linalg.norm(points - points[seed_index], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return np.asarray(chosen), float(d.max())


def build(demo: str, rounds=("r1",), attract_round: str = "attract", n_stills: int = 450,
          m_desc: int = 4, takes_per_chunk: int = 4) -> dict:
    from demos import load as load_spec

    spec = load_spec(demo)
    exp = paths.DATA / demo / "export"
    kin = json.loads((exp / "kinematics.json").read_text())
    q_index = np.asarray(kin["q_index"], dtype=np.int64)
    out = paths.demo_dir(demo, "db")

    # ---------------------------------------------------------------- takes
    visitor = take_dirs(demo, rounds)
    attract = take_dirs(demo, (attract_round,))
    chosen_attract = spec.pick_attract(attract) if attract and hasattr(spec, "pick_attract") else attract[:1]
    order = [(t, 1) for t in chosen_attract] + [(t, 0) for t in visitor]
    per_take = []
    for t, kind in order:
        meta, n = _usable(t)
        if n < 30 * 60 and kind == 0:
            continue
        d = load(t)
        d = {k: v[:n] for k, v in d.items()}
        if hasattr(spec, "setpoint_from"):
            # a take that recorded its goal, not the servo law's output (the towel): replay the law,
            # over the whole take before any cut
            d["setpoint"] = spec.setpoint_from(d)
        per_take.append((t, kind, d))
    print(f"[db] {demo}: {len(per_take)} takes ({sum(k for _, k, _ in per_take)} attract)")
    step0 = [0] * len(per_take)  # each (sub-)take's first 60 Hz step in its recording
    if hasattr(spec, "keep_mask"):
        # A spec may cut stretches out of its visitor takes: the towel's asked for places out of the
        # arm's reach, where the hand sat blocked with a full lead. A plant fitted on those reads
        # alpha 0.0014 (0.0069 without), and a matcher that lands there stalls. The attract take
        # stays whole: its video frames are its database frames.
        split, s0 = [], []
        for t, kind, d in per_take:
            if kind == 1:
                split.append((t, kind, d))
                s0.append(0)
                continue
            ok = spec.keep_mask(d, json.loads((t.parent / "static.json").read_text()))
            edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= MIN_RUN_STEPS:
                    split.append((t, kind, {k: v[a:b] for k, v in d.items()}))
                    s0.append(int(a))
        kept = sum(len(d["tcp"]) for _, k, d in split if k == 0) / max(1, sum(len(d["tcp"]) for _, k, d in per_take if k == 0))
        print(f"[db] kept {100 * kept:.0f}% of the visitor steps, in {len(split)} runs")
        per_take, step0 = split, s0

    # ---------------------------------------------------- deformable, local frame
    frame_of = getattr(spec, "deform_frame", "tcp")
    name = getattr(spec, "deform_name", "sponge")
    kind_of = getattr(spec, "deform_kind", "mesh")
    topo = np.load(exp / "deform_topology.npz") if kind_of == "mesh" else None
    if kind_of == "mesh":
        tris_all = topo[f"{name}.tris"].astype(np.int64)
        surf = np.unique(tris_all.reshape(-1))  # only the particles a triangle draws
        remap = np.full(int(tris_all.max()) + 1, -1, dtype=np.int64)
        remap[surf] = np.arange(len(surf))
        tris_s = remap[tris_all]
    else:  # a centreline (the whip's capsule chain), drawn as a tube
        surf = np.arange(per_take[0][2]["deform"].shape[1])
        tris_s = None
    locals_, worlds = [], []
    for _, _, d in per_take:
        P = d["deform"][:, surf].astype(np.float64)  # [n, Vs, 3]
        if frame_of == "tcp":
            R = _quat_matrix_batch(d["tcp"][:, 3:7].astype(np.float64))
            t = d["tcp"][:, :3].astype(np.float64)
            L = np.einsum("nji,nvj->nvi", R, P - t[:, None, :])  # R^T (p - t)
        else:
            L = P
        locals_.append(L[::2].reshape(len(L[::2]), -1).astype(np.float32))
        worlds.append(P[::2])
    Xall = np.concatenate(locals_)
    pca = pca_fit(Xall, k_max=int(getattr(spec, "pca_k_max", 128)))
    k = pca["k"]
    print(f"[db] PCA k={k} energy={pca['energy']:.6f} last={pca['curve'][-1]}")

    # ------------------------------------------------------------ per frame
    cols: dict[str, list[np.ndarray]] = {n: [] for n in ("q", "hand", "setpoint", "cursor", "deform", "fd", "hv", "ink", "kind",
                                                         "broken", "rows")}
    takes_tab, hands60, setp60 = [], [], []
    settled_mask = []
    start = 0
    for (t, kind, d), L30 in zip(per_take, locals_):
        hand = d["tcp"][:, :3].astype(np.float64)
        hands60.append(hand)
        setp60.append(d["setpoint"][:, :3].astype(np.float64))
        n60 = len(hand)
        hv = np.gradient(hand, DT60, axis=0)
        fd = np.concatenate([hand[np.minimum(np.arange(n60) + s, n60 - 1)] - hand for s in FD_STEPS], axis=1)
        sel = np.arange(0, n60, 2)
        C = (L30 - pca["mean"]) @ pca["basis"].T
        cols["q"].append(d["joint_q"][sel][:, q_index].astype(np.float32))
        cols["hand"].append(hand[sel].astype(np.float32))
        cols["setpoint"].append(d["setpoint"][sel, :3].astype(np.float32))
        cols["cursor"].append(d["cursor"][sel].astype(np.float32))
        cols["deform"].append(C.astype(np.float32))
        cols["fd"].append(fd[sel].astype(np.float32))
        cols["hv"].append(hv[sel].astype(np.float32))
        if "bond_active" in d:  # the peel's monotone keys
            cols["broken"].append((~d["bond_active"][sel].astype(bool)).sum(axis=1).astype(np.uint16))
            cols["rows"].append(np.asarray(d["rows"][sel], dtype=np.float32))
        if "ink_rate" in d:
            ir = d["ink_rate"].astype(np.float32)
            ir2 = np.pad(ir, ((0, len(ir) % 2), (0, 0)))
            cols["ink"].append(0.5 * (ir2[0::2] + ir2[1::2])[: len(sel)])
        # settledness at 30 Hz: hand speed and max particle speed, over the last 0.5 s
        W = d["deform"][sel].astype(np.float64)
        pv = np.linalg.norm(np.diff(W, axis=0), axis=2).max(axis=1) * 30.0
        pv = np.concatenate([[pv[0] if len(pv) else 0.0], pv])
        hs = np.linalg.norm(hv[sel], axis=1)
        calm = (hs < 0.003) & (pv < float(getattr(spec, "settle_deform_speed", 0.005)))
        run = np.zeros(len(calm), dtype=np.int64)
        for i in range(len(calm)):
            run[i] = run[i - 1] + 1 if (calm[i] and i) else int(calm[i])
        settled_mask.append(run >= 15)
        takes_tab.append((start, len(sel), kind))
        start += len(sel)
    arr = {n: np.concatenate(v) if v else None for n, v in cols.items()}
    N = start
    settled = np.concatenate(settled_mask)

    # -------------------------------------------------------------- features
    dm = arr["deform"][:, :m_desc]
    dv = np.zeros_like(dm)
    for s, n, _ in takes_tab:
        dv[s : s + n] = np.gradient(dm[s : s + n], 1.0 / 30.0, axis=0) if n > 1 else 0.0
    groups = [("fd", arr["fd"]), ("hp", arr["hand"]), ("hv", arr["hv"]), ("dm", dm), ("dv", dv)]
    if arr["broken"] is not None:  # peel progress: what fraction of the bonds are broken
        n_bonds = float(json.loads((per_take[0][0].parent / "static.json").read_text()).get("n_bonds", 1))
        groups.append(("pg", (arr["broken"].astype(np.float32) / n_bonds)[:, None]))
    feats, gmeta = [], []
    default_w = {"fd": 1.0, "hp": 1.0, "hv": 0.6, "dm": 0.5, "dv": 0.25, "pg": 3.0}
    for gname, x in groups:
        mu = x.mean(axis=0)
        sig = float(np.sqrt(np.mean(np.var(x, axis=0)))) or 1.0
        feats.append((x - mu) / sig)
        gmeta.append({"name": gname, "dims": int(x.shape[1]), "mu": mu.tolist(), "sigma": sig, "w": default_w[gname]})
    F = np.concatenate(feats, axis=1).astype(np.float16)

    # ---------------------------------------------------------------- plant
    plant = fit_plant(hands60, setp60)
    print(f"[db] plant alpha={plant['alpha']:.4f} beta={plant['beta']:.4f} rms={plant['rms_step_mm']:.3f} mm/step")

    # ---------------------------------------------------------------- stills
    cand = np.flatnonzero(settled)
    att_frames = [i for s, n, kd in takes_tab if kd == 1 for i in range(s, s + n)]
    if len(cand):
        hp = arr["hand"][cand]
        pts = np.concatenate([(hp - hp.mean(0)) / 0.01, (dm[cand] - dm[cand].mean(0)) / (dm[cand].std(0) + 1e-9)], axis=1)
        pick, radius = fps(pts, n_stills)
        stills = np.sort(cand[pick])
        cover_mm = float(np.max(np.min(np.linalg.norm(hp[:, None, :] - arr["hand"][stills][None, :, :], axis=2), axis=1)) * 1e3) \
            if len(cand) < 60000 else radius
    else:
        stills, cover_mm = np.zeros(0, dtype=np.int64), float("nan")
    still_col = np.full(N, -1, dtype=np.int32)
    still_col[stills] = np.arange(len(stills), dtype=np.int32)
    print(f"[db] settled frames {len(cand)} / {N}; stills {len(stills)}; hand covering radius {cover_mm:.1f} mm")

    # ------------------------------------------------------------- ink code
    ink_meta = None
    if arr["ink"] is not None:
        rate = arr["ink"]
        rmax = float(np.percentile(rate[rate > 0], 99.99)) if (rate > 0).any() else 1.0
        r0 = 0.01
        v = np.log1p(np.clip(rate, 0, rmax) / r0) / np.log1p(rmax / r0) * 255.0
        ink_u8 = np.rint(v).astype(np.uint8)
        ink_meta = {"r0": r0, "rmax": rmax}
        # attract takes: the recorded intensity per frame (browser takeover)
        ink_state = []
        for s, n, kd in takes_tab:
            if kd != 1:
                continue
            inten = np.ones(rate.shape[1])
            rows = []
            for i in range(s, s + n):
                rows.append(inten.copy())
                live = inten > 0
                inten[live] = np.maximum(0.0, inten[live] - rate[i][live] / 30.0)
            ink_state.append(np.asarray(rows))
        ink_state = np.rint(np.concatenate(ink_state) * 255).astype(np.uint8) if ink_state else None

    # ------------------------------------------------------------- chunks
    groups_of_takes = [[i for i, (_, _, kd) in enumerate(takes_tab) if kd == 1]]
    rest = [i for i, (_, _, kd) in enumerate(takes_tab) if kd == 0]
    groups_of_takes[0] += rest[:2]
    rest = rest[2:]
    while rest:
        groups_of_takes.append(rest[:takes_per_chunk])
        rest = rest[takes_per_chunk:]
    files = []
    for ci, tids in enumerate(groups_of_takes):
        rows = np.concatenate([np.arange(takes_tab[t][0], takes_tab[t][0] + takes_tab[t][1]) for t in tids])
        # CONTRACTS.md §3: take starts are GLOBAL frame indices and a chunk tiles
        # [frame0, frame0 + N) exactly, appended in frame0 order.
        assert np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))), "chunk frames must be contiguous"
        sub_tab = [(takes_tab[t][0], takes_tab[t][1], takes_tab[t][2]) for t in tids]
        chunk = {
            "takes": np.asarray(sub_tab, dtype=np.uint32),
            "q": ("q16d", arr["q"][rows]),
            "hand": arr["hand"][rows],
            "setpoint": arr["setpoint"][rows],
            "cursor": arr["cursor"][rows],
            "deform": ("q16d", arr["deform"][rows]),
            "feat": F[rows],
            "still": still_col[rows],
        }
        if arr["broken"] is not None:
            chunk["broken"] = arr["broken"][rows]
            chunk["front"] = np.rint(np.clip(arr["rows"][rows], 0, 1) * 255).astype(np.uint8)
        if ink_meta is not None:
            chunk["ink"] = ink_u8[rows]
            if ci == 0 and ink_state is not None:
                chunk["ink_state"] = ink_state
        path = out / f"db{ci}.rssb.gz"
        info = rssb.write(path, chunk, {"chunk": ci, "frames": int(len(rows)), "frame0": int(rows[0])})
        files.append({"file": path.name, "frames": int(len(rows)), "gz_bytes": info["gz_bytes"], "q16d_err": info["q16d_max_err"]})
        print(f"[db] chunk {ci}: {len(rows)} frames, {info['gz_bytes'] / 1e6:.2f} MB gz")

    # ------------------------------------------------------------- deform file
    if kind_of == "mesh":
        fin = json.loads(str(topo[f"{name}.finish"]))
        rssb.write(out / "deform.rssb.gz", {
            "tris": tris_s.astype(np.uint32),
            "uv": topo[f"{name}.uv"][surf].astype(np.float32),
            "particles": surf.astype(np.uint32),
            "mean": pca["mean"].reshape(-1, 3),
            "basis": pca["basis"].reshape(k, -1, 3),
        }, {"name": name, "mode": "pca", "kind": "mesh", "frame": frame_of, "arm": getattr(spec, "deform_arm", "right"),
            "finish": fin, "color": topo[f"{name}.color"].tolist()})
    else:
        rssb.write(out / "deform.rssb.gz", {
            "mean": pca["mean"].reshape(-1, 3),
            "basis": pca["basis"].reshape(k, -1, 3),
        }, {"name": name, "mode": "pca", "kind": "tube", "frame": frame_of, "arm": getattr(spec, "deform_arm", "right"),
            "radius": float(json.loads((per_take[0][0].parent / "static.json").read_text()).get("radius", 0.0025)),
            "color": list(getattr(spec, "tube_color", (0.015, 0.015, 0.017))),
            "finish": {"roughness": 0.45, "metallic": 0.0, "opacity": 1.0}})

    summary = {
        "demo": demo, "frames": int(N), "takes": [(int(s), int(n), int(kd)) for s, n, kd in takes_tab],
        "pca": {"k": k, "energy": pca["energy"], "curve": pca["curve"][-3:]},
        "features": {"groups": gmeta, "horizons": list(FD_STEPS), "m": m_desc},
        "plant": plant, "stills": stills.tolist(), "still_cover_mm": cover_mm, "ink": ink_meta,
        "chunks": files, "attract_takes": [str(t) for t in chosen_attract],
        "take_paths": [str(t) for t, _, _ in per_take],
        "take_step0": step0,
        "attract_frames": [int(att_frames[0]), int(att_frames[-1]) + 1] if att_frames else None,
    }
    (out / "db.json").write_text(json.dumps(summary, indent=1))
    return summary
