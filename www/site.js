// The project page: the typed prompt, the five-step stage (four clips and the 3D replay), the citation.
//
// The 3D block (.live[data-id], step 4) plays a recorded rollout of a task (bar, peel, cable) in viser's static
// client (viser/index.html, built by viser-build-client), from rollouts/<task>.viser: no server, the visitor
// orbits and zooms. The picker under it swaps the task. Recordings: demo/tools/record_preview.py --rollout.
// The client runs ?embedded (scripts/viser-embed.patch): it draws no controls of its own, posts
// {viser: "time", t, d} as it plays, and takes {viser: "seek" | "play" | "pause"}: the bar under the stage.
// - It loads when scrolled into view (a third of it showing). On a phone (or any touch screen) it does not
//   load in the page at all: the block shows its clip and the button, and the replay opens full screen.
// - An invisible gate lies over it while the page scrolls, so scrolling down the page never gets caught in the
//   replay. The moment the mouse really moves over it (not the still pointer the page scrolls under), the gate
//   lifts and the replay takes focus: the wheel zooms it and W/A/S/D move its camera, no click needed. A press
//   on the gate is handed through, so the first drag already turns the view. Leaving the block, or the page
//   scrolling, gives the wheel and the keys back to the page. On a phone a tap opens the replay full screen.
(() => {
  // each task's opening camera: from beyond the work, facing the robots, so the arms never block it (the
  // peel from higher: the phone leans back toward the robots, and its glass only reads from above)
  const VIEWS = {
    bar: { position: [0.95, 0.08, 1.2], look_at: [0.6, -0.08, 0.92] },
    peel: { position: [0.69, -0.07, 1.25], look_at: [0.48, -0.09, 0.81] },
    cable: { position: [0.82, 0.22, 1.05], look_at: [0.55, -0.02, 0.79] },
  };

  const url = (id) => `viser/?embedded&playbackPath=../rollouts/${id}.viser` +
    `&initialCameraPosition=${VIEWS[id].position}&initialCameraLookAt=${VIEWS[id].look_at}&initialCameraUp=0,0,1`;
  const small = () => matchMedia("(pointer: coarse), (max-width: 760px)").matches;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (root, sel) => root.querySelector(sel);

  // ---- the 3D block
  const b = $(document, ".live[data-id]");
  const s = { frame: null, ratio: 0, fade: 0, dur: 0, scrubbing: false };
  const bar = $(document, ".tools .scrub");
  const TITLES = { bar: "Wipe the bar", peel: "Peel the film", cable: "Route the cable" };

  function connect() {
    if (s.frame) return;
    const f = document.createElement("iframe");
    f.title = `${b.dataset.title}: 3D replay`;
    f.allow = "fullscreen";
    f.addEventListener("load", () => {
      if (s.frame !== f) return;
      clearTimeout(s.fade);
      s.fade = setTimeout(() => { b.classList.add("ready"); b.classList.remove("connecting"); }, 600);
    });
    s.frame = f;
    b.classList.add("connecting");
    b.classList.remove("ready");
    f.src = url(b.dataset.id);
    $(b, ".screen").appendChild(f);
  }
  function disconnect() {
    if (!s.frame) return;
    release();
    s.frame.remove();
    s.frame = null;
    clearTimeout(s.fade);
    b.classList.remove("connecting", "ready");
    s.dur = 0;
    setBar(0);
  }

  // the bar under the stage: how far the rollout has played; press or drag it to seek
  function setBar(f) {
    $(bar, "i").style.width = `${(f * 100).toFixed(2)}%`;
    bar.setAttribute("aria-valuenow", String(Math.round(f * 100)));
  }
  const tell = (msg) => { if (s.frame && s.frame.contentWindow) s.frame.contentWindow.postMessage(msg, "*"); };
  addEventListener("message", (e) => {
    const m = e.data;
    if (!m || m.viser !== "time" || !s.frame || e.source !== s.frame.contentWindow) return;
    s.dur = m.d;
    if (!s.scrubbing && m.d > 0) setBar(m.t / m.d);
  });
  function seek(e) {
    const r = bar.getBoundingClientRect(), f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    setBar(f);
    tell({ viser: "seek", t: f * s.dur });
  }
  bar.addEventListener("pointerdown", (e) => {
    if (!s.dur) return;
    s.scrubbing = true;
    bar.setPointerCapture(e.pointerId);
    tell({ viser: "pause" });
    seek(e);
  });
  bar.addEventListener("pointermove", (e) => { if (s.scrubbing) seek(e); });
  const endScrub = () => { if (!s.scrubbing) return; s.scrubbing = false; tell({ viser: "play" }); };
  bar.addEventListener("pointerup", endScrub);
  bar.addEventListener("pointercancel", endScrub);

  function drive() {
    connect();
    b.classList.add("active");
    if (small()) {
      b.classList.add("full");
      document.body.classList.add("locked");
    }
    $(b, ".gate").setAttribute("aria-hidden", "true");
    if (s.frame && !small()) s.frame.focus({ preventScroll: true });  // its keys (W/A/S/D) without a click
  }
  function release() {
    if (!b.classList.contains("active")) return;
    b.classList.remove("active", "full");
    $(b, ".gate").removeAttribute("aria-hidden");
    document.body.classList.remove("locked");
    if (s.frame && document.activeElement === s.frame) s.frame.blur();  // the page's keys again
    if (small()) disconnect();  // back to the clip: no 3D page in a phone-wide block
  }
  $(b, ".gate").addEventListener("pointerdown", (e) => {
    const loaded = s.frame && b.classList.contains("ready");
    drive();
    if (small() || !loaded) return;
    // hand the press to the replay's canvas (same origin), so this drag already turns the view
    const r = s.frame.getBoundingClientRect(), doc = s.frame.contentDocument;
    const x = e.clientX - r.left, y = e.clientY - r.top, el = doc && doc.elementFromPoint(x, y);
    if (!el) return;
    const init = { bubbles: true, cancelable: true, clientX: x, clientY: y, button: e.button, buttons: e.buttons,
                   pointerId: e.pointerId, pointerType: e.pointerType, isPrimary: true, view: s.frame.contentWindow };
    el.dispatchEvent(new s.frame.contentWindow.PointerEvent("pointerdown", init));
    el.dispatchEvent(new s.frame.contentWindow.MouseEvent("mousedown", init));
  });
  $(document, ".tools .close").addEventListener("click", release);
  // hover to explore: real mouse movement over the gate lifts it (a scroll under a still pointer makes
  // movement-less mousemoves, and those keep it down)
  let moved = 0;
  $(b, ".gate").addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse" || small() || !s.frame) return;
    moved += Math.abs(e.movementX) + Math.abs(e.movementY);
    if (moved > 6) { moved = 0; drive(); }
  });
  addEventListener("scroll", () => { moved = 0; if (b.classList.contains("active") && !b.classList.contains("full")) release(); }, { passive: true });
  let leave = 0;
  b.addEventListener("mouseleave", () => { if (!b.classList.contains("full") && !small()) leave = setTimeout(release, 500); });
  b.addEventListener("mouseenter", () => clearTimeout(leave));
  addEventListener("keydown", (e) => { if (e.key === "Escape") release(); });

  // the block is only visible on step 4: the observer sees ratio 0 on the other steps, so it loads when step 4 shows
  const watch = new IntersectionObserver(([e]) => {
    s.ratio = e.intersectionRatio;
    if (e.intersectionRatio >= 0.34 && !small()) connect();
  }, { threshold: [0, 0.1, 0.34, 0.6, 1] });
  watch.observe(b);

  // the picker: which task the block shows
  const tabs = [...document.querySelectorAll(".thumbs [data-demo]")];
  function show(id) {
    if (!TITLES[id]) return;
    if (b.dataset.id !== id) {
      disconnect();
      b.dataset.id = id;
      b.dataset.title = TITLES[id];
      const v = $(b, ".screen > video");
      v.poster = `video/${id}.jpg`;
      v.src = `video/${id}.mp4`;
      v.play().catch(() => {});
      if (s.ratio >= 0.34 && !small()) connect();
    }
    for (const t of tabs) t.setAttribute("aria-selected", String(t.dataset.demo === id));
  }
  for (const t of tabs) t.addEventListener("click", () => { show(t.dataset.demo); history.replaceState(null, "", `#demo=${t.dataset.demo}`); });

  // ---- the prompt, typed out once it scrolls into view
  const prompt = $(document, "#prompt"), typed = $(document, "#typed"), src = $(document, "#prompt-src").content;
  function type() {
    const nodes = [...src.childNodes];
    if (reduced) { for (const n of nodes) typed.appendChild(n.cloneNode(true)); prompt.classList.add("done"); hookAt(); return; }
    let i = 0, j = 0;
    const tick = () => {
      if (i >= nodes.length) { prompt.classList.add("done"); hookAt(); return; }
      const n = nodes[i];
      if (n.nodeType === 3) {  // text: one character at a time
        if (j === 0) typed.appendChild(document.createTextNode(""));
        typed.lastChild.data += n.data[j++];
        if (j >= n.data.length) { i++; j = 0; }
        setTimeout(tick, 28 + Math.random() * 30);
      } else {  // a link or the @-chip: typed inside its own element
        if (j === 0) { const c = n.cloneNode(false); c.textContent = ""; typed.appendChild(c); }
        typed.lastChild.textContent += n.textContent[j++];
        if (j >= n.textContent.length) { i++; j = 0; }
        setTimeout(tick, 18);
      }
    };
    tick();
  }
  // the @-mention in the prompt and the chip on the input clip are the same thing: hover one, both light up
  function hookAt() {
    const ats = [...document.querySelectorAll(".at[data-ref='clip']")];
    for (const a of ats) {
      a.addEventListener("mouseenter", () => ats.forEach((x) => x.classList.add("hot")));
      a.addEventListener("mouseleave", () => ats.forEach((x) => x.classList.remove("hot")));
    }
  }

  // ---- the stage: five steps, one showing. The section is tall and its content sticks, so scrolling down
  // walks through the steps; the rail and the Next button scroll the page to a step's place.
  const flow = $(document, "#workflow"), stage = $(document, "#stage");
  const panels = [...stage.querySelectorAll(".panel")], steps = [...document.querySelectorAll(".rail [data-step]")];
  const next = $(document, "#next");
  const NEXT = { 1: "Next: the coding agent at work", 2: "Next: the simulation it produced", 3: "Next: the simulation in 3D", 4: "Next: Sim2Real" };
  const N = steps.length;
  let step = 0, typedOnce = false;
  const vh = () => innerHeight;
  const span = () => flow.offsetHeight - vh();  // the scroll distance the section holds
  const where = (n) => flow.offsetTop + span() * ((n - 1) / N + 0.08);  // scroll position that shows step n
  function render(n) {
    n = Math.max(1, Math.min(N, n));
    if (n === step) return;
    step = n;
    flow.dataset.step = step;
    for (const p of panels) {
      const on = +p.dataset.step === step;
      p.classList.toggle("on", on);
      for (const v of p.querySelectorAll("video")) { if (on) v.play().catch(() => {}); else v.pause(); }
    }
    for (const t of steps) {
      const k = +t.dataset.step;
      if (k === step) t.setAttribute("aria-current", "step"); else t.removeAttribute("aria-current");
      t.parentElement.classList.toggle("done", k < step);
    }
    if (step < N) $(next, "span").textContent = NEXT[step];
    if (step !== 4) release();
    if (step === 1 && !typedOnce) { typedOnce = true; type(); }
  }
  function goTo(n) { scrollTo({ top: where(n), behavior: reduced ? "auto" : "smooth" }); }
  function onScroll() {
    const y = scrollY - flow.offsetTop;
    if (y < -vh() * 0.5 || y > span() + vh()) { for (const v of stage.querySelectorAll("video")) v.pause(); return; }  // the section is off screen
    render(Math.floor(Math.max(0, Math.min(0.999, y / span())) * N) + 1);
  }
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", onScroll);
  for (const t of steps) t.addEventListener("click", () => goTo(+t.dataset.step));
  next.addEventListener("click", () => goTo(step + 1));
  addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, iframe") || b.classList.contains("active")) return;
    if (e.key === "ArrowRight") goTo(step + 1);
    if (e.key === "ArrowLeft") goTo(step - 1);
  });
  function fromHash() {
    let m;
    if ((m = /^#demo=(\w+)$/.exec(location.hash))) { show(m[1]); scrollTo({ top: where(4), behavior: "auto" }); }
    else if ((m = /^#step=([1-5])$/.exec(location.hash))) scrollTo({ top: where(+m[1]), behavior: "auto" });
    onScroll();
  }
  addEventListener("hashchange", fromHash);
  $(document, ".hero .down").addEventListener("click", (e) => { e.preventDefault(); goTo(1); });
  for (const a of document.querySelectorAll('a[href^="#demo="]')) a.addEventListener("click", (e) => { e.preventDefault(); history.replaceState(null, "", a.getAttribute("href")); show("bar"); goTo(4); });
  fromHash();
  // step 3's slider: the range input's value is the clip line
  const cmp = $(stage, ".panel.cmp"), slide = $(cmp, ".slide");
  slide.addEventListener("input", () => { cmp.style.setProperty("--x", slide.value + "%"); });

  // ---- the gallery clip plays only in view; the citation's Copy button
  const playing = new IntersectionObserver((entries) => {
    for (const e of entries) { if (e.intersectionRatio >= 0.4) e.target.play().catch(() => {}); else e.target.pause(); }
  }, { threshold: [0, 0.4] });
  for (const v of document.querySelectorAll(".gallery video")) playing.observe(v);
  const copy = $(document, ".cite .copy");
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($(document, "#bibtex").textContent); copy.textContent = "Copied"; }
    catch (e) { copy.textContent = "Select and copy"; }
    setTimeout(() => { copy.textContent = "Copy"; }, 1800);
  });
})();
