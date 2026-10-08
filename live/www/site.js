// The project page's live demos, banner and section bar.
//
// Each case study's block (.live[data-id]) embeds that task's viser page from its own subdomain.
// - It connects only when scrolled into view (a third of it showing), at most MAX_LIVE at once, and
//   lets go once it has been out of view for RELEASE_MS: no hidden streams, no held driver's seat.
// - Until a visitor presses "Drive the robot" a gate lies over it, so the page scrolls past the
//   simulation instead of the simulation's camera eating the wheel or the swipe. On a desktop the
//   gate comes back when the pointer leaves the block; on a phone (or a narrow window) the demo
//   opens full screen with a close button, where the viser page has room for its own controls.
// - The loading bar fills on the frame's load event, and completes when the page in the frame
//   posts {rsw: "ready"} (pages.ready_ping: once this viewer's scene has been sent).
(() => {
  const HOST = /sslip\.io$/.test(location.hostname) ? location.hostname : "95-133-252-186.sslip.io";
  const url = (id) => `https://${id}.${HOST}/`;
  const MAX_LIVE = 2, RELEASE_MS = 20000, SLOW_MS = 20000;
  const small = () => matchMedia("(pointer: coarse), (max-width: 760px)").matches;
  const blocks = [...document.querySelectorAll(".live[data-id]")];
  const state = new Map(blocks.map((b) => [b, { frame: null, ratio: 0, outSince: null, slow: 0, fade: 0 }]));

  const $ = (b, sel) => b.querySelector(sel);
  function progress(b, f, msg) {
    $(b, ".lbar i").style.width = `${Math.round(f * 100)}%`;
    if (msg) $(b, ".lmsg").textContent = msg;
  }
  function liveBlocks() { return blocks.filter((b) => state.get(b).frame); }

  function connect(b) {
    const s = state.get(b);
    if (s.frame) return;
    const f = document.createElement("iframe");
    f.title = `${b.dataset.title}: live simulation`;
    f.allow = "fullscreen; clipboard-write";
    f.addEventListener("load", () => { if (s.frame === f && !b.classList.contains("ready")) progress(b, 0.9, "Sending the scene…"); });
    s.frame = f;
    b.classList.add("connecting");
    b.classList.remove("ready");
    progress(b, 0.15, "Connecting to the simulation…");
    clearTimeout(s.slow);
    s.slow = setTimeout(() => { if (!b.classList.contains("ready")) progress(b, 0.9, "Still starting up; this simulation may be restarting."); }, SLOW_MS);
    f.src = url(b.dataset.id);
    $(b, ".screen").appendChild(f);
    requestAnimationFrame(() => progress(b, 0.45));
    // keep at most MAX_LIVE: let go of the least visible one that nobody is driving
    const extra = liveBlocks().filter((x) => x !== b && !x.classList.contains("active"))
      .sort((x, y) => state.get(x).ratio - state.get(y).ratio);
    while (liveBlocks().length > MAX_LIVE && extra.length) disconnect(extra.shift());
  }

  function disconnect(b) {
    const s = state.get(b);
    if (!s.frame) return;
    release(b);
    s.frame.remove();
    s.frame = null;
    clearTimeout(s.slow);
    b.classList.remove("connecting", "ready");
    progress(b, 0, "");
  }

  function ready(b) {
    const s = state.get(b);
    clearTimeout(s.slow);
    progress(b, 1, "Ready");
    clearTimeout(s.fade);
    s.fade = setTimeout(() => { b.classList.add("ready"); b.classList.remove("connecting"); }, 350);
  }

  addEventListener("message", (e) => {
    if (!e.data || e.data.rsw !== "ready") return;
    for (const b of blocks) if (state.get(b).frame && state.get(b).frame.contentWindow === e.source) ready(b);
  });

  // ---- driving: the gate, full screen on a phone, letting go
  function drive(b) {
    for (const x of blocks) if (x !== b) release(x);
    connect(b);
    b.classList.add("active");
    if (small()) {
      b.classList.add("full");
      document.body.classList.add("locked");
      $(b, ".expand").textContent = "Close";
    }
    $(b, ".gate").setAttribute("aria-hidden", "true");
  }
  function release(b) {
    if (!b.classList.contains("active")) return;
    b.classList.remove("active", "full");
    $(b, ".expand").textContent = "Full screen";
    $(b, ".gate").removeAttribute("aria-hidden");
    if (!blocks.some((x) => x.classList.contains("full"))) document.body.classList.remove("locked");
  }
  for (const b of blocks) {
    const open = $(b, "a.open");
    if (open) open.href = url(b.dataset.id);
    $(b, ".gate").addEventListener("click", () => drive(b));
    $(b, ".close").addEventListener("click", () => release(b));
    $(b, ".expand").addEventListener("click", () => {
      if (b.classList.contains("full")) { release(b); return; }
      drive(b);
      b.classList.add("full");
      document.body.classList.add("locked");
      $(b, ".expand").textContent = "Exit full screen";
    });
    let leave = 0;
    b.addEventListener("mouseleave", () => {
      if (b.classList.contains("full") || small()) return;
      leave = setTimeout(() => release(b), 500);
    });
    b.addEventListener("mouseenter", () => clearTimeout(leave));
  }
  addEventListener("keydown", (e) => { if (e.key === "Escape") blocks.forEach(release); });

  // ---- connect what is in view, let go of what has been out of view a while
  const watch = new IntersectionObserver((entries) => {
    const now = Date.now();
    for (const e of entries) {
      const s = state.get(e.target);
      s.ratio = e.intersectionRatio;
      s.outSince = e.intersectionRatio > 0 ? null : (s.outSince || now);
      if (e.intersectionRatio >= 0.34) connect(e.target);
    }
  }, { threshold: [0, 0.1, 0.34, 0.6, 1] });
  blocks.forEach((b) => watch.observe(b));
  setInterval(() => {
    const now = Date.now();
    for (const b of blocks) {
      const s = state.get(b);
      if (s.frame && s.outSince && now - s.outSince > RELEASE_MS && !b.classList.contains("active")) disconnect(b);
    }
  }, 4000);

  // ---- which simulations answer: the gateway's status route says 204 (up) or 502 (down or restarting)
  async function poll() {
    for (const b of blocks) {
      let ok = false;
      try { ok = (await fetch(`https://${HOST}/status/${b.dataset.id}`, { cache: "no-store" })).ok; } catch (e) { ok = false; }
      for (const dot of document.querySelectorAll(`[data-status="${b.dataset.id}"]`)) {
        dot.classList.toggle("up", ok);
        dot.title = ok ? "Live" : "Starting or restarting";
      }
    }
  }
  poll();
  setInterval(poll, 20000);

  // ---- the section bar marks the section in view (the band a third of the way down the window)
  const links = new Map([...document.querySelectorAll(".topbar ol a[href^='#']")].map((a) => [a.getAttribute("href").slice(1), a]));
  const sections = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  let marked = null;
  function mark() {
    const line = innerHeight / 3;
    let on = sections[0];
    for (const el of sections) if (el.getBoundingClientRect().top <= line) on = el;
    if (on === marked) return;
    marked = on;
    for (const [id, a] of links) a.classList.toggle("on", id === on.id);
    const a = links.get(on.id), bar = a.parentElement.parentElement;  // keep it in sight on a narrow bar
    if (bar.scrollWidth > bar.clientWidth) bar.scrollTo({ left: a.offsetLeft - (bar.clientWidth - a.offsetWidth) / 2, behavior: "smooth" });
  }
  addEventListener("scroll", mark, { passive: true });
  addEventListener("resize", mark);
  mark();
})();
