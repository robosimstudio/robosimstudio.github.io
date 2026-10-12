// The project page: the typed prompt, the five-step stage (four clips and the live demo), the citation.
//
// The live block (.live[data-id], step 4) embeds a task's viser page from its own subdomain; the picker under it
// swaps which task (bar, peel, cable) the block shows.
// - It connects when scrolled into view (a third of it showing) and lets go once it has been out of
//   view for RELEASE_MS: no hidden streams, no held driver's seat. On a phone (or any touch screen)
//   it does not connect in the page at all: the block shows its clip and the Drive button, and the
//   live page loads only when Drive opens it full screen, and goes again on Close.
// - Until a visitor presses "Drive the robot" a gate lies over it, so the page scrolls past the
//   simulation instead of the simulation's camera eating the wheel or the swipe. On a desktop the
//   gate comes back when the pointer leaves the block; on a phone the demo opens full screen.
// - The loading bar fills on the frame's load event, and completes when the page in the frame
//   posts {rsw: "ready"} (pages.ready_ping: once this viewer's scene has been sent).
(() => {
  const HOST = /sslip\.io$/.test(location.hostname) ? location.hostname : "95-133-252-186.sslip.io";
  const url = (id) => `https://${id}.${HOST}/`;
  const RELEASE_MS = 20000, SLOW_MS = 20000;
  const small = () => matchMedia("(pointer: coarse), (max-width: 760px)").matches;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (root, sel) => root.querySelector(sel);

  // ---- the live block
  const b = $(document, ".live[data-id]");
  const s = { frame: null, ratio: 0, outSince: null, slow: 0, fade: 0 };
  const TITLES = { bar: "Wipe the bar", peel: "Peel the film", cable: "Route the cable" };

  function progress(f, msg) {
    $(b, ".lbar i").style.width = `${Math.round(f * 100)}%`;
    if (msg) $(b, ".lmsg").textContent = msg;
  }
  function connect() {
    if (s.frame) return;
    const f = document.createElement("iframe");
    f.title = `${b.dataset.title}: live simulation`;
    f.allow = "fullscreen; clipboard-write";
    f.addEventListener("load", () => { if (s.frame === f && !b.classList.contains("ready")) progress(0.9, "Sending the scene…"); });
    s.frame = f;
    b.classList.add("connecting");
    b.classList.remove("ready");
    progress(0.15, "Connecting to the simulation…");
    clearTimeout(s.slow);
    s.slow = setTimeout(() => { if (!b.classList.contains("ready")) progress(0.9, "Still starting up; this simulation may be restarting."); }, SLOW_MS);
    f.src = url(b.dataset.id);
    $(b, ".screen").appendChild(f);
    requestAnimationFrame(() => progress(0.45));
  }
  function disconnect() {
    if (!s.frame) return;
    release();
    s.frame.remove();
    s.frame = null;
    clearTimeout(s.slow);
    b.classList.remove("connecting", "ready");
    progress(0, "");
  }
  function ready() {
    clearTimeout(s.slow);
    progress(1, "Ready");
    clearTimeout(s.fade);
    s.fade = setTimeout(() => { b.classList.add("ready"); b.classList.remove("connecting"); }, 350);
  }
  addEventListener("message", (e) => {
    if (e.data && e.data.rsw === "ready" && s.frame && s.frame.contentWindow === e.source) ready();
  });

  function drive() {
    connect();
    b.classList.add("active");
    if (small()) {
      b.classList.add("full");
      document.body.classList.add("locked");
      $(document, ".tools .expand").textContent = "Close";
    }
    $(b, ".gate").setAttribute("aria-hidden", "true");
  }
  function release() {
    if (!b.classList.contains("active")) return;
    b.classList.remove("active", "full");
    $(document, ".tools .expand").textContent = "Full screen";
    $(b, ".gate").removeAttribute("aria-hidden");
    document.body.classList.remove("locked");
    if (small()) disconnect();  // back to the clip: no live page in a phone-wide block
  }
  $(b, ".gate").addEventListener("click", drive);
  $(document, ".tools .close").addEventListener("click", release);
  $(document, ".tools .expand").addEventListener("click", () => {
    if (b.classList.contains("full")) { release(); return; }
    drive();
    b.classList.add("full");
    document.body.classList.add("locked");
    $(document, ".tools .expand").textContent = "Exit full screen";
  });
  let leave = 0;
  b.addEventListener("mouseleave", () => { if (!b.classList.contains("full") && !small()) leave = setTimeout(release, 500); });
  b.addEventListener("mouseenter", () => clearTimeout(leave));
  addEventListener("keydown", (e) => { if (e.key === "Escape") release(); });

  // the block is only visible on step 4: the observer sees ratio 0 on the other steps, so it connects
  // when step 4 is shown in view, and lets go a while after another step (or the page's end) hides it
  const watch = new IntersectionObserver(([e]) => {
    s.ratio = e.intersectionRatio;
    s.outSince = e.intersectionRatio > 0 ? null : (s.outSince || Date.now());
    if (e.intersectionRatio >= 0.34 && !small()) connect();
  }, { threshold: [0, 0.1, 0.34, 0.6, 1] });
  watch.observe(b);
  setInterval(() => {
    if (s.frame && s.outSince && Date.now() - s.outSince > RELEASE_MS && !b.classList.contains("active")) disconnect();
  }, 4000);

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
      $(document, ".tools a.open").href = url(id);
      $(document, ".tools .dot").dataset.status = id;
      if (s.ratio >= 0.34 && !small()) connect();
    }
    for (const t of tabs) t.setAttribute("aria-selected", String(t.dataset.demo === id));
    poll();
  }
  for (const t of tabs) t.addEventListener("click", () => { show(t.dataset.demo); history.replaceState(null, "", `#demo=${t.dataset.demo}`); });
  $(document, ".tools a.open").href = url(b.dataset.id);

  // which simulation answers: the gateway's status route says 204 (up) or 502 (down or restarting)
  async function poll() {
    const id = b.dataset.id;
    let ok = false;
    try { ok = (await fetch(`https://${HOST}/status/${id}`, { cache: "no-store" })).ok; } catch (e) { ok = false; }
    if (b.dataset.id !== id) return;
    const dot = $(document, ".tools .dot");
    dot.classList.toggle("up", ok);
    dot.title = ok ? "Live" : "Starting or restarting";
  }
  poll();
  setInterval(poll, 20000);

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
  // walks through the steps; the rail, the arrows and the Next button scroll the page to a step's place.
  const flow = $(document, "#workflow"), stage = $(document, "#stage");
  const panels = [...stage.querySelectorAll(".panel")], steps = [...document.querySelectorAll(".rail [data-step]")];
  const next = $(document, "#next");
  const NEXT = { 1: "Next: the coding agent at work", 2: "Next: the simulation it produced", 3: "Next: drive it yourself 🔥", 4: "Next: Sim2Real" };
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
  $(stage, ".arrow.next").addEventListener("click", () => goTo(step + 1));
  $(stage, ".arrow.prev").addEventListener("click", () => goTo(step - 1));
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
