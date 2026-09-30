import "./style.css";
import { App, type DemoFactory } from "./app/App.ts";
import { FreeDragDemo } from "./modes/FreeDrag.ts";

const factories: Record<string, DemoFactory> = {
  free: (stage, id, manifest) => new FreeDragDemo(stage, id, manifest),
};

// Track demos load lazily; the first page view is almost always a free-drag demo.
factories.track = (stage, id, manifest) => {
  let inner: ReturnType<DemoFactory> | null = null;
  // The app lays the stage out before the module arrives; keep that placement for the demo.
  let placed: Parameters<ReturnType<DemoFactory>["resize"]>[0] | null = null;
  const pending = import("./modes/Track.ts").then((m) => {
    inner = new m.TrackDemo(stage, id, manifest);
    if (placed) inner.resize(placed);
    (window as unknown as { __rsw: unknown }).__rsw = inner; // tests and the console see the real demo
    return inner;
  });
  return {
    start: async () => (await pending).start(),
    frame: (now, dt) => inner?.frame(now, dt),
    pointerDown: (e, p) => inner?.pointerDown(e, p),
    pointerMove: (e, p) => inner?.pointerMove(e, p),
    pointerUp: (e, p) => inner?.pointerUp(e, p),
    key: (e) => inner?.key?.(e),
    resize: (p) => {
      placed = p;
      inner?.resize(p);
    },
    roi: () => (inner ? inner.roi() : (manifest.roi as [number, number, number, number])),
    dispose: () => {
      void pending.then((d) => d.dispose());
    },
  };
};

void new App(factories).start().catch((e: unknown) => {
  console.error(e);
});
