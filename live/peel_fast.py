"""The peel page, faster: a cheaper stepping budget, a leaner collision pass, an instant "New film".

**Where the time went** (2026-10-01, RTX PRO 6000, one control step replayed from its captured
graph). ``rgbs/panel`` stages ``cloth_vbd_bimanual@v1``: 60 Hz control, 20 substeps, 80 VBD sweeps:

    VBD, 80 sweeps x 20 substeps   159.9 ms   (96 us a sweep: ~22 tiny kernels, latency-bound)
    MuJoCo arms x 20                 8.3 ms
    the coupler's collision x 20     7.3 ms
    bond field + pins x 20           0.1 ms
                                   -------
    the substep block              176.0 ms   -> 5.6 control steps/s, 0.09x real time

**The budget: keep the dt, cut the sweeps.** Measured by playing the routine's pre-roll and then
the page's autopilot peel at each budget and comparing bond-break schedules, peel front, strain,
peak load, latch and film shape against the task's own (20 x 80, which reproduces to 0.4 mm):

* every budget with fewer **substeps** fails: at 10 the last row never lets go and the sheet
  stretches to +94 % (10 x 20, 10 x 40; 10 x 10 reaches +39 %), and at 6 or 4 no bond breaks at
  all while the film stretches 30-40 fold. 1/1200 s is the floor the task says it is.
* at 20 substeps the **sweeps** can go. 20 x 10 breaks 14 of the 15 bond rows within 2.5 mm of hand
  travel of the reference (20 x 160, twice the task's sweeps), one row 14 mm late -- which 20 x 40
  and 20 x 20 do too -- and the task's own 20 x 80 sits 1.1 mm (mean) off that same reference,
  where a twice-finer dt (40 x 40) moves three rows by 13 mm. Peak strain 5.0 % against 4.5-4.7 %.
  20 x 5 is cheaper again and visibly further off (two more rows 7-9 mm early, peak strain 6.1 %).

So :data:`SUBSTEPS` stays the task's and :data:`ITERATIONS` is 10. The scene is built exactly as
the task stages it -- settle, bonds, anchor -- and only the stepping budget changes afterwards, at
runtime, through :func:`apply_budget`.

**The collision pass** (:class:`SoftContactsOnly`): the coupler collides all 242 rigid shapes every
substep for contacts VBD never reads; the soft half alone is the same contacts (checked on every
install) for ~4 ms less a control step (8.7 -> 5.1 ms at 20 substeps).

**"New film"** used to reset and replay the routine's hold-and-pinch: 801 control steps, ~150 s at
the task's budget. The state the pre-roll ends in is always the same, so :class:`FilmSnapshot`
keeps it -- every device array of both states and the control, the bond field, the anchor pin's
flags, the latch (the routine's ``ParticleDragger``, which stays installed) and MuJoCo's warm start
-- and :meth:`FilmSnapshot.restore` puts it back in milliseconds. A restored film peels exactly as
the fresh one did (bond schedule identical, film within 0.3 mm), also after the state has been
poisoned with NaN. The routine, its latch and its follower hook are the same objects throughout;
nothing is re-installed, so nothing can be left behind.
"""

from __future__ import annotations

import numpy as np
import warp as wp

SUBSTEPS = 20
"""Physics substeps per 60 Hz control step: the task's own 20 (dt = 1/1200 s), kept."""
ITERATIONS = 10
"""VBD sweeps per substep (the task's 80). See the module docstring for what 10 costs."""

_MJ_BUFFERS = ("qpos", "qvel", "qacc_warmstart", "qfrc_applied", "ctrl", "act", "xfrc_applied")
"""MuJoCo-Warp data that carries from one step to the next (the rest is re-derived every step)."""


def apply_budget(env, substeps: int = SUBSTEPS, iterations: int = ITERATIONS) -> tuple[int, int]:
    """Step ``env`` at ``substeps`` x ``iterations`` from now on. Returns the budget it had.

    The pipeline's substep count and dt, and the VBD island's sweep count; the captured graph is
    dropped and re-captured on the next step (one eager step first). MuJoCo takes its timestep
    from the dt it is stepped with, so the arms follow without being told.
    """
    if int(substeps) < 2 or int(substeps) % 2:
        raise ValueError(f"substeps must be even and >= 2 (the state swap), got {substeps}")
    pipe = env.pipeline
    soft = pipe.solver.soft
    before = (int(pipe.substeps), int(soft.iterations))
    pipe.substeps = int(substeps)
    pipe.sim_dt = pipe.control_dt / pipe.substeps
    soft.iterations = int(iterations)
    pipe.invalidate_graph()
    return before


class SoftContactsOnly:
    """The one-way coupler's per-substep collision pass, without the half nobody reads.

    ``SolverOneWay`` runs a full Newton ``CollisionPipeline`` every substep and hands its contacts
    to VBD. VBD in external-rigid mode (the arms are MuJoCo's) reads only the **soft** contacts --
    particle against shape -- and skips every rigid-rigid record; MuJoCo finds its own. So the
    broad phase and the GJK/MPR/mesh narrow phase over all 242 rigid shapes are computed 1200
    times a second for nothing. This is that pipeline's own soft-contact launch, verbatim
    (``newton/_src/sim/collide.py``), behind the counter reset it fuses into its first kernel.
    :func:`install_soft_contacts_only` checks it against the real pipeline before using it.
    """

    def __init__(self, pipeline) -> None:
        from newton._src.geometry.kernels import create_soft_contacts

        self.inner = pipeline
        self.model = pipeline.model
        self.device = pipeline.device
        self.kernel = create_soft_contacts

    def contacts(self):
        return self.inner.contacts()

    def collide(self, state, contacts, *, soft_contact_margin: float | None = None) -> None:
        inner, model = self.inner, self.model
        margin = inner.soft_contact_margin if soft_contact_margin is None else soft_contact_margin
        contacts.contact_counters.zero_()  # [rigid count, soft count]: no rigid records at all
        if not (state.particle_q and inner.soft_contact_max > 0 and inner.soft_rigid_contact_pair_count > 0):
            return
        wp.launch(
            kernel=self.kernel,
            dim=inner.soft_rigid_contact_pair_count,
            inputs=[
                inner.soft_rigid_contact_pairs,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.particle_world,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_source_ptr,
                model._shape_mesh_properties,
                model.shape_world,
                margin,
                model.shape_margin,
                inner.soft_contact_max,
                model.shape_flags,
                model.shape_heightfield_index,
                model.heightfield_data,
                model.heightfield_elevations,
            ],
            outputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_tids,
            ],
            device=self.device,
        )


_SOFT_FIELDS = ("soft_contact_particle", "soft_contact_shape", "soft_contact_body_pos", "soft_contact_body_vel",
                "soft_contact_normal", "soft_contact_indices", "soft_contact_barycentric")


def _soft_records(contacts) -> np.ndarray:
    n = int(contacts.soft_contact_count.numpy()[0])
    cols = [np.asarray(getattr(contacts, f).numpy()[:n], dtype=np.float64).reshape(n, -1) for f in _SOFT_FIELDS]
    rec = np.concatenate(cols, axis=1)
    return rec[np.lexsort(rec[:, :2].T[::-1])]


def install_soft_contacts_only(env) -> str:
    """Swap the coupler's collision pass for :class:`SoftContactsOnly` if it gives the same soft contacts.

    Returns what happened, for the log. Anything unexpected keeps the original pipeline.
    """
    solver = env.pipeline.solver
    original = getattr(solver, "collision", None)
    if original is None or not hasattr(original, "soft_rigid_contact_pairs") or isinstance(original, SoftContactsOnly):
        return "kept the full collision pass (nothing to swap)"
    try:
        state, contacts = env.pipeline.state_0, solver.contacts
        original.collide(state, contacts)
        want = _soft_records(contacts)
        lean = SoftContactsOnly(original)
        lean.collide(state, contacts)
        got = _soft_records(contacts)
        if want.shape != got.shape or not np.array_equal(want, got):
            original.collide(state, contacts)
            return f"kept the full collision pass (soft contacts differ: {want.shape} vs {got.shape})"
    except Exception as exc:  # a Newton that moved on: keep its own pass
        return f"kept the full collision pass ({type(exc).__name__}: {exc})"
    solver.collision = lean
    env.pipeline.invalidate_graph()
    return f"soft-contact-only collision pass ({len(want)} soft contacts identical)"


def _arrays(obj) -> dict[str, wp.array]:
    return {k: v for k, v in vars(obj).items() if isinstance(v, wp.array)}


class FilmSnapshot:
    """Everything a peel in progress carries, taken once the routine has the tab latched.

    Args:
        env: The page's environment.
        routine: The ``RoutineFilmPeel`` that just reached ``phase == "peel"``.
    """

    def __init__(self, env, routine) -> None:
        self.routine = routine
        pipe = env.pipeline
        self._state0 = {k: wp.clone(v) for k, v in _arrays(pipe.state_0).items()}
        self._state1 = {k: wp.clone(v) for k, v in _arrays(pipe.state_1).items()}
        self._control = {k: wp.clone(v) for k, v in _arrays(pipe.control).items()}
        film = routine.film
        self._bonds = wp.clone(film.bonds.bond_active)
        pin = getattr(film, "pin", None)
        self._pin = np.copy(pin._enabled_host) if pin is not None else None
        d = routine.dragger
        self._dragger = {k: wp.clone(v) for k, v in _arrays(d).items()}
        self._dragger_host = {k: np.copy(v) for k, v in vars(d).items() if isinstance(v, np.ndarray)}
        data = pipe.solver.rigid.mjw_data
        self._mj = {k: wp.clone(getattr(data, k)) for k in _MJ_BUFFERS if isinstance(getattr(data, k, None), wp.array)}
        self._clock = (pipe.step_count, pipe.sim_time, env.step_count)
        self.n_latched = int(d.n_grabbed)
        self.restores = 0

    def restore(self, env) -> None:
        """Put the film, both hands and the latch back where the pre-roll left them."""
        pipe = env.pipeline
        routine = self.routine
        for target, saved in ((pipe.state_0, self._state0), (pipe.state_1, self._state1), (pipe.control, self._control)):
            for k, v in saved.items():
                wp.copy(getattr(target, k), v)
        film = routine.film
        wp.copy(film.bonds.bond_active, self._bonds)
        if film.bonds._active_host is not None:
            film.bonds._active_host.invalidate()
        pin = getattr(film, "pin", None)
        if pin is not None and self._pin is not None:
            pin._enabled_host[:] = self._pin
            pin._enabled.assign(pin._enabled_host)
        d = routine.dragger
        for k, v in self._dragger.items():
            wp.copy(getattr(d, k), v)
        for k, v in self._dragger_host.items():
            setattr(d, k, np.copy(v))
        film.latch = d
        data = pipe.solver.rigid.mjw_data
        for k, v in self._mj.items():
            wp.copy(getattr(data, k), v)
        pipe.step_count, pipe.sim_time, env.step_count = self._clock
        pipe.mark_dirty()
        wp.synchronize()
        self.restores += 1

    def healthy(self, env) -> bool:
        """The restored state is finite and the latch still holds the tab."""
        q = env.pipeline.state_0.particle_q.numpy()
        return bool(np.isfinite(q).all()) and self.routine.dragger.n_grabbed == self.n_latched > 0
