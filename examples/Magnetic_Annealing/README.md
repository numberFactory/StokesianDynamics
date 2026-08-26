# Mag_Anneal.py

Stokesian Dynamics simulation of magnetic microparticles near a wall:
permanent + optionally mutually-polarized induced dipoles, a rotating
in-plane field, and a scheduled vertical (z) field.

## Run

```bash
OMP_NUM_THREADS=1 python Mag_Anneal.py
```

Set `OMP_NUM_THREADS=1` for best performance — without it, OpenMP threads
spawned by the linear algebra underneath the Stokesian Dynamics solver can
oversubscribe the CPU alongside numba's own parallel threading, making
runs slower rather than faster.

No config file — every parameter below is a plain variable near the top
of `main()`; edit and rerun. Frames are written to `mag_anneal_frames/`
(or `mag_anneal_stochastic_frames/` if `Stoch=True`), next to the script.

## Parameters

### `B_z_schedule` — vertical field protocol
List of `(time_s, B_z_mT)` steps. `B_z` takes the value from the *last*
entry whose time has passed:

```python
B_z_schedule = [
    (0.0,   0.0),     # off at t=0
    (50.0,  0.435),   # steps up to 0.435 mT at t=50s
    (100.0, 0.0),     # back off at t=100s
]
```

Add, remove, or reorder entries (keep them sorted by time) to script any
ramp/pulse/hold protocol you want. Make sure `t_end` reaches at least the
last time you care about, or the schedule's tail is never used.

### `N` — number of particles
Set alongside `phi` further down `main()`; the two together determine the
box size `L_xy` (solved so `N` particles fit at that packing fraction).
Cost scales with the number of neighbor-list pairs, not `N²` — but larger
`N` is still proportionally more expensive per step, and `place_particles`
can get slow (many rejected placement attempts) if `N` and `phi` are both
pushed high at once.

### Also worth knowing
- **`B_0`** (mT) — rotating in-plane field amplitude.
- **`B_freq`** (Hz) — rotation frequency of that field.
- **`Stoch`** — `True` adds Brownian motion; `False` (default) is purely
  deterministic.
- **`mutual_polarizability_mag`** — `False` (default): each particle's
  induced moment is a simple linear response to the applied field. `True`:
  solved self-consistently via GMRES, accounting for every particle's
  field on every other — more physically complete, notably more expensive
  per step.
- **`phi`** — target areal packing fraction, used with `N` to size the box.
