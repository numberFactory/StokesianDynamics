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
of `main()`; edit and rerun. Frames and configuration snapshots are
written together into a directory named from the current run's own
parameters (see **Output directory naming** below) next to the script.

`main()` also accepts `restart_dir`/`restart_index` to continue a run from
a saved configuration instead of placing particles fresh — see
**Restarting from a saved configuration**.

## Output directory naming

The output directory name is built from the parameters that most affect
what a run actually is:

```
mag_anneal_N{N}_B0_{B_0}_Bfreq_{B_freq}_Bfreqy_{B_freq_y}_dt_{dt}_stoch_{Stoch}_mutualpol_{mutual_polarizability_mag}
```

e.g. `mag_anneal_N1000_B0_0.95_Bfreq_5_Bfreqy_5_dt_0.005_stoch_False_mutualpol_False`.
`dt` is derived from `B_freq_y` (see below), not set independently, but is
included in the name since it affects run cost and step count directly.
Two runs with the same label overwrite each other's output — change at
least one of these parameters (or note that a restart run's label always
has `_restart` appended, so it never collides with its source) if you
want to keep both.

## Output files

Every `n_plot` steps, two files are written together into the output
directory, sharing the same 7-digit, zero-padded frame number:

- **`frame_{i:07d}.png`** — a snapshot: particles as 2-D discs with an
  orientation arrow (projection of the body x-axis into the xy plane),
  plus the applied-field and induced-moment arrows described in
  `plot_frame`'s own docstring.
- **`config_{i:07d}.txt`** — the full configuration behind that same
  frame: one row per particle, columns `x y z qw qx qy qz` (position,
  then a scalar-first quaternion). This is what **Restarting from a saved
  configuration** below reads back in.

`frame_0000042.png` and `config_0000042.txt` always describe the same
instant — they're written from the same `bodies` state, one right after
the other.

## Restarting from a saved configuration

To continue a run from a previously saved configuration instead of
placing particles fresh, call `main()` with `restart_dir` (the earlier
run's output directory) and `restart_index` (which `config_{i:07d}.txt`
to load):

```python
main(restart_dir='mag_anneal_N1000_B0_0.95_Bfreq_5_Bfreqy_5_dt_0.005_stoch_False_mutualpol_False',
     restart_index=42)
```

A few things worth knowing about how this works:

- **Only position and orientation are read from `restart_dir`.** Every
  physical, field, and simulation parameter (`N`, `a`, `B_0`, `dt`,
  `B_z_schedule`, ...) still comes from the current call's own settings
  in `main()` — not from whatever produced `restart_dir`. This means you
  can restart a finished run under a *different* field protocol or
  parameter set, starting from wherever the earlier configuration left
  off, as long as `N` matches (see below).
- **`N` must match the loaded configuration's particle count**, or
  `main()` raises immediately rather than silently truncating or padding
  the configuration. Set `N` to match the source run before restarting
  from it.
- **The output directory is still named from the *current* run's own
  parameters** (per **Output directory naming** above), with `_restart`
  appended — never the source directory's name, even if some parameters
  differ between the two runs. So restarting
  `mag_anneal_N500_B0_0.95_..._mutualpol_False` under a new `B_0` produces
  something like `mag_anneal_N500_B0_1.10_..._mutualpol_False_restart`,
  not a directory named after the source.
- **A small provenance file is written into the new output directory**,
  named after the source directory with `.txt` appended (e.g.
  `mag_anneal_N500_B0_0.95_..._mutualpol_False.txt`), so `ls` on a
  restarted run's output immediately shows where it came from. Its
  contents are just the source directory's full path and the
  `restart_index` used.

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
pushed high at once. If restarting from a saved configuration, `N` must
match that configuration's particle count exactly (see **Restarting from
a saved configuration** above).

### Also worth knowing

- **`B_0`** (mT) — rotating in-plane field amplitude.
- **`B_freq`** (Hz) — rotation frequency of that field.
- **`B_freq_y`** — y-component frequency of the rotating field direction
  (`m_rot_fn`); currently set equal to `B_freq`, but is its own variable
  and can be changed independently to trace out a non-circular (Lissajous)
  path instead of a simple rotation. Also sets the timestep: `dt =
  0.025/B_freq_y`.
- **`Stoch`** — `True` adds Brownian motion; `False` (default) is purely
  deterministic.
- **`mutual_polarizability_mag`** — `False` (default): each particle's
  induced moment is a simple linear response to the applied field. `True`:
  solved self-consistently via GMRES, accounting for every particle's
  field on every other — more physically complete, notably more expensive
  per step.
- **`phi`** — target area packing fraction for initialization only.
