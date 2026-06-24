# Project: Gradient Coils for Halbach LowField MRI

Designing gradient coils (X, Y, Z) for a Halbach-based low-field MRI scanner using **pyCoilGen** (Python).

## Hardware constraint — read before touching physics

- Scanner B0 is **parallel to +Y** (transverse to the bore).
- Gradient fields must therefore be oriented along **Y** as well (the relevant component is the one parallel to B0).
- pyCoilGen internally optimizes **Bz** — mesh rotation is used to align geometry so that the optimized component ends up matching the physical Y axis. Keep this in mind when porting designs or interpreting results.

## Layout

```
standalone/          # 3-step mold workflow; USER PARAMETERS in each script
pipeline/            # same workflow; edit coil_mold_common.py only
  coil_mold_common.py
  gradiente_belen_santi_main.py
  add_coil_leads.py
  generate_coil_shell_split.py
  run_coil_mold_pipeline.py
assets/
  cilindros_gradientes_grandes/   # Fusion 360 shell halves (g_2a, g_2b, …)
  CoilGen_MatLab/                 # read-only MATLAB reference
resultados/          # outputs (gitignored) — see resultados/STORAGE.md
obsolete/            # superseded scripts (golden angle, old shell gen, sweeps)
```

Frozen per-axis copies and golden-angle ports live under `obsolete/CodigosGitLabOSI/`.

## Current task: MATLAB → pyCoilGen port

- Sources: `CoilGen_MatLab/script_belen_santi.m` and `GoldenAngle_GradientCoil.m` (both known-working MATLAB designs).
- Create **new files** for each ported design; do not overwrite `gradiente_halbach_test.py` or `Curvas_L.py` unless explicitly asked.
- Goal: **1:1 port** where feasible, adapted to pyCoilGen's API.
- **Comment ported code thoroughly** — explain each step so the user can follow MATLAB → Python correspondence.
- For each MATLAB design, keep one **frozen** per-direction/channel copy + one **editable main** with `GRADIENT_AXIS` / `CHANNEL` selector. Frozen copies live in `.claudeignore`.

## Conventions

- Code, comments, and docstrings in **English**.
- Units: SI (meters, amperes, tesla).
- When adding physics, state assumptions about coordinate frame (mesh vs. optimized component vs. physical B0).

## pyCoilGen API — hard-won rules

**Valid parameter keys**: Every key passed to `pyCoilGen(log, arg_dict)` must be registered in `sub_functions/parse_input.py` OR in a plugin's `register_args()` (mesh plugins: `build_cylinder_mesh.py`, `create_stl_mesh.py`, etc.; export: `export_cad_file.py`). An unregistered key raises `KeyError` immediately. Before adding a new key, grep `add_argument` in those files.

- `smooth_flag` **does not exist** — smoothing is controlled only by `smooth_factor` (>1 enables it).
- `coil_mesh_file` is valid (registered by `create_stl_mesh.py`); `coil_mesh` is the newer alias.
- `field_shape_function` is **ignored** when `target_field_definition_file != 'none'`; the file takes full precedence over the symbolic function.

**Target field file format** (`.npy` pickle, replaces MATLAB `.mat`):
```python
np.save(path, np.array([{'coords': arr_3xN, 'field_name': arr_N}], dtype=object), allow_pickle=True)
```
`define_target_field.py` does `[loaded] = np.load(..., allow_pickle=True)` and reads `loaded['coords']` + `loaded[field_name]`.

**`cross_sectional_points`**: must be a *clean, convex* closed 2-D polygon — passed to Delaunay triangulation. The MATLAB oval construction in `script_belen_santi.m` was buggy (two disjoint semicircles); replaced with a smooth ellipse sampled via `np.linspace(0, 2π, N)`.

**Extracting results after the run**:
- `solution.solution_errors.combined_field_layout_per1Amp[2]` → realized Bz at 1 A [T/A]
- `solution.solution_errors.target_field_1A.b[2]` → target Bz at 1 A [T/A] (pyCoilGen-scaled)
- `solution.target_field.coords` → target point coordinates [m], shape (3, N)
