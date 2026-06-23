# Project: Gradient Coils for Halbach LowField MRI

Designing gradient coils (X, Y, Z) for a Halbach-based low-field MRI scanner using **pyCoilGen** (Python).

## Hardware constraint — read before touching physics

- Scanner B0 is **parallel to +Y** (transverse to the bore).
- Gradient fields must therefore be oriented along **Y** as well (the relevant component is the one parallel to B0).
- pyCoilGen internally optimizes **Bz** — mesh rotation is used to align geometry so that the optimized component ends up matching the physical Y axis. Keep this in mind when porting designs or interpreting results.

## Layout

- `gradiente_halbach_test.py` — current working script to generate a single gradient coil. Editable.
- `Curvas_L.py` — L-curve sweeps (Tikhonov / turns / mesh divs) for regularization studies. Editable.
- `gradientes.py` — helpers / shared utilities.
- `gradiente_belen_santi_main.py` — **active** belen_santi port; editable. Switch axis with `GRADIENT_AXIS = 'x'|'y'|'z'`.
- `gradiente_belen_santi_Gx/Gy/Gz.py` — **frozen** per-axis reference copies; listed in `.claudeignore`, do not modify.
- `gradiente_golden_angle_main.py` — **active** GoldenAngle_GradientCoil.m port; editable. Switch channel with `CHANNEL = 1|2|3`. Gradient direction is non-axis-aligned (`rot_i @ [1,0,0]`).
- `gradiente_golden_angle_Ch1/Ch2/Ch3.py` — **frozen** per-channel reference copies; listed in `.claudeignore`, do not modify.
- `resultados_halbach/` — output STLs, plots, pickles.
- `resultados_belen_santi_main_*/` — outputs of the active belen_santi port.
- `resultados_golden_angle_main_Ch*/` — outputs of the active golden_angle port.
- `lcurve_x_tikhonov/` — L-curve outputs.
- `CoilGen_MatLab/` — **read-only reference**. Partner's working MATLAB designs. `script_belen_santi.m` is the authoritative design to port.
- `*_BACKUP.py` — **do not modify**.

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
