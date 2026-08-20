# Halbach Gradient Coils

Design and manufacturing workflow for `Gx`, `Gy`, and `Gz` gradient coils for a low-field MRI scanner based on a Halbach magnet. The project uses the included `pyCoilGen` engine to optimise wire tracks, then generates lead wires and printable shell geometry for a negative winding mould.

This repository is self-contained: cloning it provides both the Halbach-specific application and the local `pyCoilGen` source it requires.

## Scope and status

The application provides a configuration-driven pipeline for:

1. Creating a cylindrical design mesh and target field.
2. Optimising a wire layout with `pyCoilGen`.
3. Checking non-adjacent wire segments for insufficient clearance.
4. Adding lead wires to the coil layout.
5. Cutting the wire grooves into an STL shell for manufacturing.
6. Optionally evaluating resistance and inductance through FastHenry2.

The current workflow is intended for iterative engineering work. Treat generated geometry and electrical values as design inputs that still require physical and simulation validation before manufacture.

## Repository layout

```text
halbach-gradient-coils/
├── halbach_coils/          # Halbach application (formerly named pruebas)
│   ├── coilgen/            # pipeline implementation and configuration
│   ├── gui/                # Tkinter user interface
│   ├── assets/shells/      # tracked printable shell halves
│   ├── obsolete/           # historical code retained for reference
│   ├── run_pipeline.py     # command-line entry point
│   └── run_gui.py          # graphical entry point
├── pyCoilGen/              # vendored coil-layout engine
├── data/                   # runtime data searched by pyCoilGen
├── tests/                  # automated regression tests for pyCoilGen
├── requirements-project.txt
└── pyproject.toml
```

Simulation outputs are written under `halbach_coils/resultados/`. They are deliberately ignored by Git so that a run never changes the source history or adds large generated files to a commit.

## Coordinate system: read before changing physics

The scanner's static field `B0` is parallel to physical `+Y`, transverse to the bore. The relevant gradient-field component is therefore also parallel to physical `Y`.

`pyCoilGen` optimises its internal `Bz` component. To align this internal convention with the scanner, the cylindrical mesh is rotated by `R_y(pi/2)`. `Config.internal_axis` performs the physical-to-internal mapping:

| Physical gradient axis | pyCoilGen internal axis |
| --- | --- |
| `x` | `y` |
| `y` | `z` |
| `z` | `x` |

When modifying geometry, target fields, or field metrics, document which frame each vector uses. Mixing scanner coordinates and pyCoilGen coordinates is the most likely source of physically plausible but incorrect results.

## Requirements

- Windows with Python **3.11** (the validated environment).
- A working C/C++-free Python installation; all required packages are provided as wheels for the validated setup.
- FastHenry2 only if resistance and inductance metrics are needed. The pipeline remains usable without it and reports those values as `n/a`.

Python 3.14 is not the baseline for this codebase. Dependency upgrades and Python-modernisation work should be done in a separate branch after preserving the current validated behaviour.

## Installation

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-project.txt
```

`requirements-project.txt` installs the local `pyCoilGen` package in editable mode and adds the Halbach-specific runtime packages (`manifold3d` and `pandas`) plus `pytest` for verification.

To confirm the installation:

```powershell
python -m pytest -q
```

The current baseline is 56 passing tests. Two deprecation warnings from SciPy's MATLAB reader are expected and do not fail the suite.

## Running the application

Run the application commands from `halbach_coils/`:

```powershell
cd halbach_coils
python run_gui.py
```

The GUI has three modes:

- **Pipeline**: runs gradient generation, leads, and shell creation in sequence.
- **Standalone**: runs one stage against an existing input file.
- **Tikhonov sweep**: evaluates a coarse/fine range of regularisation values and writes a summary table.

For a non-interactive run:

```powershell
cd halbach_coils
python run_pipeline.py --axis y --tikhonov 2500 --levels 26 --layer 2
python run_pipeline.py --axis z --tikhonov 10000 --layer 3
```

Useful options include:

```text
--no-plots                  Run without interactive plots.
--skip-gradient             Reuse an existing wire STL.
--skip-leads --skip-shell   Generate only the wire layout.
--no-overlap-warn           Disable the clearance warning.
```

Each pipeline run uses a unique directory such as `resultados/pipeline/Gy_tk2500_lvl26/`; repeated runs receive a numeric suffix instead of overwriting prior results.

## Configuration and pipeline stages

`halbach_coils/coilgen/config.py` is the single source of truth for geometry, conductor, target field, lead-wire, shell, and FastHenry settings. Avoid duplicating configuration values in scripts.

| Module | Responsibility |
| --- | --- |
| `coilgen.gradient` | prepares pyCoilGen arguments, runs optimisation, extracts field metrics |
| `coilgen.overlap` | checks non-adjacent wire clearance |
| `coilgen.leads` | creates and joins lead wires |
| `coilgen.shell` | produces the grooved shell geometry |
| `coilgen.sweep` | runs Tikhonov parameter studies |
| `coilgen.metrics` | records slope, RMSE, relative error, and electrical values |
| `coilgen.fasthenry` | resolves an optional FastHenry2 executable |

Lead wires exit along the bore (`+/-X` after the mesh rotation). Their angular station is configured per gradient axis. These defaults are starting points, not manufacturing guarantees; inspect generated geometry and collision warnings for each design.

## FastHenry2

FastHenry2 is optional. The resolver checks, in order:

1. A path configured in `Config.fasthenry.bin_path`.
2. An executable available on `PATH` (`FastHenry2.exe`, `fasthenry.exe`, `fasthenry2`, or `fasthenry`).
3. The historical Windows installation path, only if it exists.

If no executable is found, the design proceeds and the metrics file explicitly records unavailable R/L values. Do not treat `n/a` as a physical result.

## pyCoilGen integration notes

The local engine is intentionally part of this repository because the Halbach workflow relies on it directly. When changing the interface between the application and the engine:

- Every argument passed to `pyCoilGen(log, arg_dict)` must be registered in `pyCoilGen/sub_functions/parse_input.py` or by the relevant plugin.
- `smooth_flag` is not a valid pyCoilGen argument; smoothing is controlled by `smooth_factor`.
- A `target_field_definition_file` takes precedence over `field_shape_function`.
- Preserve the GPL licence and upstream attribution when redistributing modifications.

## Development workflow

1. Start from a clean Git branch.
2. Run the test suite before and after changes to `pyCoilGen` or shared geometry code.
3. Use a deliberately named output directory for physics experiments, and retain the resulting metrics outside Git when needed.
4. Record coordinate-frame assumptions and units in code comments and commits.
5. Keep `obsolete/` read-only unless a historical path has been independently reproduced by the current pipeline.

## Project history

The Halbach application was previously stored in a folder named `pruebas/` and versioned separately from `pyCoilGen`. The `migration/monorepo` branch introduces the self-contained layout above while preserving the previous Git history. Review and merge that branch before using this structure as the default `main` branch.
