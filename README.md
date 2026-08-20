# Halbach Gradient Coils

Design and manufacturing workflow for `Gx`, `Gy`, and `Gz` gradient coils for a low-field MRI scanner based on a Halbach magnet. The project uses the included `pyCoilGen` engine to optimise wire tracks, then generates lead wires and printable shell geometry for a negative winding mould.

This repository is self-contained: cloning it provides both the Halbach-specific application and the local `pyCoilGen` source it requires.

## Scope and status

The application provides a configuration-driven pipeline for:

1. Creating a cylindrical design mesh and target field.
2. Optimising a wire layout with `pyCoilGen`.
3. Detecting locations where three or more return-path branches compete for
   the same two radial cable layers.
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
├── GradientDesign.bat      # Windows one-click setup and GUI launcher
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

- Windows with 64-bit CPython **3.11, 3.12, 3.13, or 3.14**. Python 3.14.1 is excluded because NetworkX does not support that interpreter patch release; use the newest available 3.14 patch instead.
- A working C/C++-free Python installation; the pinned dependency set provides Windows wheels for every supported version.
- FastHenry2 only if resistance and inductance metrics are needed. The pipeline remains usable without it and reports those values as `n/a`.

The supported range is declared in `pyproject.toml` and exercised by the Windows compatibility workflow. Versions outside that range are rejected rather than installed with an untested dependency combination.

## Installation

### One-click setup and launch on Windows

Install any supported 64-bit Python version first, then clone or extract the complete
repository. Double-click `GradientDesign.bat` in the repository root.

On its first run, the launcher:

1. Uses the newest compatible interpreter available through the Windows Python Launcher, in the order 3.14, 3.13, 3.12, then 3.11. If the launcher is unavailable, it accepts a compatible `python` on `PATH`.
2. Creates or repairs `.venv`.
3. Installs and verifies every dependency in `requirements-project.txt`.
4. Opens the GUI.

Later launches skip installation when the environment passes its dependency
check and open the GUI directly. If setup fails, the window remains open with
the error and the next launch retries it. The launcher never installs Python
itself or modifies a global Python environment.

### Manual installation

From the repository root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-project.txt
```

Replace `3.14` with `3.13`, `3.12`, or `3.11` when that is the interpreter you want to use. A virtual environment remains tied to the Python installation that created it; the launcher recreates an incompatible or broken environment automatically.

`requirements-project.txt` installs the local `pyCoilGen` package in editable mode and all application dependencies. This includes `networkx`, `rtree`, and `scikit-image`: they are optional `trimesh` extras, but this pipeline uses them for mesh component splitting, spatial queries, and voxel remeshing respectively. `pytest` is included for verification.

The commands deliberately call `.\.venv\Scripts\python.exe` directly. This
avoids relying on PowerShell activation or on whichever global `python` happens
to be on the system PATH.

### Resetting the virtual environment

From the repository root, close the GUI and any terminals or Python processes using the environment. Then run the following in a new PowerShell window. It removes only the repository's `.venv` directory and recreates it from the declared dependencies:

```powershell
Remove-Item -LiteralPath .venv -Recurse -Force
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-project.txt
```

If `Remove-Item` reports that a file is in use, close the process named in the message and rerun the command. Do not delete the repository root—only `.venv`.

To confirm the installation:

```powershell
.\.venv\Scripts\python.exe -c "import networkx, rtree; from skimage import measure; import trimesh; print('Runtime dependencies OK')"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

The current compatibility suite contains 58 passing tests. Two deprecation warnings from SciPy's MATLAB reader are expected and do not fail the suite.

## Running the application

Run the application commands from the repository root. Always use the virtual
environment executable so the GUI cannot start with a global Python missing the
project dependencies:

```powershell
.\.venv\Scripts\python.exe halbach_coils\run_gui.py
```

The GUI has three modes:

- **Pipeline**: runs gradient generation, leads, and shell creation in sequence.
- **Standalone**: runs one stage against an existing input file.
- **Tikhonov sweep**: evaluates a coarse/fine range of regularisation values and writes a summary table.

For a non-interactive run:

```powershell
.\.venv\Scripts\python.exe halbach_coils\run_pipeline.py --axis y --tikhonov 2500 --levels 26 --layer 2
.\.venv\Scripts\python.exe halbach_coils\run_pipeline.py --axis z --tikhonov 10000 --layer 3
```

Useful options include:

```text
--no-plots                  Run without interactive plots.
--skip-gradient             Reuse an existing wire STL.
--skip-leads --skip-shell   Generate only the wire layout.
--no-overlap-warn           Disable the multi-wire congestion warning.
```

Each pipeline run uses a unique directory such as `resultados/pipeline/Gy_tk2500_lvl26/`; repeated runs receive a numeric suffix instead of overwriting prior results.

## Configuration and pipeline stages

`halbach_coils/coilgen/config.py` is the single source of truth for geometry, conductor, target field, lead-wire, shell, and FastHenry settings. Avoid duplicating configuration values in scripts.

| Module | Responsibility |
| --- | --- |
| `coilgen.gradient` | prepares pyCoilGen arguments, runs optimisation, extracts field metrics |
| `coilgen.overlap` | groups pyCoilGen crossings and detects 3+ cables in one zone |
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

The Halbach application was previously stored in a folder named `pruebas/` and versioned separately from `pyCoilGen`. The current self-contained layout preserves that Git history while making a fresh clone directly installable.
