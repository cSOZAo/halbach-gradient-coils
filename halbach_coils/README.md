# Halbach application package

This directory contains the supported Halbach-specific application built on the
vendored `pyCoilGen` engine. Start with the repository-level [README](../README.md)
for requirements, first-time installation, coordinate conventions, validation
warnings, and the complete workflow.

## Entry points

Run commands from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe halbach_coils\run_gui.py
.\.venv\Scripts\python.exe halbach_coils\run_pipeline.py --help
```

On Windows, `GradientDesign.bat` at the repository root is the recommended
launcher because it creates and verifies `.venv` before opening the GUI.

The GUI starts in English. Its only settings menu currently contains
**Settings > Language**, where the interface can be changed to Spanish for the
current session.

## Directory guide

| Path | Purpose |
| --- | --- |
| `coilgen/` | Supported pipeline, configuration, metrics, sweep, lead, and shell code |
| `gui/` | Tkinter panels and the English/Spanish translation catalogue |
| `assets/shells/` | Tracked matching STL halves discovered by the GUI |
| `run_pipeline.py` | Non-interactive pipeline entry point |
| `run_gui.py` | Graphical entry point |
| `resultados/` | Generated run directories; ignored by Git |
| `obsolete/` | Historical implementations; unsupported and retained only for reference |

## Shell asset convention

Each printable shell layer is a matching pair named `g_Na.stl` and `g_Nb.stl`,
where `N` is the layer number. Both files must exist in `assets/shells/` before
the pair appears in the GUI. Restart the GUI after adding a pair.

Do not commit simulation output to the assets directory. A generated run belongs
under `resultados/` or another output directory selected by the user.

## Where to make changes

- Change physical and manufacturing defaults in `coilgen/config.py`; it is the
  configuration source of truth.
- Add GUI source text in English and its Spanish translation in `gui/i18n.py`.
- Keep scanner coordinates distinct from `pyCoilGen` internal coordinates; the
  mapping and mesh rotation are documented in the main README.
- Treat `obsolete/` as read-only history. Its scripts use old layouts and may
  contain hard-coded paths or assumptions that no longer match the application.

Before submitting changes, run:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```
