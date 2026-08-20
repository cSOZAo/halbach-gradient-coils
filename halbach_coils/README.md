# Halbach application package

The runnable Halbach-specific application lives in this directory. Its project-level documentation, installation instructions, coordinate conventions, and workflow are maintained in the repository [README](../README.md).

From the repository root, run entry points with the project's virtual
environment. This prevents accidentally using a global Python without the
project dependencies:

```powershell
.\.venv\Scripts\python.exe halbach_coils\run_gui.py
.\.venv\Scripts\python.exe halbach_coils\run_pipeline.py --help
```

Generated output is written to `resultados/` and is not tracked by Git. Historical implementations are retained in `obsolete/` for reference only; the supported implementation is the `coilgen/` package.
