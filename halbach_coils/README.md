# Halbach application package

The runnable Halbach-specific application lives in this directory. Its project-level documentation, installation instructions, coordinate conventions, and workflow are maintained in the repository [README](../README.md).

Run entry points from this directory:

```powershell
python run_gui.py
python run_pipeline.py --help
```

Generated output is written to `resultados/` and is not tracked by Git. Historical implementations are retained in `obsolete/` for reference only; the supported implementation is the `coilgen/` package.
