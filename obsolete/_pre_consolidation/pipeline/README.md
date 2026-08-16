# Pipeline — shared configuration

Edit **`coil_mold_common.py`** only (geometry, paths, subtract mode, lead params).

```bash
python run_coil_mold_pipeline.py
```

Flags in `coil_mold_common.py`:

- `RUN_GRADIENT` — run pyCoilGen (slow)
- `RUN_LEADS` — add lead wires
- `RUN_SHELL` — carve Fusion shell halves

Outputs go to ``../resultados/pipeline/Gy_tk2500_lvl26/`` (or ``…(2)`` on re-run).
Each full pipeline run gets its own subfolder with gradient, leads, and shell STLs.
