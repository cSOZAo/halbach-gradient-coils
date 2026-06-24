# Pipeline — shared configuration

Edit **`coil_mold_common.py`** only (geometry, paths, subtract mode, lead params).

```bash
python run_coil_mold_pipeline.py
```

Flags in `coil_mold_common.py`:

- `RUN_GRADIENT` — run pyCoilGen (slow)
- `RUN_LEADS` — add lead wires
- `RUN_SHELL` — carve Fusion shell halves

Outputs go to `../resultados/resultados_grande_{axis}/final/` by default.
