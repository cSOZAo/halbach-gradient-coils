# Gradient coils — Halbach low-field MRI (pyCoilGen)

## Layout

```
pruebas/
  standalone/     # 3 independent scripts — USER PARAMETERS in each file
  pipeline/       # same workflow — all knobs in coil_mold_common.py
  assets/         # Fusion cylinder STLs + MATLAB reference (read-only)
  resultados/     # simulation outputs (not in git)
  obsolete/       # old experiments — do not use for production
```

## Workflow (unchanged logic)

1. **Gradient** — `gradiente_belen_santi_main.py` → wire STL  
2. **Leads** — `add_coil_leads.py` → `*_with_leads.stl`  
3. **Mold** — `generate_coil_shell_split.py` → `*_shell_g2a.stl`, `*_g2b.stl`

### Standalone (manual parameters)

```bash
cd standalone
python gradiente_belen_santi_main.py    # writes output/
python add_coil_leads.py
python generate_coil_shell_split.py
```

Edit the **USER PARAMETERS** block at the top of each script.

### Pipeline (shared config)

```bash
cd pipeline
# edit coil_mold_common.py once
python run_coil_mold_pipeline.py
```

Or run steps individually from `pipeline/` (they import `coil_mold_common` only).

## Assets

- `assets/cilindros_gradientes_grandes/` — Fusion 360 half-cylinders `g_2a.stl`, `g_2b.stl`, …
- `assets/CoilGen_MatLab/` — partner MATLAB reference (`script_belen_santi.m`)

## Results disk usage (~81 GB total)

See `resultados/STORAGE.md` for a breakdown and deletion candidates.
