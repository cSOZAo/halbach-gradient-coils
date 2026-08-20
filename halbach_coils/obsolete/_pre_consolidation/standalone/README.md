# Standalone scripts — self-contained parameters

Each script has a **USER PARAMETERS** block at the top. No `coil_mold_common.py`.

Default output folder: ``../resultados/standalone/Gy_tk2500_lvl26/``

Re-running with the same axis / Tikhonov / levels appends ``(2)``, ``(3)``, … to
filenames instead of overwriting. PNG verify images are not saved.

```bash
python gradiente_belen_santi_main.py
python add_coil_leads.py
python generate_coil_shell_split.py
```

Shell halves are read from `../assets/cilindros_gradientes_grandes/`.

Update `INPUT_STL`, `OUTPUT_DIR`, and filenames in each script if you change
the gradient naming convention.
