# Pipeline Gz — inner cylinder (g_3a / g_3b)

> **Archived and unsupported.** The values below record an old Gz experiment;
> they are not current manufacturing defaults. Use the current configuration and
> entry points described in the [project README](../../../../README.md).

Copy of `pipeline/` configured for a **physical Z gradient** on Fusion layer 3.

| Setting | `pipeline/` (Gy) | `pipeline_gz/` (Gz) |
|---------|------------------|---------------------|
| Gradient axis | Y | Z |
| Fusion halves | g_2a / g_2b | g_3a / g_3b |
| CoilGen radius | 150 mm | 141.6 mm |
| CoilGen height | 430 mm | 430 mm (unchanged) |
| ROI / turns / wire | 125 mm sphere, 26 levels, 2.25 mm | same |
| Tikhonov | 2500 | **25000** |

Edit **`coil_mold_common.py`** only, then:

```bash
cd pipeline_gz
python run_coil_mold_pipeline.py
```

Outputs: `../resultados/pipeline/Gz_tk25000_lvl26/` (or `…(2)` on re-run).
