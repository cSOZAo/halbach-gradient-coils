# Halbach Gradient Coils

Herramientas reproducibles para diseñar bobinas de gradiente (`Gx`, `Gy`, `Gz`) para un escáner MRI de bajo campo basado en Halbach. El repositorio incluye el motor base **pyCoilGen** y la aplicación específica del proyecto, por lo que un clon contiene todo el código fuente necesario.

## Estructura

```text
halbach-gradient-coils/
├── pyCoilGen/       # motor de cálculo base (incluido en el repositorio)
├── halbach_coils/   # aplicación Halbach: CLI, GUI, assets y código de diseño
├── tests/           # pruebas del motor pyCoilGen
├── data/, docs/, examples/, utilities/
└── pyproject.toml
```

`halbach_coils/` se llamaba anteriormente `pruebas/`. Los resultados de simulación se escriben bajo `halbach_coils/resultados/` y no se versionan.

## Instalación (Windows)

Se recomienda Python 3.11. Desde la raíz del repositorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-project.txt
```

FastHenry2 es opcional: sin él se calculan los diseños, pero las métricas de resistencia/inductancia quedan como `n/a`.

## Uso

```powershell
cd halbach_coils
python run_gui.py
python run_pipeline.py --axis y --tikhonov 2500 --levels 26 --layer 2
```

Consulta [halbach_coils/README.md](halbach_coils/README.md) para los parámetros físicos, la arquitectura del flujo y ejemplos adicionales.

## Verificación

```powershell
python -m pytest -q
```
