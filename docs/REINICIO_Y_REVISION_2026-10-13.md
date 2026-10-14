# Reinicio desde GitHub y revisión — 13 de octubre de 2026

## Base y respaldo

- Base nueva: repositorio https://github.com/cSOZAo/halbach-gradient-coils, commit `474dc01`.
- Carpeta principal nueva: `C:\Proyectos\GradientDesign-principal`.
- Lanzador: `C:\Proyectos\GradientDesign-principal\GradientDesign.bat`.
- Respaldo adicional del trabajo anterior: `C:\Proyectos\GradientDesign-respaldo-modelo-20261013`, commit original `a276cd6`, con historial, código, documentación, archivos no versionados y resultados copiados.
- Se verificó igualdad SHA-256 de los 283 archivos versionados y no ignorados del original, incluidos los tres archivos aún no guardados en Git.
- La carpeta original `C:\Proyectos\GradientDesign` permanece íntegra, con todos sus resultados, temporales y entorno. No se migraron los cambios del modelo anterior a la base limpia.
- La duplicación de temporales agotó el espacio libre durante el respaldo. Se eliminó solo la copia nueva de `tmp`; el original conserva esos temporales. El respaldo adicional también omite `.venv` y caches.

## Plan ejecutado

1. Preservar el trabajo anterior y clonar GitHub.
2. Incorporar B0 seleccionable, ejes correctos y gráficos persistentes.
3. Incorporar una primera versión revisable de parámetros avanzados, conservando los valores iniciales publicados.
4. Verificar física, interfaz, presets y una ejecución real integrada.
5. Registrar defectos heredados sin corregirlos fuera del alcance.
6. Dejar diseño en lote y encajes para la siguiente etapa, después de revisar ejemplos de Fusion.

## Cambios disponibles

**B0 y componente del campo.** El menú superior permite ±X/±Y/±Z, con +Y por defecto. Objetivo, optimización, significancia de contornos, polaridades, errores y métricas usan `B_parallel = B · B0_hat`. Gx/Gy/Gz son derivadas físicas según X/Y/Z. CLI: `--b0=+y` o `--b0=-x`. Los presets conservan B0.

**Visor.** Muestra X/Y/Z, flecha B0 firmada y componente seleccionada. El cilindro aparece a lo largo de X, como la geometría generada; la comprobación de ROI elipsoidal también usa ese eje. Conserva la orientación de cámara al cambiar parámetros.

**Gráficos.** Se guardan tres PNG y un NPZ de datos numéricos: objetivo/generado a 1 A, linealidad/residuo y campo/error espacial muestreado. El objetivo no se ajusta artificialmente a la pendiente generada. El visor tiene selector y herramientas de Matplotlib. El menú Resultados permite reabrir diseños con su B0 original.

**Parámetros avanzados.** Panel colapsable con malla, ROI, trazado y cálculo; ayuda EN/ES al pasar el mouse, unidades, validación de valores y resumen de densidad. Incluye subdivisiones y refinamiento, largo de bobinado, muestreo ROI, desfase y significancia de contornos, corte, suavizados, método de niveles, orientación de cortes, cuadratura y optimización. La vista geométrica usa muestras reducidas para responder rápido; no predice el nuevo campo sin ejecutar el diseño. La revisión de qué variables conviene mantener continúa pendiente con el usuario.

### Diferencia física importante respecto de la base

La base publicada decía B0=+Y, pero el motor optimizaba Bz y permutaba nombres de gradiente sin transformar realmente los vectores. Se corrigió dentro del punto 1. Los nuevos diseños pueden diferir de los anteriores aunque se llamen igual. Véase `B0_FRAME.md`. También se corrigió la selección de referencia del plano de corte para B0 negativo, necesaria para soportar las direcciones nuevas.

## Problemas heredados encontrados, sin corregir

| Prioridad | Problema y evidencia | Consecuencia / propuesta posterior |
| --- | --- | --- |
| Alta | Cada pestaña tiene su `WorkerRunner`; `gui/runner.py` redirige stdout global y `coilgen/gradient.py::_chdir` cambia el directorio global. No hay exclusión entre pestañas. | Ejecutar dos trabajos simultáneos puede mezclar directorios objetivo y logs. Resolver con exclusión global de trabajos o procesos aislados. Mientras tanto, ejecutar un trabajo a la vez. |
| Media | `pyCoilGen/sub_functions/evaluate_field_errors.py` calcula corriente óptima con `mean(target/layout)`. Campo exacto `[0,1,2]` produce `0/0` y NaN. | Esa métrica puede ser inválida aunque el campo sea correcto. Propuesta: ajuste por mínimos cuadrados con comprobación de campo nulo. |
| Media | `coilgen/fasthenry.py` declara `DEFAULT_WINDOWS_PATH` pero no lo añade a los candidatos. FastHenry está instalado aquí en esa ruta y la búsqueda automática devuelve vacío. | No aparecen métricas FastHenry pese a estar instalado. Agregar la ruta predeterminada y una selección explícita de ejecutable en una etapa posterior. |
| Limitación | `run_pipeline.py` calcula métricas y campo en `run_gradient`, antes de `run_leads`. | Los gráficos describen el bobinado de CoilGen, sin el efecto del corte/terminales añadidos después. Recalcular Biot–Savart y métricas eléctricas sobre el conductor final sería una etapa separada. |
| Media | `_seed_wire_from_previous_run` / `_ensure_wire_exists` reutilizan STL por nombre/eje/Tikhonov/niveles, sin comprobar radios, ROI, conductor o demás parámetros. | Se puede combinar un bobinado anterior con parámetros incompatibles. Comparar un snapshot completo de configuración y versión del modelo físico antes de reutilizarlo. |
| Media | La rama iterativa en `stream_function_optimization.py` utiliza `result.x` sin comprobar `result.success`. | Puede continuar con una solución no convergida. Revisar estado y residuo antes de aceptarla. Tikhonov sigue siendo el valor inicial. |

También aparecieron advertencias existentes en mallas gruesas: desviación radial de la geometría respecto de la envolvente analítica, promedios vacíos y puntos de corte coincidentes en posprocesamiento. No se cambiaron sus algoritmos ni se considera que una ejecución sin excepción valide la fabricación.

## Propuesta para lote y encajes — pendiente

Primero añadir un lote secuencial Gx/Gy/Gz con B0, ROI y marco XYZ compartidos, radios anidados definidos por capa y carpetas de salida separadas. Guardar una configuración común y mostrar juntos los tres resultados para revisar holguras, salidas de cables e interferencias. Mantener una referencia angular única: alrededor del bore X, ángulo cero en +Y y ángulo positivo hacia +Z. No rotar una bobina para acomodar los encajes, porque alteraría su gradiente físico.

Después generar en los extremos libres del trazado una referencia axial de asiento y dos guías asimétricas pin/alojamiento que impidan rotación y montaje invertido. Las claves deben transmitirse entre capas desde la misma referencia, con holgura de impresión ajustable y espacio para los terminales. Comprobar interferencias con ranuras y conductores antes de exportar los STL.

Antes de fijar la geometría se revisarán los ejemplos de Fusion: tipo de encaje, orden de capas, zona disponible, orientación de impresión y tolerancias reales. Aún no se implementaron lote ni encajes.

## Validación

- Pruebas de las 18 combinaciones de eje de gradiente y B0, signos, optimización, Biot–Savart y métricas.
- Seis ejecuciones reales con malla pequeña exportaron STL y métricas para las seis direcciones de B0.
- Ejecución integrada de Pipeline con Gx/B0 +Y, ROI elipsoidal y orden de Gauss 3: tres PNG y NPZ que se reabre; pendiente 0.507339 mT/(m·A), valor de prueba con malla gruesa.
- Pruebas de interfaz Tk, idioma español, parámetros, ayuda, presets y actualización de vista previa.
- La validación final del clon principal se registra al cierre de esta entrega. No se ejecutó una campaña completa de fabricación ni una simulación electromagnética independiente.
