# Correcciones y validación — 14 de octubre de 2026

Actualización posterior: la extracción COM real de FastHenry ya fue verificada.
Véase [la organización de salidas y los resultados](SALIDAS_Y_FASTHENRY.md), que
actualiza ese pendiente y documenta el ajuste de flechas al hacer zoom.

La carpeta de trabajo principal es `C:\Proyectos\GradientDesign`. La iteración
anterior permanece en `C:\Proyectos\GradientDesign-Overengineered` y los respaldos
en `C:\Proyectos\GradientDesign-Backups`.

## Cambios solicitados

- El cilindro usa Z longitudinal en cálculo, visor y STL. X/Y son radiales.
  B0 conserva su selector de seis direcciones y la componente optimizada es
  `B · B0_hat`; Gx/Gy/Gz derivan respecto de las coordenadas físicas X/Y/Z.
- Los parámetros avanzados tienen un botón para restaurar sus valores por
  defecto, preservando geometría principal, material y B0.
- Se retiraron de la interfaz los presets y los parámetros avanzados de leads.
  El barrido conserva su configuración interna para recuperar una fila previa.
- La rueda del mouse permite hacer zoom en los visores 3D.
- Detener cancela el proceso de cálculo y sus descendientes, incluso durante
  operaciones numéricas largas. Los tres paneles comparten un único trabajo
  activo. Cerrar la aplicación también cancela ese trabajo. Los archivos
  parciales quedan identificados mediante `.job_status.json`.

## Leads y resultados finales

La corrección de la iteración anterior estaba en `experimental/leads_surface.py`
y su adaptador; su `leads.py` de producción coincidía con la versión nueva.
Se trasladó el método geométrico necesario, sin incorporar el resto de aquella
arquitectura. Conserva las secciones originales del canal, evita aplanarlas
radialmente y valida curvatura, separación y mallas antes de publicar los STL.

Cada diseño guarda una referencia compacta de su trayectoria y secciones. El
STL y sus archivos asociados quedan vinculados mediante hashes. La ruta final
conserva el sentido de corriente de CoilGen, incluyendo el tramo retenido y
ambos terminales. A partir de ella se recalculan gráficos, pendiente, errores,
longitud, resistencia analítica y, cuando está disponible, FastHenry. Los cables
externos de alimentación y retorno quedan fuera del modelo; la inductancia
extraída corresponde solo al conductor modelado.

Los resultados de CoilGen anteriores a los leads se mantienen identificados
como tales. Los gráficos finales usan archivos separados con sufijo
`_with_leads..._field_*` y las métricas finales se guardan en JSON.

## Problemas heredados corregidos

- Corriente óptima por mínimos cuadrados, evitando dividir punto a punto entre
  campos que pueden valer cero.
- Verificación de convergencia y valores finitos del optimizador; límites
  correctos cuando toda la función de corriente tiene signo negativo.
- Detección de la instalación estándar de FastHenry en Windows, extracción
  acotada en tiempo y estados explícitos de fallo o indisponibilidad.
- Resistencia y campo del conductor final, incluyendo terminales; la resistencia
  analítica se distingue del resultado del solver.
- Reutilización de STL condicionada a configuración, coordenadas e integridad
  compatibles. Un archivo antiguo sin referencia y estado debe regenerarse.
- Los trabajos de la interfaz se ejecutan en procesos independientes; el cambio
  de directorio y la salida del cálculo no interfieren entre pestañas.
- El flujo normal deja de guardar las matrices internas completas en enormes
  archivos de solución `.npy`. Conserva STL, parámetros, métricas, gráficos y
  referencias compactas. La API de bajo nivel permite seguir guardando la
  solución completa cuando se solicita.

## Comprobaciones

- Suite completa: **260 pruebas aprobadas**, con dos advertencias de deprecación
  de SciPy. Comprobación final de la interfaz: **35 pruebas aprobadas**.
- Trabajo real de la GUI: configuración serializada, despacho del pipeline,
  resultado recibido y exclusión entre los tres botones de ejecución.
- Diseños reales Gx y Gy: malla 200 × 10, radio exterior 39 mm, longitud 200 mm,
  ROI de radios 15 mm, ocho niveles y B0=+Y. El método corregido generó leads
  y preservó el signo del campo. Pendientes antes/después de leads:
  Gx 3.460797/3.460349 y Gy 2.228514/2.227903 mT/(m·A).
- Flujo actualizado de Gx desde su diseño verificado: leads, recálculo de campo
  y molde cerrado. El molde es estanco y tiene Z longitudinal. Longitud final
  4.962047 m; resistencia analítica 0.026535 Ω; error medio respecto del objetivo
  1.978 %, máximo 7.976 %. FastHenry estuvo desactivado en esta ejecución.
- Se inspeccionaron visualmente el visor en español y los gráficos finales.
  Las pruebas del cliente FastHenry usan un solver simulado, incluidos errores
  y timeout; no equivalen a una extracción COM real.

## Pendientes acordados

1. **Gz:** el caso probado no tiene una ruta aceptada por las restricciones de
   curvatura y separación, incluso con la búsqueda ampliada de la versión
   anterior. Se informa el rechazo y se conserva el diseño. Por indicación del
   usuario, queda para revisar juntos geometría y configuración; no se relajan
   las comprobaciones para forzar una salida.
2. **FastHenry COM:** la instalación usa un servidor externo. El cliente tiene
   timeout y el proceso de cálculo se cancela, pero un servidor COM bloqueado
   podría seguir vivo. La extracción real y ese comportamiento requieren una
   comprobación específica con el solver instalado.
3. **Diseño en lote Gx/Gy/Gz y encajes:** siguen reservados para la siguiente
   etapa, tras revisar los ejemplos de Fusion. La propuesta es fijar una
   referencia angular compartida y añadir encajes asimétricos en zonas libres
   de canales, comprobando interferencias entre las tres capas antes de exportar.

El conjunto de parámetros avanzados puede revisarse después con el usuario,
como estaba previsto.
