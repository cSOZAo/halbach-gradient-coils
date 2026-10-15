# Cierre de Gz y cancelación de FastHenry — 14 de octubre de 2026

El usuario aprobó los parámetros avanzados tal como están. Esa revisión queda
cerrada. Se investigaron y corrigieron los otros dos pendientes técnicos.

## Gz

El caso guardado fallaba porque su zona de terminales estaba configurada hacia
Y. Para B0=Y, el bobinado Gz tiene un corredor utilizable cerca de X. Se cambió
solo esa referencia de corte/salida; no se rotó el bobinado, no se cambió el
campo objetivo y no se relajaron límites de curvatura o separación.

Se verificaron dos diseños con B0=+Y, radio exterior 39 mm y malla 200 × 10:

| Caso | Altura / ROI / niveles | R FastHenry | L FastHenry | Pendiente final |
| --- | --- | --- | --- | --- |
| Caso guardado que fallaba | 200 mm / 15 mm / 8 | 0.0173712 Ω | 4.023946 µH | 1.336469 mT/(m·A) |
| Valores iniciales de la interfaz, seleccionando Gz | 170 mm / 20 mm / 12 | 0.0252737 Ω | 7.844540 µH | 2.080715 mT/(m·A) |

Ambos completaron leads, cálculo del campo final, FastHenry real y molde. El
radio mínimo de curva de los terminales fue 7.144 mm y 6.861 mm respectivamente,
por encima del límite de 6 mm. La separación exigida se mantuvo en 0.3 mm; los
chequeos de intersección entre terminales y bobina dieron volumen cero. Las
secciones y los vértices retenidos del bobinado se conservaron.

Al verificar el archivo exportado del segundo molde aparecieron cuatro caras
de área cero por colapso de aristas muy pequeñas a la precisión float32 del
STL. Se corrigió la exportación para eliminar caras degeneradas/duplicadas y
validar una segunda lectura antes de publicar. Los dos STL finales son estancos
y tienen orientación coherente. Si un STL no supera esa validación, se informa
el error y no se publica como resultado correcto.

Estos resultados corresponden a las dos geometrías descritas; otras opciones
de B0 o dimensiones siguen pasando por las comprobaciones del generador.

## FastHenry COM

El programa inicia el ejecutable registrado con `/Embedding` y espera a que
su automatización esté lista. Ese proceso propio se asigna a un trabajo de
Windows con `KILL_ON_JOB_CLOSE`. El mismo mecanismo de contención se comparte
con el trabajador de la GUI, sin depender de que COM responda a `Quit`.

Se comprobaron con FastHenry instalado:

- Extracción normal: mismos R/L que antes, y ningún servidor residual.
- Servidor deliberadamente congelado suspendiendo sus hilos: el timeout cerró
  cliente y servidor y dejó R/L vacíos, con estado de fallo explícito.
- Detener desde el trabajador de la interfaz, con ese bloqueo: terminación en
  0.047 s y 0.062 s en las dos comprobaciones; servidor cerrado y trabajo cancelado.
- Una instancia preexistente de prueba permaneció abierta; el nuevo cálculo se
  informó como no disponible y no se reutilizó ni cerró esa instancia.

Los tiempos medidos son los de esas comprobaciones, no límites de latencia universales.
La inductancia sigue correspondiendo solo al conductor modelado, sin cableado
externo de alimentación o retorno.

La suite completa de la copia principal terminó con **273 pruebas aprobadas**,
además de estas comprobaciones reales del solver y de los dos diseños Gz.

## Trabajo pendiente

Quedan el diseño en lote de Gx/Gy/Gz y los encajes entre capas. Para estos
últimos se revisarán los ejemplos de Fusion antes de fijar la geometría.
