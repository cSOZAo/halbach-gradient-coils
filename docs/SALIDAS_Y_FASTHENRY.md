# Salidas y FastHenry — 14 de octubre de 2026

La prueba del Escritorio (`test`) sí había completado FastHenry correctamente:
el archivo `Zc2a01df25192e.mat` contiene la impedancia calculada a 1 Hz, y el
JSON final registra R = 0.0362738 Ω y L = 14.7778548 µH. El TXT de métricas
visible correspondía al bobinado anterior a los leads y mostraba `n/a`.
La presentación hacía difícil encontrar el resultado final.

Se repitió la extracción con el servidor COM instalado, sobre la misma
trayectoria de 3963 puntos (2683 en el solver), y se obtuvieron los mismos
valores. Esta comprobación usa FastHenry real, no un solver simulado. La
limitación de cancelación de un servidor COM externo bloqueado sigue vigente.

## Organización

Cada salida tiene un `Resumen.txt` con los valores finales y las carpetas:

- `Molde`: STL del molde.
- `Leads`: conductor final, terminales separados, trayectoria numérica, campo
  final, métricas y reporte geométrico.
- `Graficos`: PNG del conductor final; `CoilGen` conserva los anteriores.
- `Bobina`: STL original, configuración verificada, referencia de secciones,
  datos de campo y métricas anteriores a los terminales.
- `Diagnostico`: entradas objetivo y evidencia de FastHenry.

Los archivos vinculados por hashes se conservan juntos y sin modificar sus
datos binarios. Las referencias de rutas de los JSON y del visor se actualizan.
Las carpetas antiguas sin esta estructura siguen siendo compatibles. Los
nombres de iteraciones nuevas reservan también los resultados ya organizados
para evitar sobrescribirlos. Los archivos ajenos al programa quedan intactos.

La ventana de fin de ejecución y el visor de campo final muestran R en Ω y L
en µH. Un fallo o una indisponibilidad aparece con su estado y motivo. El TXT
de `Bobina` se identifica explícitamente como anterior a los leads y remite al
resumen final.

La inductancia corresponde al conductor modelado; los cables externos de
alimentación y retorno quedan fuera del cálculo.

## Verificación adicional

También se ejecutó el pipeline completo con los valores iniciales de la
interfaz: Gx, B0=+Y, radio 39 mm, altura 170 mm, ROI 20 mm, Tikhonov 2500 y
12 niveles. FastHenry real obtuvo R = 0.0362719 Ω y L = 14.7768680 µH.
El molde es estanco y las salidas organizadas conservan un estado de diseño
válido. La diferencia con la prueba del Escritorio corresponde a diseños
distintos; aquella usaba B0=-Y.

Las flechas de los ejes y B0 ahora ajustan su longitud al rango visible durante
el zoom. Sus puntas mantienen un tamaño en pantalla y las etiquetas acompañan
la nueva longitud. Se aplica también a B0 en los gráficos espaciales.
