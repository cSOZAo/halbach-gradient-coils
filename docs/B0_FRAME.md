# B0 y marco de coordenadas

Los diseños nuevos usan un único marco físico XYZ para la malla, los STL,
los puntos objetivo, los vectores de campo y el visor. El cilindro tiene su eje
longitudinal en **+Z** por defecto. Cambiar B0 no mueve ese cilindro
ni intercambia los radios X/Y/Z de la región objetivo.

B0 puede apuntar a +X, −X, +Y, −Y, +Z o −Z; por defecto apunta a +Y.
Se optimiza la componente firmada `B_parallel = B · B0_hat`. Por tanto:

- Gx es `d(B_parallel)/dx`.
- Gy es `d(B_parallel)/dy`.
- Gz es `d(B_parallel)/dz`.

Cambiar el signo de B0 invierte la componente física objetivo y el sentido
de corriente correspondiente para conservar el signo positivo del gradiente
solicitado. El vector se pasa también a la selección de cortes de CoilGen.
La significancia de los contornos, selección de polaridades, errores de campo,
derivadas y regresión usan la misma proyección. Los campos vectoriales completos
se conservan en el resultado para inspección.

Las métricas incluyen `b0_direction`, `b0_vector`, `coordinate_frame` y la
componente firmada. El estado JSON de cada diseño guarda B0 y la configuración.
En CLI se acepta, por ejemplo, `--axis x --b0=+y` o `--axis z --b0=-x`.

## Compatibilidad con diseños anteriores

El flujo anterior llamaba Gx/Gy/Gz a objetivos internos y/z/x respectivamente,
pero mantenía todos los campos en el marco de la geometría y optimizaba Bz.
Rotar la malla por sí solo no convertía Bz en By. El flujo nuevo corrige esa
inconsistencia como parte de la selección de B0; con B0=+Y los diseños pueden
cambiar respecto de los antiguos. No se deben comparar resultados antiguos y
nuevos únicamente por su nombre Gx/Gy/Gz.

Los archivos antiguos y sus nombres se conservan. La función de permutación
histórica solo se mantiene para localizar STL antiguos. Para reutilizar un STL,
se requieren su estado de diseño y sus archivos asociados, con parámetros y
hashes compatibles. Los diseños antiguos sin ese estado deben regenerarse;
cambiar las etiquetas del visor no convierte su geometría al marco nuevo.
El motor de bajo nivel pyCoilGen conserva +Z como valor por defecto cuando
se llama directamente sin proporcionar `b_0_direction`.

## Verificación

`tests/test_b0_physics.py` comprueba las 18 combinaciones de eje de gradiente y
dirección de B0, la inversión de un operador de campo conocido en el optimizador,
la derivada analítica frente a diferencias finitas de Biot–Savart, los contornos
transversales, la selección de polaridad y la regresión en coordenadas físicas.

Después de generar los leads se calcula de nuevo Biot–Savart sobre la trayectoria
real retenida y los dos terminales, conservando el sentido de corriente de
CoilGen. Los resultados finales excluyen los cables externos de alimentación
y retorno. Véase [la validación y los pendientes](CORRECCIONES_2026-10-14.md).
