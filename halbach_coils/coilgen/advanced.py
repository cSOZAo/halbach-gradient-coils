"""Supported advanced controls, with the existing pipeline defaults.

Geometry stays in Config's existing cylinder/target/winding blocks. Only
additional pyCoilGen engine arguments belong to AdvancedConfig.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass


@dataclass
class AdvancedConfig:
    gauss_order: int = 2
    iteration_num_mesh_refinement: int = 0
    level_set_method: str = 'primary'
    cut_plane_definition: str = 'nearest'
    sf_opt_method: str = 'tikhonov'
    minimize_method: str = 'SLSQP'
    maxiter: int = 100
    tol: float = 1e-6

    def validate(self) -> None:
        for spec in PARAMETERS:
            if spec.path.startswith('advanced.'):
                spec.parse(getattr(self, spec.path.split('.')[1]))

    def to_coilgen_args(self) -> dict:
        """Return engine options only; never override geometry or output flags."""
        parsed = {
            spec.path.split('.')[1]: spec.parse(getattr(self, spec.path.split('.')[1]))
            for spec in PARAMETERS if spec.path.startswith('advanced.')
        }
        return {
            'gauss_order': parsed['gauss_order'],
            'iteration_num_mesh_refinement': parsed['iteration_num_mesh_refinement'],
            'level_set_method': parsed['level_set_method'],
            'cut_plane_definition': parsed['cut_plane_definition'],
            'sf_opt_method': parsed['sf_opt_method'],
            'minimize_method': parsed['minimize_method'],
            'minimize_method_parameters': repr({'tol': parsed['tol']}),
            'minimize_method_options': repr({'disp': True, 'maxiter': parsed['maxiter']}),
        }


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    group: str
    label_en: str
    label_es: str
    default: object
    kind: str
    help_en: str
    help_es: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple = ()
    positive: bool = False

    def label(self, language='en') -> str:
        return self.label_es if language == 'es' else self.label_en

    def help(self, language='en') -> str:
        return self.help_es if language == 'es' else self.help_en

    def parse(self, value, language='en'):
        """Reject invalid entries rather than substituting defaults."""
        label = self.label(language)
        if self.kind == 'choice':
            if value not in self.choices:
                raise ValueError(f'{label}: {", ".join(self.choices)}')
            return value
        if self.kind == 'triple':
            parts = re.split(r'[,\s]+', value.strip()) if isinstance(value, str) else value
            try:
                if len(parts) != 3:
                    raise ValueError
                parsed = [self._number(item, integer=True) for item in parts]
            except (TypeError, ValueError, OverflowError):
                message = 'use tres enteros entre 1 y 101 (p. ej. 7, 7, 7)' if language == 'es' else 'use three integers from 1 to 101 (e.g. 7, 7, 7)'
                raise ValueError(f'{label}: {message}') from None
            return parsed
        try:
            return self._number(value, integer=self.kind == 'int')
        except (TypeError, ValueError, OverflowError):
            kind = ('un entero' if self.kind == 'int' else 'un número finito') if language == 'es' else ('an integer' if self.kind == 'int' else 'a finite number')
            scale = 1000 if self.path == 'winding.cut_width' else 1
            lower = '> 0' if self.positive and self.minimum == 0 else f'>= {self.minimum * scale:g}'
            limits = f'{lower}, <= {self.maximum * scale:g}'
            raise ValueError(f'{label}: {kind}; {limits}') from None

    def _number(self, value, *, integer=False):
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
        if not math.isfinite(number) or (integer and not number.is_integer()):
            raise ValueError
        if self.positive and number <= 0:
            raise ValueError
        if self.minimum is not None and number < self.minimum:
            raise ValueError
        if self.maximum is not None and number > self.maximum:
            raise ValueError
        return int(number) if integer else number


# Labels include display units. cut_width is the sole value shown in mm;
# all Config lengths remain in metres.
PARAMETERS = (
    ParameterSpec('cylinder.n_circ', 'mesh', 'Circumferential divisions', 'Divisiones circunferenciales', 200, 'int',
                  'Cylinder samples around the bore (8–720). More samples resolve winding shape better but increase memory and solve time.',
                  'Muestras alrededor del cilindro (8–720). Más muestras detallan mejor el trazado y aumentan memoria y tiempo.', 8, 720),
    ParameterSpec('cylinder.n_long', 'mesh', 'Longitudinal divisions', 'Divisiones longitudinales', 10, 'int',
                  'Cylinder divisions along the bore (2–100). A finer axial mesh adds design freedom and increases solve cost.',
                  'Divisiones a lo largo del cilindro (2–100). Una malla axial fina agrega libertad de diseño y aumenta el costo del cálculo.', 2, 100),
    ParameterSpec('cylinder.mesh_length_factor', 'mesh', 'Winding length / shell length', 'Largo del bobinado / largo del molde', 0.95, 'float',
                  'Dimensionless factor, greater than 0 and at most 1. The winding mesh uses this fraction of shell length; 0.95 reserves space at both ends.',
                  'Factor sin unidades, mayor que 0 y hasta 1. La malla usa esta fracción del largo del molde; 0.95 reserva espacio en ambos extremos.', 0, 1, positive=True),
    ParameterSpec('advanced.iteration_num_mesh_refinement', 'mesh', 'Mesh refinement passes', 'Pasadas de refinamiento', 0, 'int',
                  'Each pass splits each triangle into four (0–3 passes). Memory can rise sharply; the preview shows the selected effective mesh density.',
                  'Cada pasada divide cada triángulo en cuatro (0–3). La memoria puede aumentar mucho; la vista previa muestra la densidad efectiva seleccionada.', 0, 3),
    ParameterSpec('target.resol_radial', 'roi', 'ROI radial samples', 'Muestras radiales de ROI', 8, 'int',
                  'Radial sampling of the target ellipsoid, including the centre and surface (2–40). More points improve sampling and increase memory and time.',
                  'Muestras radiales del elipsoide objetivo, incluido centro y superficie (2–40). Más puntos mejoran el muestreo y aumentan memoria y tiempo.', 2, 40),
    ParameterSpec('target.resol_angular', 'roi', 'ROI angular samples', 'Muestras angulares de ROI', 28, 'int',
                  'Samples of each spherical angle (4–180). Total stored target points are radial × angular², including repeated centre and pole points.',
                  'Muestras de cada ángulo esférico (4–180). Los puntos almacenados son radial × angular², con repeticiones en centro y polos.', 4, 180),
    ParameterSpec('winding.pot_offset_factor', 'winding', 'Contour potential offset', 'Desfase de potencial de contornos', 0.5, 'float',
                  'Dimensionless offset of contour levels within each potential step (0–1). Moves turn locations; it does not change the requested turn count.',
                  'Desfase sin unidades de los contornos dentro de cada paso de potencial (0–1). Desplaza las vueltas sin cambiar la cantidad solicitada.', 0, 1),
    ParameterSpec('winding.min_loop_signif', 'winding', 'Minimum loop significance [%]', 'Significancia mínima de lazo [%]', 5, 'int',
                  'Threshold (0–100) on 100 × max |loop field| / mean |combined loop field| in the selected component. Higher values discard more loops and can change field quality.',
                  'Umbral (0–100) sobre 100 × máx. |campo del lazo| / promedio |campo combinado| en la componente elegida. Un valor alto descarta más lazos y cambia el campo.', 0, 100),
    ParameterSpec('winding.cut_width', 'winding', 'Interconnection cut width [mm]', 'Ancho de corte de interconexión [mm]', 0.001, 'float',
                  'Positive opening width for connecting contours, displayed in mm (up to 100 mm). Changes winding connections and their field; independent of terminal lead cuts.',
                  'Ancho positivo de apertura al conectar contornos, en mm (hasta 100 mm). Cambia conexiones y su campo; es independiente del corte de los terminales.', 0, 0.1, positive=True),
    ParameterSpec('winding.smooth_factor', 'winding', 'Track smoothing factor', 'Factor de suavizado del trazado', 3, 'int',
                  'Moving-average half-window in path samples (1–101); 1 disables this smoothing. Higher values smooth bends but can move turns.',
                  'Semiventana de promedio móvil, en muestras del trazado (1–101); 1 desactiva este suavizado. Un valor alto suaviza curvas pero puede mover vueltas.', 1, 101),
    ParameterSpec('winding.normal_shift_smooth', 'winding', 'Crossing smoothing [3 integers]', 'Suavizado de cruces [3 enteros]', [7, 7, 7], 'triple',
                  'Three sample counts: shift plateau, ramp and local normal averaging. Enter three integers 1–101. Larger values spread the transition between radial layers.',
                  'Tres cantidades de muestras: meseta del desplazamiento, rampa y promedio local de normales. Use tres enteros 1–101. Un valor alto extiende la transición entre capas radiales.', 1, 101),
    ParameterSpec('advanced.level_set_method', 'winding', 'Contour level method', 'Método de niveles de contorno', 'primary', 'choice',
                  'primary shares the strongest surface step; combined uses the global potential range; independent uses each surface range. They coincide for a single cylinder surface.',
                  'primary comparte el paso de la superficie principal; combined usa el rango global; independent usa el rango de cada superficie. Coinciden para un solo cilindro.', choices=('primary', 'combined', 'independent')),
    ParameterSpec('advanced.cut_plane_definition', 'winding', 'Connection cut orientation', 'Orientación del corte de conexión', 'nearest', 'choice',
                  'nearest selects a nearby loop direction when its normal is parallel to B0; B0 uses a field-aligned fallback. Changes the cut position only in this degenerate case.',
                  'nearest elige la dirección de un lazo cercano cuando su normal es paralela a B0; B0 usa una alternativa alineada al campo. Solo cambia el corte en ese caso degenerado.', choices=('nearest', 'B0')),
    ParameterSpec('advanced.gauss_order', 'solver', 'Gauss integration order', 'Orden de integración de Gauss', 2, 'int',
                  'Quadrature order per triangle (1–8). Each triangle uses order² integration points. Higher orders can improve integration but cost more time.',
                  'Orden de cuadratura por triángulo (1–8). Cada triángulo usa orden² puntos de integración. Un orden alto mejora la integración y requiere más tiempo.', 1, 8),
    ParameterSpec('advanced.sf_opt_method', 'solver', 'Stream-function optimizer', 'Optimizador de función de corriente', 'tikhonov', 'choice',
                  'tikhonov is the existing direct solution. minimize adds a bounded iterative solve initialized with Tikhonov; it can be slower and change the objective tradeoff.',
                  'tikhonov es la solución directa actual. minimize agrega un cálculo iterativo con cotas iniciado con Tikhonov; puede ser más lento y cambiar el equilibrio del objetivo.', choices=('tikhonov', 'minimize')),
    ParameterSpec('advanced.minimize_method', 'solver', 'Iterative method', 'Método iterativo', 'SLSQP', 'choice',
                  'SLSQP or L-BFGS-B, used only when optimizer is minimize. Both support the engine bounds; results may differ. Ignored for tikhonov.',
                  'SLSQP o L-BFGS-B, solo con el optimizador minimize. Ambos admiten las cotas del motor y pueden dar resultados distintos. Se ignora con tikhonov.', choices=('SLSQP', 'L-BFGS-B')),
    ParameterSpec('advanced.maxiter', 'solver', 'Maximum optimizer iterations', 'Iteraciones máximas del optimizador', 100, 'int',
                  'Iteration limit (1–10000), used only by minimize. A higher limit allows more work but does not guarantee convergence.',
                  'Límite de iteraciones (1–10000), solo con minimize. Un límite alto permite más trabajo sin garantizar convergencia.', 1, 10000),
    ParameterSpec('advanced.tol', 'solver', 'Optimizer tolerance', 'Tolerancia del optimizador', 1e-6, 'float',
                  'Positive convergence tolerance (up to 1), used only by minimize. Smaller values request tighter convergence and can increase runtime.',
                  'Tolerancia positiva de convergencia (hasta 1), solo con minimize. Un valor pequeño exige mayor precisión y puede aumentar el tiempo.', 0, 1, positive=True),
)


def config_values(cfg) -> dict:
    """Read catalog values, including defaults for legacy Config callers."""
    values = {}
    for spec in PARAMETERS:
        block, field = spec.path.split('.')
        if block == 'advanced' and not hasattr(cfg, block):
            value = spec.default
        else:
            target = getattr(cfg, block, None)
            if target is None or not hasattr(target, field):
                raise ValueError(f'Missing configuration value: {spec.path}')
            value = getattr(target, field)
        values[spec.path] = copy.deepcopy(value)
    return values


def validate_advanced_config(cfg) -> None:
    values = config_values(cfg)
    for spec in PARAMETERS:
        spec.parse(values[spec.path])


def apply_advanced_values(cfg, values: dict, *, language='en') -> None:
    """Validate all requested changes before mutating a configuration."""
    specs = {spec.path: spec for spec in PARAMETERS}
    unknown = values.keys() - specs.keys()
    if unknown:
        raise ValueError(f'Unknown advanced parameters: {", ".join(sorted(unknown))}')
    parsed = {path: specs[path].parse(value, language) for path, value in values.items()}
    candidate = copy.deepcopy(cfg)
    if not hasattr(candidate, 'advanced'):
        candidate.advanced = AdvancedConfig()
    for path, value in parsed.items():
        block, field = path.split('.')
        setattr(getattr(candidate, block), field, value)
    validate_advanced_config(candidate)
    if not hasattr(cfg, 'advanced'):
        cfg.advanced = AdvancedConfig()
    for path, value in parsed.items():
        block, field = path.split('.')
        setattr(getattr(cfg, block), field, value)


def density_summary(cfg) -> dict:
    """Exact counts for the cylinder and stored spherical target samples."""
    validate_advanced_config(cfg)
    passes = getattr(getattr(cfg, 'advanced', None), 'iteration_num_mesh_refinement', 0)
    scale = 2 ** passes
    n_circ, n_long = cfg.cylinder.n_circ * scale, cfg.cylinder.n_long * scale
    return {
        'vertices': n_circ * (n_long + 1),
        'triangles': 2 * n_circ * n_long,
        'target_points': cfg.target.resol_radial * cfg.target.resol_angular ** 2,
        'winding_length_m': cfg.cylinder.height * cfg.cylinder.mesh_length_factor,
    }
