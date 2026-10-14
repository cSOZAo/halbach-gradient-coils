"""Arrange generated files by stage, preserving names and reusable sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


def result_subfolder(name: str) -> str | None:
    """Only recognize application outputs; leave unrelated user files alone."""
    if '_field_' in name and name.endswith('.png'):
        return 'Graficos' if '_with_leads' in name else 'Graficos/CoilGen'
    if '_shell_' in name and name.endswith('.stl'):
        return 'Molde'
    if any(tag in name for tag in ('_with_leads', '_coil_open', '_leads_only')):
        if name.endswith(('.stl', '.npz', '.json')):
            return 'Leads'
    if (('_wire_' in name and name.endswith(('.stl', '_design.json', '_lead_reference.npz')))
            or name.endswith(('_field_data.npz', '_metrics.txt'))):
        return 'Bobina'
    return None


def relocate_paths(value, moves: dict[str, str]):
    """Update returned artifacts and stored JSON references after moving files."""
    if isinstance(value, dict):
        return {key: relocate_paths(item, moves) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [relocate_paths(item, moves) for item in value]
    if isinstance(value, str):
        normalized = os.path.normcase(value)
        for old, new in moves.items():
            if normalized == os.path.normcase(old):
                return new
            if normalized.startswith(os.path.normcase(old + os.sep)):
                return new + value[len(old):]
    return value


def organize_outputs(output_dir, *, replace=False) -> dict[str, str]:
    """Move only known outputs, checking all destinations before any mutation.

    STL sidecars stay beside their STL. No numerical arrays are rewritten, so
    provenance hashes remain valid. Existing flat directories are supported.
    """
    root = Path(output_dir).resolve()
    if not root.is_dir():
        return {}
    planned = []
    for source in root.iterdir():
        if source.is_symlink():
            continue
        if source.is_file():
            group = result_subfolder(source.name)
            if group is None:
                continue
            destination = root / group / source.name
        elif source.is_dir() and (source.name.startswith('final_fasthenry_')
                                  or source.name in ('target_fields', 'debug')):
            if not any(source.iterdir()):
                source.rmdir()
                continue
            destination = root / 'Diagnostico' / source.name
            number = 2
            while destination.exists():
                destination = root / 'Diagnostico' / f'{source.name}({number})'
                number += 1
        else:
            continue
        assert source.resolve().is_relative_to(root)
        assert destination.resolve().is_relative_to(root)
        if destination.exists() and not replace:
            raise FileExistsError(f'Result already exists; refusing to overwrite: {destination}')
        planned.append((source, destination))
    moves = {str(source): str(destination) for source, destination in planned}
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            source.replace(destination)
        else:
            shutil.move(str(source), str(destination))
    # JSON files contain filenames/parameters plus a few absolute artifact paths.
    for group in ('Bobina', 'Leads', 'Diagnostico'):
        for filename in (root / group).rglob('*.json'):
            try:
                previous = json.loads(filename.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            updated = relocate_paths(previous, moves)
            if updated != previous:
                filename.write_text(json.dumps(updated, indent=2, ensure_ascii=False,
                                                allow_nan=False) + '\n', encoding='utf-8')
    return moves


def write_result_summary(output_dir, *, stem='', error=None) -> str:
    """One visible entry point, with final FastHenry values rather than early n/a."""
    root = Path(output_dir).resolve()
    if not stem:
        marker = root / '.active_project_stem'
        stem = marker.read_text(encoding='utf-8').strip() if marker.is_file() else ''
    matches = list((root / 'Leads').glob(f'{stem}_wire_0_with_leads*_final_metrics.json')) if stem else []
    metrics_path = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
    lines = ['RESULTADO DEL DISEÑO', f'Diseño: {stem or root.name}', '']
    if error:
        lines.extend(['Ejecución incompleta: ' + str(error), ''])
    if metrics_path:
        result = json.loads(metrics_path.read_text(encoding='utf-8'))
        fh = result['fasthenry']
        lines.extend([
            'Conductor final: bobinado retenido y ambos terminales.',
            f"Gradiente: G{result['gradient_axis'].upper()} | B0: {result['b0_direction'].upper()}",
            f"Pendiente: {result['slope_mT_per_m_per_A']:.7g} mT/(m·A)",
            f"Longitud: {result['wire_length_m']:.7g} m",
            f"Resistencia analítica: {result['resistance_ohm_analytic']:.7g} Ω",
            f"FastHenry: {fh['status']}",
        ])
        if fh['status'] == 'success':
            lines.extend([f"Resistencia FastHenry: {fh['resistance_ohm']:.7g} Ω",
                          f"Inductancia FastHenry: {fh['inductance_H'] * 1e6:.7g} µH"])
        elif fh.get('reason'):
            lines.append('Motivo: ' + fh['reason'])
        lines.extend(['Los cables externos de alimentación y retorno no están incluidos.',
                      'Métricas completas: ' + str(metrics_path.relative_to(root)), ''])
    else:
        if list((root / 'Bobina').glob('*_wire_*.stl')):
            lines.extend(['Solo hay resultados del bobinado de CoilGen.',
                          'FastHenry del conductor final se calcula después de generar los leads.', ''])
        elif list((root / 'Molde').glob('*.stl')):
            lines.extend(['Etapa de molde completada.', 'Esta etapa no calcula FastHenry.', ''])
        else:
            lines.extend(['No hay métricas finales disponibles.', ''])
    lines.extend([
        'ARCHIVOS',
        'Molde/: STL del molde listo para inspección o fabricación.',
        'Leads/: conductor final, piezas de terminales, campo y métricas finales.',
        'Graficos/: gráficos finales; CoilGen/ conserva los anteriores a los leads.',
        'Bobina/: STL original, parámetros y referencias necesarias para reutilizarlo.',
        'Diagnostico/: entradas de cálculo y evidencia del solver.',
        '',
        'Las métricas de Bobina/ son anteriores a los leads. Consulte este resumen',
        'y el JSON final de Leads/ para los valores finales de resistencia e inductancia.',
    ])
    path = root / 'Resumen.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(metrics_path) if metrics_path else ''
