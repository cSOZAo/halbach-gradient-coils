"""3D orientation arrows sized relative to the visible coordinate range."""

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class ViewScaledArrow(FancyArrowPatch):
    """Keep an orientation marker legible while the user zooms the geometry."""

    def __init__(self, axes, origin, direction, *, fraction, label, color,
                 linewidth=1.4, fontsize=8, head_size=9):
        super().__init__((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=head_size,
                         color=color, linewidth=linewidth, shrinkA=0, shrinkB=0)
        self.origin = np.asarray(origin, dtype=float)
        vector = np.asarray(direction, dtype=float)
        self.direction = vector / np.linalg.norm(vector)
        self.fraction = fraction
        self.label = axes.text(*origin, label, color=color, fontsize=fontsize,
                               fontweight='bold')
        axes.add_artist(self)

    def endpoints(self):
        spans = [abs(high - low) for low, high in (
            self.axes.get_xlim3d(), self.axes.get_ylim3d(), self.axes.get_zlim3d())]
        return self.origin, self.origin + self.direction * max(spans) * self.fraction

    def _project(self):
        origin, end = self.endpoints()
        self.label.set_position_3d(origin + (end - origin) * 1.1)
        x, y, z = proj3d.proj_transform(*np.column_stack((origin, end)), self.axes.get_proj())
        self.set_positions((x[0], y[0]), (x[1], y[1]))
        return float(min(z))

    def do_3d_projection(self):
        return self._project()

    def draw(self, renderer):
        self._project()
        super().draw(renderer)
