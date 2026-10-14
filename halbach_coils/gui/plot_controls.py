"""Mouse-wheel zoom for embedded matplotlib 3D axes."""


def zoom_3d(event):
    axes = event.inaxes
    if axes is None or not hasattr(axes, 'get_zlim3d'):
        return
    step = getattr(event, 'step', 0)
    if not step:
        step = 1 if event.button == 'up' else -1 if event.button == 'down' else 0
    if not step:
        return
    factor = 0.8 ** max(-4, min(4, step))
    for get_limits, set_limits in (
        (axes.get_xlim3d, axes.set_xlim3d),
        (axes.get_ylim3d, axes.set_ylim3d),
        (axes.get_zlim3d, axes.set_zlim3d),
    ):
        low, high = get_limits()
        center = (low + high) / 2
        half = max(1e-6, min(1e9, (high - low) * factor / 2))
        set_limits(center - half, center + half)
    event.canvas.draw_idle()


def enable_3d_zoom(canvas):
    cancel_draw_on_destroy(canvas)
    return canvas.mpl_connect('scroll_event', zoom_3d)


def cancel_draw_on_destroy(canvas):
    """Remove a queued Tk draw before destroying its canvas widget."""
    import tkinter as tk

    widget = canvas.get_tk_widget()

    def destroyed(event):
        if event.widget is not widget:
            return
        pending = getattr(canvas, '_idle_draw_id', None)
        if pending is not None:
            try:
                widget.after_cancel(pending)
            except tk.TclError:
                pass
            canvas._idle_draw_id = None

    widget.bind('<Destroy>', destroyed, add='+')
