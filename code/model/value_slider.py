from typing import Callable
from matplotlib.widgets import Slider, Button
from matplotlib import pyplot as plt

class ValueSlider:
    """Wrapper class for managing a `matplotlib` slider and it's state."""
    slider: Slider
    button: Button
    label: str
    slider_callback: Callable[[str, float], any]
    toggle_callback: Callable[[str, bool], any]
    active: bool

    # pylint: disable = too-many-arguments
    def __init__(self, label: str, slider_ax: plt.Axes, button_ax: plt.Axes, slider_callback: Callable[[
                 str, float], any], toggle_callback: Callable[[str, bool], any], default: float = .0, min_max: float = 1) -> None:
        self.slider = Slider(ax=slider_ax, label=label, valmin=-min_max, valmax=min_max, valinit=0)
        self.button = Button(button_ax, 'On')
        self.button.on_clicked(lambda _: self.toggle())
        self.active = True
        self.label = label
        self.toggle_callback = toggle_callback
        self.slider_callback = slider_callback
        self.slider.set_val(default)
        self.slider.on_changed(self.on_slider_changed)

    def on_slider_changed(self, new_value: float):
        """Default on_changed callback."""
        self.slider_callback(self.label, new_value)

    def set_slider_value(self, new_value: float):
        """Sets the value of the slider."""
        try:
            self.slider.set_val(new_value)
        except AttributeError as e:
            print('putain de merde', e)

    def on_toggle_changed(self, new_value: bool):
        """Default on_changer callback."""
        self.toggle_callback(self.label, new_value)

    def set_toggle(self, new_value: bool):
        """Sets the value of the toggle."""
        self.active = new_value
        self._update_toggle()
        self.on_toggle_changed(self.active)

    def get_toggle(self):
        """Gets the current toggle value."""
        return self.active

    def toggle(self):
        """Activates or deactivates the slider, sending the appropriate callback."""
        self.active = not self.active
        self._update_toggle()
        self.on_toggle_changed(self.active)

    def _update_toggle(self):
        self.button.label.set_text('On' if self.active else 'Off')
