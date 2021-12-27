from typing import Callable
from matplotlib.widgets import Slider, Button
from matplotlib import pyplot as plt


class ValueSlider:
    """Wrapper class for managing a `matplotlib` slider and it's state."""
    slider: Slider
    button: Button
    label: str
    sliderCallback: Callable[[str, float], any]
    toggleCallback: Callable[[str, bool], any]
    active: bool

    def __init__(self, label: str, slider_ax: plt.Axes, button_ax: plt.Axes, sliderCallback: Callable[[str, float], any], toggleCallback: Callable[[str, bool], any], default: float = .0, min_max: float = 1) -> None:
        self.slider = Slider(ax=slider_ax, label=label, valmin=-min_max, valmax=min_max, valinit=0)
        self.button = Button(button_ax, 'On')
        self.button.on_clicked(lambda _: self.toggle())
        self.active = True
        self.label = label
        self.toggleCallback = toggleCallback
        self.sliderCallback = sliderCallback
        self.slider.set_val(default)
        self.slider.on_changed(self.onSliderChanged)

    def onSliderChanged(self, new_value: float):
        """Default on_changed callback."""
        self.sliderCallback(self.label, new_value)

    def setSliderValue(self, new_value: float):
        """Sets the value of the slider."""
        self.slider.set_val(new_value)


    def onToggleChanged(self, new_value: bool):
        """Default on_changer callback."""
        self.toggleCallback(self.label, new_value)

    def setToggle(self, new_value: bool):
        """Sets the value of the toggle."""
        self.active = new_value
        self._updateToggle()
        self.onToggleChanged(self.active)

    def getToggle(self):
        """Gets the current toggle value."""
        return self.active

    def toggle(self):
        """Activates or deactivates the slider, sending the appropriate callback."""
        self.active = not self.active
        self._updateToggle()
        self.onToggleChanged(self.active)

    def _updateToggle(self):
        self.button.label.set_text('On' if self.active else 'Off')