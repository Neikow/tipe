"""Types"""
from typing import Callable


# pylint: disable = unsubscriptable-object
Uncertainty = dict[str, dict[str, float]]
ScaleFunction = Callable[..., float]
