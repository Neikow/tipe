from typing import Union
from matplotlib import pyplot as plt

from matplotlib.gridspec import SubplotSpec
from model.scope import Data

from model.types import Uncertainty


class Curve:
    """Curve wrapper object.

    Handles data and uncertainties for a given `pyplot` graph."""

    has_data: bool

    x_data: list[float]
    y_data: list[float]
    x_unc: list[float]
    y_unc: list[float]
    labels: list[str]

    ids: dict[str, int]

    x_label: str
    y_label: str

    ax: plt.Axes

    uncertainty: bool

    def __init__(self, X: str, Y: str, use_uncertainty: bool = False) -> None:
        self.has_data = False
        self.x_label = X
        self.y_label = Y
        self.uncertainty = use_uncertainty

        self.labels = []
        self.ids = {}
        self.x_data = []
        self.y_data = []
        self.x_unc = []
        self.y_unc = []

    def populate(self, data: Data, uncert: Uncertainty = None, ignore: list[str] = None):
        """Populates axes with `data` and `uncert`."""

        for var in [self.x_label, self.y_label]:
            if var == 'graph':
                assert False, 'Use `Helper.traceMeasurementGraphs` to trace measurement graphs.'
            elif var not in list(data.values())[0]:
                assert False, f'Error while plotting the graph, unknown variable: "{var}"'

        for i, key in enumerate(data.keys()):
            if ignore and key in ignore:
                continue

            self.labels.append(key)
            self.ids[key] = i

            for (axis, label) in [(self.x_data, self.x_label), (self.y_data, self.y_label)]:
                value = data[key][label]
                if isinstance(value, float):
                    axis.append(value)

            if self.uncertainty and uncert is not None:
                for (var, unc) in [(self.x_label, self.x_unc), (self.y_label, self.y_unc)]:
                    try:
                        unc.append(uncert[key][var])
                    except KeyError:
                        unc.clear()
                        break

        self.has_data = True
        return self.plot()

    def update(self, key: str, data: dict[str, float], uncert: dict[str, float] = None):
        """Updates the `data` and `uncert` of an already populated graph."""
        if key not in self.ids:
            assert False, 'Unknown key.'

        index = self.ids[key]

        self.x_data[index] = data[self.x_label]
        self.y_data[index] = data[self.y_label]

        if self.uncertainty and len(self.x_unc) != 0 and len(self.y_unc) != 0 and uncert is not None:
            self.x_unc[index] = uncert[self.x_label]
            self.y_unc[index] = uncert[self.y_label]

        return self.plot()

    # pylint: disable = unsubscriptable-object
    def plot(self) -> dict[str, Union[list[float], list[str], bool, str]]:
        """Returns a usable `dict` representing the curve."""

        assert self.has_data, 'Can\'t plot a curve without data.'
        return {'x': self.x_data,
                'y': self.y_data,
                'labels': self.labels,
                'unc': self.uncertainty,
                'ux': self.x_unc,
                'uy': self.y_unc,
                'x_label': self.x_label,
                'y_label': self.y_label}

    def traceCurve(self, spec: SubplotSpec, data: Data, uncert: Uncertainty = None):
        self.ax = plt.subplot(spec)
        self.ax.scatter(self.x_data, self.y_data)

    def updateCurve(self, data: Data, uncert: Uncertainty = None):
        ...

    def getAxis(self):
        return self.ax
