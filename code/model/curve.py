from typing import Union

class Curve:
    """Curve wrapper object.
    
    Handles data and uncertainties for a given `pyplot` graph."""

    hasData: bool

    X: list[float]
    Y: list[float]
    X_unc: list[float]
    Y_unc: list[float]
    labels: list[str]

    ids: dict[str, int]

    x_label: str
    y_label: str

    uncertainty: bool

    def __init__(self, X: str, Y: str, use_uncertainty: bool = False) -> None:
        self.hasData = False
        self.x_label = X
        self.y_label = Y
        self.uncertainty = use_uncertainty

        self.labels = []
        self.ids = {}
        self.X = []
        self.Y = []
        self.X_unc = []
        self.Y_unc = []
    
    def populate(self, data: dict[str, dict[str, float]], uncert: dict[str, dict[str, float]] = {}):
        """Populates axes with `data` and `uncert`."""

        for var in [self.x_label, self.y_label]:
            if var == 'graph':
                assert False, f'Use `Helper.traceMeasurementGraphs` to trace measurement graphs.'
            elif not var in list(data.values())[0]:
                assert False, f'Error while plotting the graph, unknown variable: "{var}"'

        for i, id in enumerate(data.keys()):
            self.labels.append(id)
            self.ids[id] = i

            for (ax, var) in [(self.X, self.x_label), (self.Y, self.y_label)]:
                ax.append(data[id][var])
                
            if self.uncertainty:
                for (var, unc) in [(self.x_label, self.X_unc), (self.y_label, self.Y_unc)]:
                    try: unc.append(uncert[id][var])
                    except: unc.clear(); break

        self.hasData = True
        return self.plot()
    
    def update(self, id: str, data: dict[str, float], uncert: dict[str, float] = {}):
        """Updates the `data` and `uncert` of an already populated graph."""
        if not id in self.ids.keys():
            assert False, 'Unknown id.'
        
        index = self.ids[id]

        self.X[index] = data[self.x_label]
        self.Y[index] = data[self.y_label]

        if self.uncertainty and len(self.X_unc) != 0 and len(self.Y_unc) != 0:
            self.X_unc[index] = uncert[self.x_label]
            self.Y_unc[index] = uncert[self.y_label]

        return self.plot()


    def plot(self) -> dict[str, Union[list[float], list[str], bool, str]]:
        """Returns a usable `dict` representing the curve."""

        assert self.hasData, 'Can\'t plot a curve without data.'
        return {'x': self.X,
                'y': self.Y,
                'labels': self.labels,
                'unc': self.uncertainty,
                'ux': self.X_unc,
                'uy': self.Y_unc,
                'x_label': self.x_label,
                'y_label': self.y_label }
        
