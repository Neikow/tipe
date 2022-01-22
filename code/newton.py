import os
from math import ceil, sqrt, inf, tan
import numpy as np
from typing import Callable, Literal
from matplotlib import pyplot as plt
from matplotlib.pyplot import Line2D
from matplotlib.gridspec import SubplotSpec, GridSpec
from matplotlib.widgets import Button, Slider
from inspect import signature
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.integrate import simpson


class Constants:
    """Constants used in computations."""
    # Accélération terrestre.
    g: float = 9.80665      # m.s-2
    # Masse par défaut d'une bille.
    M: float = 6.9E-3       # kg
    # Hauteur du capteur.
    Hc: float = 4.5E-3      # m
    # Pi
    pi: float = 3.141592653589793

    # Incertitude sur la mesure de la hauteur.
    u_h: float = .5E-3       # m
    # Incertitude sur la masse de la bille.
    u_M: float = 0.1E-3
    # Incertitude sur la valeur de la résistance.
    u_r: float = 1


class Uncertainties:
    """Wrapper class for uncertainties calculation.

    Wikipedia: https://fr.wikipedia.org/wiki/Propagation_des_incertitudes"""

    @staticmethod
    def sum(dB: float, dC: float) -> float:
        '''`A = B + C`'''
        return abs(dB + dC)

    @staticmethod
    def product(B: float, C: float, dB: float, dC: float) -> float:
        '''`A = B x C`'''
        return abs(B * dC + C * dB)

    @staticmethod
    def quotient(B: float, C: float, dB: float, dC: float) -> float:
        '''`A = B / C`'''
        assert C != 0, 'Division by 0'
        return abs((dB + (B * dC) / C) / C)

    @staticmethod
    def power(B: float, n: float, dB: float) -> float:
        '''`A = B^n`'''
        return abs(n * dB * (B ** n) / B)


ScaleFunction = Callable[..., float]
ScopeData = dict[str, list[float]]


class Scope:
    """A wrapper class for holding scope data."""
    id: str

    # Labels
    x_label: str
    y_label: str

    # Data
    x_data: list[float]
    y_data: list[float]

    # Plot Axes
    ax_wrapper: plt.Axes    # Wrapper axis
    ax_og: plt.Axes         # Original axis
    ax_sc: plt.Axes         # Scaled axis

    # Plotted Lines
    ln_og: Line2D          # Original plot
    ln_sc: Line2D          # Scaled plot

    # Parameters
    on_same_graph: bool
    plot_original: bool
    plot_scaled: bool

    def __init__(self, id: str, scope_data: ScopeData) -> None:
        keys = list(scope_data.keys())

        self.id = id

        self.x_label = keys[0]
        self.x_data = scope_data[self.x_label]
        self.y_label = keys[1]
        self.y_data = scope_data[self.y_label]

        self.ax_og = None
        self.ax_sc = None

        self.ln_og = None
        self.ln_sc = None

    def setXdata(self, new_data: list[float]):
        """Sets a new value for the X axis."""
        assert len(new_data) == len(self.y_data)
        self.x_data = new_data

    def setYdata(self, new_data: list[float]):
        """Sets a new value for the Y axis."""
        assert len(new_data) == len(self.x_data)
        self.y_data = new_data

    # pylint: disable = unsubscriptable-object, inconsistent-return-statements
    def getAxis(self, axis_type: Literal['wrapper', 'og', 'sc']) -> plt.Axes:
        """Returns the given axis if it exists."""
        if axis_type == 'wrapper':
            return self.ax_wrapper
        if self.on_same_graph:
            return self.ax_og
        if axis_type == 'og':
            assert self.plot_original, 'Original graph is not drawn.'
            return self.ax_og
        if axis_type == 'sc':
            assert self.plot_scaled, 'Scaled graph is not drawn.'
            return self.ax_sc

    # pylint: disable = too-many-arguments
    def traceGraph(self, spec: SubplotSpec,
                   scaling_function: ScaleFunction = None,
                   scaling_arguments: list[float] = None,
                   on_same_graph: bool = False,
                   should_plot_original: bool = True,
                   should_plot_scaled: bool = True) -> list[Line2D]:
        """Draws the scope graph using the provided `spec`."""

        self.on_same_graph = on_same_graph
        self.plot_original = should_plot_original
        self.plot_scaled = should_plot_scaled

        def plotOriginal():
            self.ln_og, = self.getAxis('og').plot(self.x_data, self.y_data)

        def plotScaled():
            if scaling_function:
                y_data = [scaling_function(x, *(scaling_arguments or [])) for x in self.y_data]
            else:
                y_data = self.y_data
            self.ln_sc, = self.getAxis('sc').plot(self.x_data, y_data)

        def createAxes():
            self.ax_wrapper = plt.subplot(spec)
            if on_same_graph:
                self.ax_og = self.ax_wrapper
                self.ax_sc = None
            elif should_plot_original and should_plot_scaled:
                inner = spec.subgridspec(2, 1, wspace=0.05, hspace=0.1)
                curr_fig = plt.gcf()
                self.ax_og = curr_fig.add_subplot(inner[0, 0])

                # pylint: disable = expression-not-assigned
                [t.set_color('white') for t in self.ax_wrapper.yaxis.get_ticklabels() + self.ax_wrapper.xaxis.get_ticklabels()]
                [self.ax_wrapper.tick_params(axis=_ax, colors='white') for _ax in ['x', 'y']]
                [spine.set_color('white') for spine in self.ax_wrapper.spines.values()]

                self.ax_wrapper.set_xlabel(self.x_label)
                self.ax_wrapper.set_ylabel(self.y_label)
                self.ax_sc = curr_fig.add_subplot(
                    inner[1, 0], sharex=self.ax_og)
            else:
                if should_plot_original:
                    self.ax_og = self.ax_wrapper
                elif should_plot_scaled:
                    self.ax_sc = self.ax_wrapper

        createAxes()

        if should_plot_original:
            plotOriginal()

        if should_plot_scaled:
            plotScaled()

        return [self.ln_og, self.ln_sc]

    def updateGraph(self, scaling_function: ScaleFunction = None,
                    scaling_arguments: list[float] = None):
        """Updates the scaled graph."""
        assert self.ln_sc, 'Nothing to update.'

        if scaling_function:
            y_data = [scaling_function(x, *(scaling_arguments or [])) for x in self.y_data]
        else:
            y_data = self.y_data

        self.ln_sc.set_ydata(y_data)

    def copy(self):
        """Returns a copy of the scope. Without the plot data."""
        return Scope(self.id, dict([(self.x_label, self.x_data), (self.y_label, self.y_data)]))

    def plot(self,
             show_original: bool = True,
             interpolate: bool = False,
             fitting_function: Callable[...,
                                        float] = None,
             bounds: tuple[float,
                           float] = (-inf,
                                     inf),
             x_scale: str = None,
             y_scale: str = None):
        """Plots the scope in a new window."""
        if not show_original and not interpolate:
            raise Exception('Nothing to show.')

        xnew = np.linspace(min(self.x_data), max(self.x_data), 300)

        if show_original:
            plt.plot(self.x_data, self.y_data)

        if interpolate:
            spl = make_interp_spline(self.x_data, self.y_data, k=3)
            Y = spl(xnew)
            plt.plot(xnew, Y)

            Y = savgol_filter(Y, 29, 13)
            plt.plot(xnew, Y)

        if fitting_function:
            popt = curve_fit(fitting_function, self.x_data, self.y_data, bounds=bounds)[0]
            plt.plot(self.x_data, fitting_function(self.x_data, *popt), 'r-')

        if x_scale:
            print(x_scale)

            plt.xscale(x_scale)

        if y_scale:
            plt.yscale(y_scale)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        Helper.show(f'scope_{self.id}' + ('_interpolated' if interpolate else ''))
        # plt.show()


DataEntry = dict[str, float | Scope]
Uncertainty = dict[str, dict[str, float]]


class ExperienceData (dict[str, DataEntry]):
    """Custom `dict` class holding experience data."""

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key) -> DataEntry:
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, __key: str, val: DataEntry) -> None:
        if not isinstance(val, dict):
            raise Exception('The item must be a `dict`.')

        return super().__setitem__(__key, val)

    def getDataKeys(self) -> list[str]:
        """Returs the keys of the"""
        return list(self[list(self.keys())[0]].keys())

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        """Merges `self` with another dict"""
        for key, val in dict(*args, **kwargs).items():
            self[key] = val


Data = dict[str, dict[str, float | Scope]]


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
        return self.getData()

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

        return self.getData()

    # pylint: disable = unsubscriptable-object
    def getData(self) -> dict[str, list[float] | list[str] | bool | str]:
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


class Expression:
    """A wrapper function that add an entry to a dataset using data from this dataset."""
    function: Callable[..., float]
    defaults: dict[str, float]
    arguments: list[str]

    def __init__(self, function: Callable[..., float], defaults: dict[str, float] = None) -> None:
        self.function = function
        self.defaults = defaults or {}
        self.arguments = signature(function).parameters

    def __call__(self, data: dict[str, float], uncert: dict[str, float] = None) -> float:
        args = []
        value = None
        for arg in self.arguments:
            if uncert and arg.strip().startswith('u_'):
                value = uncert.get(arg.replace('u_', ''))
            elif arg in list(data.keys()):
                value = data.get(arg)
            else:
                value = self.defaults.get(arg)
            if value is None:
                raise Exception(f'Unknown key: "{arg}". Available keys: {list(data.keys())}')
            args.append(value)
        return self.function(*args)


ExpressionsList = dict[str, Expression]


class Expressions:
    """Useful expressions."""
    # Energie éléctrique aux bornes de la résistance.
    Ee = Expression(lambda graph, r, threshold, offset: Helper.computeIntegral(
        graph.x_data, graph.y_data, scaling_function=lambda x: (x + offset) ** 2, threshold=threshold) / r, {'threshold': 0, 'offset': 0})

    # Incertitude sur l'énérgie éléctrique aux bornes de la résistance.
    u_Ee = Expression(lambda Ee, r, u_r: Uncertainties.quotient(Ee * r, r, 0.1, u_r))

    # Incertitude sur la mesure de distance.
    u_h = Expression(lambda: Constants.u_h)

    # Energie potentielle.
    # Ep = mgh
    Ep = Expression(lambda h: Constants.M * Constants.g * h)

    # Incertitude sur l'énérgie potentielle.
    u_Ep = Expression(lambda h, u_h: Uncertainties.product(Constants.M * Constants.g, h, Constants.u_M, u_h))

    # Temps de chute.
    # T = sqrt(2h/g)
    T = Expression(lambda h: sqrt(2 * h / Constants.g))

    u_T = Expression(lambda T, u_h: Uncertainties.power(T ** 2, 1 / 2, u_h))

    # Rapport Ee / Ep.
    Ee_Ep = Expression(lambda Ee, Ep: Ee / Ep)

    # Incertitude rapport Ee/Ep.
    u_Ee_Ep = Expression(lambda Ee, Ep, u_Ee, u_Ep: Uncertainties.quotient(Ee, Ep, u_Ee, u_Ep))

    # Incertitude sur R:
    u_r = Expression(lambda: Constants.u_r)


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
        self.slider.on_changed(self.onSliderChanged)

    def onSliderChanged(self, new_value: float):
        """Default on_changed callback."""
        self.slider_callback(self.label, new_value)

    def setSliderValue(self, new_value: float):
        """Sets the value of the slider."""
        try:
            self.slider.set_val(new_value)
        except AttributeError as e:
            print('putain de merde', e)

    def onToggleChanged(self, new_value: bool):
        """Default on_changer callback."""
        self.toggle_callback(self.label, new_value)

    def set_toggle(self, new_value: bool):
        """Sets the value of the toggle."""
        self.active = new_value
        self._update_toggle()
        self.onToggleChanged(self.active)

    def getToggle(self):
        """Gets the current toggle value."""
        return self.active

    def toggle(self):
        """Activates or deactivates the slider, sending the appropriate callback."""
        self.active = not self.active
        self._update_toggle()
        self.onToggleChanged(self.active)

    def _updateToggle(self):
        self.button.label.set_text('On' if self.active else 'Off')


class Helper:
    """Helper class handling useful calculations and functions."""

    @staticmethod
    def init():
        plt.figure(figsize=(19, 10), dpi=100)

    @staticmethod
    # pylint: disable=E1136,R0914
    def fileToData(path: str, data_title_line: int = 0, measurements_file_prefix: str = 'scope_',
                   measurements_title_line: int = 1) -> ExperienceData:
        """Convers a measurments data file to python dictionary."""
        with open(path, 'r', encoding='utf-8') as file:
            global_data = ExperienceData()
            lines = file.readlines()
            entries = [x.strip().lower() for x in lines[data_title_line].split(',')]

            try:
                id_index = entries.index('id')
            except ValueError:
                id_index = None

            for i, line in enumerate(lines[data_title_line + 1:]):
                # Ignore 'commented' lines
                if line.strip().startswith('#'):
                    continue
                values = line.split(',')
                assert len(entries) == len(values), "Titles and values count don't match."

                if id_index is None:
                    _id = i
                else:
                    _id = values[id_index].strip()

                local_data = {}
                for j, entry in enumerate(entries):
                    if entry == 'id':
                        continue
                    local_data[entry] = float(values[j])

                    if id_index is not None:
                        scopes = Helper.measurementsToScopeData(
                            f"{path.strip().removesuffix('.csv')}/{measurements_file_prefix}{_id}.csv", measurements_title_line)

                        if len(scopes) > 1:
                            for k, _ in enumerate(scopes):
                                local_data[f'graph{k}'] = scopes[k]
                        else:
                            local_data['graph'] = scopes[0]
                global_data[_id] = local_data

        return global_data

    @staticmethod
    def fileNameFromPath(path: str):
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def measurementsToScopeData(path: str, id: str = None, measurements_title_line: int = 1) -> list[Scope]:
        """Converts measurments from a Scope to readable dict."""

        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            entries = [x.strip().lower() for x in lines[measurements_title_line].split(',')]
            columns_count = len(entries)
            temp_data: dict[str, list[float]] = {}
            for i in range(columns_count):
                temp_data[entries[i]] = []
            for line in lines[measurements_title_line + 1:]:
                if line.strip().startswith('#'):
                    continue
                try:
                    values = [float(x) for x in line.split(',')]
                    for i in range(columns_count):
                        temp_data[entries[i]].append(values[i])
                except ValueError:
                    pass

        _id = id or Helper.fileNameFromPath(path)

        return [Scope(_id, {
            entries[0]: np.array(temp_data[entries[0]]),
            entries[i]: np.array(temp_data[entries[i]])}
        ) for i in range(1, columns_count)]

    @staticmethod
    def computeIntegral(x_data: list[float],
                        y_data: list[float],
                        threshold: float = 0.0,
                        scaling_function: Callable[...,
                                                   float] = lambda y: y,
                        scaling_aguments: list[float] = None):
        """Calculates the integral for a given graph.

        `Y` axis values can be scaled using the `scaling_function` attribute.
        """

        def above_threshhold(val: float):
            return abs(val) >= abs(threshold)

        assert len(x_data) == len(y_data), "The two lists lengths don't match."

        assert len(signature(scaling_function).parameters) == len(
            scaling_aguments or []) + 1, "The argument count for the scaling function is wrong."

        for i, val in enumerate(y_data):
            if not above_threshhold(val):
                y_data[i] = 0

        _y = scaling_function(y_data, *(scaling_aguments or []))

        integral: float = simpson(_y, x_data)
        return integral

    @staticmethod
    # pylint: disable=E1136
    def computeExpressions(expr_type: Literal['data', 'uncert'], expressions: ExpressionsList,
                           data: Data, uncert: Uncertainty = None, element_id: str = None):
        """Calculates the value (or uncertainty) of a dataset entry using data from the dataset and a given expression."""

        assert (expr_type == 'uncert' and uncert) or not uncert, 'Uncertainties are required.'

        _data = uncert if expr_type == 'uncert' else data

        for (expr_key, expression) in expressions.items():
            if element_id is not None:
                if expr_type == 'uncert':
                    try:
                        _data[element_id][expr_key] = expression(data[element_id], uncert[element_id])
                    except KeyError:
                        pass
                else:
                    _data[element_id][expr_key] = expression(data[element_id])
            else:
                for _id in data.keys():
                    assert expr_key not in _data[_id].keys(), f'The key "{expr_key}" already exists in the provided data.'
                    _data[_id][expr_key] = expression(data[_id], uncert[_id] if (expr_type == 'uncert') else None)

    @staticmethod
    def defaultUncertainties(data: Data, uncert: Uncertainty):
        """Returns default uncertainties for certain dataset entries."""

        default_uncertainties = [
            ('h', Expression(lambda: Constants.u_h)),
            ('r', Expression(lambda: Constants.u_r))
        ]
        for (expr_key, expression) in default_uncertainties:
            variables = list(list(data.values())[0].keys())
            if expr_key in variables:
                for key in data.keys():
                    uncert[key][expr_key] = expression(data[key], uncert[key])

    @staticmethod
    # pylint: disable=E1136
    def getGlobalScopeMinMax(data: dict[str, dict[str, float | Scope]], axis: Literal['x', 'y'] = 'y'):
        """Computes graph's extremum values for X and Y axes on the whole dataset."""
        g_min: float = None
        g_max: float = None

        def localMinMax(arr: list[float]):
            l_min: float = arr[0]
            l_max: float = arr[0]

            for _x in arr:
                if _x < l_min:
                    l_min = _x
                if _x > l_max:
                    l_max = _x

            return (l_min, l_max)

        for _data in data.values():
            local_min, local_max = localMinMax(_data['graph'].x_data if axis == 'x' else _data['graph'].y_data)
            if min is None or local_min < g_min:
                g_min = local_min
            if max is None or local_max > g_max:
                g_max = local_max
        return (min, max)

    @staticmethod
    def safeListCopy(obj: any) -> list:
        """Returns a shallow copy of a python `list`."""
        copy: list = []
        for value in obj:
            if isinstance(value, dict):
                copy.append(Helper.safeDictCopy(obj))
            elif isinstance(value, list):
                copy.append(Helper.safeListCopy(obj))
            else:
                copy.append(value)
        return copy

    @staticmethod
    def safeDictCopy(obj: any) -> dict:
        """Returns a shallow copy of a python `dict`."""
        if isinstance(obj, ExperienceData):
            copy: ExperienceData = ExperienceData()
        else:
            copy: dict = {}
        if not hasattr(obj, 'items'):
            return obj

        for key, value in obj.items():
            if isinstance(value, dict):
                copy[key] = Helper.safeDictCopy(value)
            elif isinstance(value, list):
                copy[key] = Helper.safeListCopy(value)
            else:
                copy[key] = value
        return copy

    @staticmethod
    def trace(data: dict[str, list[float]], scaling_function=lambda x: x, custom_name: str = None):
        """Traces a graph using `matplolib.pyplot`"""

        x_label, y_label = list(data.keys())
        x_data = np.array(data[x_label])
        y_data = scaling_function(np.array(data[y_label]))

        plt.plot(x_data, y_data)

        if custom_name:
            Helper.show(custom_name)

        # plt.show()

    @staticmethod
    def initializeGraph(plots_count: int):
        """Computes the graph matrix size."""
        columns = ceil(sqrt(plots_count))
        rows = ceil(plots_count / columns)
        is_graph_2d = rows > 1
        return GridSpec(rows, columns), is_graph_2d, columns, rows

        # return plt.subplots(rows, columns, constrained_layout=True)[1], is_graph_2d, columns, rows

    @staticmethod
    # pylint: disable=E1136
    def computeAngle(angle: float, src: Literal['deg', 'rad'], dst: Literal['deg', 'rad'], ):
        if not (src in ('deg', 'rad') and dst in ('deg', 'rad')):
            raise Exception('Unknown value for source or destination unit.')
        norm: float

        destinations = {'deg': 180, 'rad': Constants.pi}

        if src == 'deg':
            # [-360; 360] -> [-1; 1]
            norm = angle / 360
        else:
            # [-2pi; 2pi] -> [-1; 1]
            norm = angle / (2 * Constants.pi)

        if -1 < norm < -0.5:
            norm += 1
        elif 0.5 < norm < 1:
            norm -= 1

        return norm * destinations[dst]

    @staticmethod
    def impedenceFromGraph(path: str, R: float, f_min: float = 30000, f_max: float = 95000, invert_data: bool = False):
        # z = u / i
        scopes = Helper.measurementsToScopeData(path)

        if len(scopes) != 2:
            raise Exception('Two scopes measurements are required to compute impedence.')
        if len(scopes[0].y_data) != len(scopes[1].y_data):
            raise Exception('Data length doesn\' match.')
        # abs(e - u) / abs(u) * r)
        # e: generator
        # u: resistor
        U = scopes[0].y_data if not invert_data else scopes[1].y_data
        E = scopes[1].y_data if not invert_data else scopes[0].y_data

        F = np.linspace(f_min, f_max, len(scopes[0].y_data))

        Z = []

        for i, _ in enumerate(U):
            Z.append(abs(E[i] - U[i]) / abs(U[i]) * R)

        return Scope(Helper.fileNameFromPath(path), {'f': np.array(F), 'Z': np.array(Z)})

    @staticmethod
    def coefficient(coeff: str, fm: float, fM: float):
        if coeff == 'k31':
            fMOverfm = fM / fm
            piOverTwo = Constants.pi / 2
            T = piOverTwo * fMOverfm * tan(piOverTwo * (fMOverfm - 1))
            return sqrt(T / (1 + T))
        else:
            raise Exception('Unknow coefficient name.')

    @staticmethod
    def expect(result: float, expected: float, unit: str):
        def addUnit(s: str):
            return s + (f' {unit}' if unit else '')

        print(addUnit(f'Expected: {expected}'))
        print(addUnit(f'Got: {result}'))
        err = round((abs(expected - result) / expected if expected !=
                    0 else abs(expected - result) / result if result != 0 else -1) * 1000) / 10
        print(f'Relative error: {err}%')

    @staticmethod
    def formatFileName(experience_name: str = None, args: list[tuple[str, str]] | str = None):
        s = experience_name
        if args:
            if isinstance(args, list):
                for x in args:
                    s += f'_({x[0]}, {x[1]})'
            elif isinstance(args, str):
                s += f'_{args}'
            else:
                raise Exception(f'Unknown type: {type(args)}')

        return s

    @staticmethod
    def show(experience_name: str = None, args: list[tuple[str, str]] | str = None):
        '''Custom `pyplot` show command that also saves the plot.'''
        if experience_name:
            fname = Helper.formatFileName(experience_name, args)
            plt.savefig(f'images/graphs/{fname}.png', dpi=500)
            plt.savefig(f'images/graphs/{fname}.svg', dpi=500)

        plt.show()


class Experiment:
    """An experiment wrapper.

    Experiment related data and useful methods are packed here.
    """
    file: str
    expr_meas: ExpressionsList
    expr_uncert: ExpressionsList
    uncert: Uncertainty
    data: ExperienceData

    def __init__(self, file: str, measurements_expressions: ExpressionsList = None,
                 uncertainties_expressions: ExpressionsList = None) -> None:
        self.file = file

        self.expr_meas = measurements_expressions or {}
        self.expr_uncert = uncertainties_expressions or {}

        self.data = Helper.fileToData(file)

        self.uncert = {x: {} for x in self.data}

        Helper.computeExpressions('data', self.expr_meas, self.data)
        Helper.defaultUncertainties(self.data, self.uncert)
        Helper.computeExpressions('uncert', self.expr_uncert, self.data, self.uncert)

    def traceMeasurementGraphs(self, ids: list[str] = None):
        """Traces Scope graphs of the current dataset."""
        if 'graph' not in self.data.getDataKeys():
            assert False, 'The current exprimentdoesn\'t have associated graph.'

        count = len(self.data) if ids is None else len(ids)
        gris_spec, is_fig_2d, col_count = Helper.initializeGraph(count)[:3]
        curves_ids = ids or list(self.data.keys())
        scopes_count = len(curves_ids)

        for i in range(count):
            spec = gris_spec[i // col_count, i % col_count] if is_fig_2d else gris_spec[i]

            if i < scopes_count:
                graph = self.data[curves_ids[i]]['graph']
                if isinstance(graph, Scope):
                    graph.traceGraph(spec, should_plot_scaled=False)
                    graph.ax_og.set_title(curves_ids[i])

        gris_spec.tight_layout(plt.gcf(), h_pad=0.3, w_pad=0.2)

        Helper.show(Helper.fileNameFromPath(self.file), 'measurements')

        # plt.show()

    def trace(self, curves: list[Curve], ignore: list[str] = None):
        """Traces `Curves` using the current dataset."""
        curves_count = len(curves)
        specs, is_fig_2d, col_count = Helper.initializeGraph(curves_count)[:3]

        for curve in curves:
            curve.populate(self.data, self.uncert, ignore)

        data_len = len(self.data.keys())

        for i in range(curves_count):
            spec = specs[i // col_count, i % col_count] if is_fig_2d else specs[i]

            if i < curves_count:
                ax = plt.subplot(spec)
                data = curves[i].getData()
                ax.scatter(data['x'], data['y'])

                ax.set_xlabel(data['x_label'])
                ax.set_ylabel(data['y_label'])

                valid_ux = len(data['ux']) == data_len
                valid_uy = len(data['uy']) == data_len

                for j, _id in enumerate(self.data.keys()):
                    x_pos: float = data['x'][j]
                    y_pos: float = data['y'][j]
                    ax.annotate(_id, (x_pos, y_pos),
                                textcoords='offset points', xytext=(0, 5), ha='center')
                    if data['unc'] and (valid_ux or valid_uy):
                        ax.errorbar(
                            data['x'][j],
                            data['y'][j],
                            data['uy'][j] if valid_uy else 0,
                            data['ux'][j] if valid_ux else 0,
                            ecolor='firebrick')
            else:
                ax.axis('off')

        Helper.show(Helper.fileNameFromPath(self.file), [(curve.x_label, curve.y_label) for curve in curves])

        # plt.show()

    def correctMeasurements(
            self,
            curve: Curve,
            corrected_values: list[str] = None,
            expression: Expression = None):
        """Starts a correction session for given values."""
        corrector = Corrector(self.file,
                              self.data,
                              expressions=self.expr_meas,
                              corrected_values=corrected_values or ['threshold', 'offset'])
        corrector.start(curve, expression)


class Corrector:
    """A wrapper object for handling correction on given dataset values."""
    curr_index: int
    prev_index: int
    correction_index: list[int]
    corrected_values: list[str]
    data: Data
    expressions: ExpressionsList
    ids: list[str]
    indexes: dict[str, int]
    columns: list[str]
    file: str

    def __init__(self,
                 path: str, data: Data,
                 expressions: ExpressionsList,
                 corrected_values: list[str] = None,
                 title_line: int = 0) -> None:
        self.data = data
        self.file = path
        self.expressions = expressions

        self.ids = list(data.keys())
        self.indexes = {k: i for i, k in enumerate(data.keys())}
        self.correction_index = []
        self.curr_index = 0
        self.prev_index = None

        self.corrected_values = corrected_values or ['threshold', 'offset']

        with open(path, 'r', encoding='utf-8') as file:
            self.columns = [x.strip().lower()
                            for x in file.readlines()[title_line].split(',')]

        needs_update = False
        for col in corrected_values:
            if col not in self.columns:
                self.columns.append(col)
                self.correction_index.append(len(self.columns) - 1)
                for key in self.data.keys():
                    if col == 'offset':
                        x_data = data[key]['graph'].x_data
                        self.data[key][col] = - sum(x_data) / len(x_data)
                    else:
                        self.data[key][col] = .0
                needs_update = True
            else:
                self.correction_index.append(self.columns.index(col))

        if needs_update:
            self.writeToFile()

    def writeToFile(self):
        """Writes current experiment settings to file."""
        # pylint: disable = unsubscriptable-object
        file_buffer: list[list[str | float]] = []
        columns = self.columns

        file_buffer.append(columns)

        for _id in self.data.keys():
            row_buffer = []
            for col in columns:
                if col == 'id':
                    row_buffer.append(_id)
                else:
                    row_buffer.append(self.data[_id][col])

            file_buffer.append(row_buffer)

        with open(self.file, 'w', newline='', encoding='utf-8') as file:
            file.writelines('\n'.join([', '.join([str(value) for value in row_buffer]) for row_buffer in file_buffer]))

    # pylint: disable = unsubscriptable-object
    def formatExpression(self, expression: Expression, data: ExperienceData):
        """Returns a formatted string of an expression evaluation."""
        res = expression(data[self.ids[self.curr_index]])
        return f'Expression: {res:.3e}'

    def start(self, curve: Curve, expression: Expression = None):
        """Begins the correction session."""

        # pylint: disable = invalid-name
        DEFAULT_COLOR = 'tab:blue'
        DEFAULT_SIZE = 20
        SELECTED_COLOR = 'tab:red'
        SELECTED_SIZE = 50

        graph_initialized: float = False
        curve_initialized = False

        data: Data = Helper.safeDictCopy(self.data)

        fig = plt.figure(figsize=(10, 8))

        graph_preview, curve_preview = GridSpec(1, 2, wspace=0.2, hspace=0.2)

        # graph_preview = plt.subplot2grid(
        #     shape=(4, 4), loc=(0, 0), colspan=2, rowspan=4)
        # curve_preview = plt.subplot2grid(shape=(4, 4), loc=(0, 2), colspan=2, rowspan=3)

        colors = ([DEFAULT_COLOR] * len(self.ids))
        sizes = [DEFAULT_SIZE] * len(self.ids)

        # graph_y_min, graph_y_max = Helper.get_global_scope_min_max(data, 'y')
        # graph_x_min, graph_x_max = Helper.get_global_scope_min_max(data, 'x')

        def handleSliderUpdate(key: str, value: float):
            """Handles modifications on sliders."""
            curr = self.ids[self.curr_index]
            data[curr][key] = value
            # print(data[curr][key], self.data[curr][key])
            Helper.computeExpressions('data', self.expressions, data, element_id=curr)

            updateGraph(curr)
            updateCurve(curr)

        def handleToggle(key: str, value: bool):
            """Handles changes to the slider toggles."""
            curr = self.ids[self.curr_index]
            ...

        sliders: dict[str, ValueSlider] = {}
        for i, key in enumerate(self.corrected_values):
            # sliders[key] = ValueSlider(key,
            #                            plt.axes([0.600, 0.1 + i * 0.05, 0.250, 0.050]),
            #                            plt.axes([0.900, 0.1 + i * 0.05, 0.050, 0.050]),
            #                            handle_slider_update,
            #                            handle_toggle,
            #                            min_max=1)
            sliders[key] = Slider(ax=plt.axes([0.600, 0.1 + i * 0.05, 0.250, 0.050]), label=key, valmin=-1, valmax=1)

        def updateCurrent(incr: int) -> tuple[str, str]:
            """Updates the current indexes and related stuff."""
            self.prev_index = self.curr_index
            self.curr_index += incr
            self.curr_index %= len(self.ids)
            prev = self.ids[self.prev_index]
            curr = self.ids[self.curr_index]

            colors[self.prev_index] = DEFAULT_COLOR
            sizes[self.prev_index] = DEFAULT_SIZE
            colors[self.curr_index] = SELECTED_COLOR
            sizes[self.curr_index] = SELECTED_SIZE

            return (curr, prev)

        graph_line = None
        thtop = None
        thbot = None
        graph_title = None
        graph_legend = None

        def updateGraph(graph_id: str):
            """Redraws the leftmost graph."""
            nonlocal graph_initialized
            nonlocal graph_line, thtop, thbot, graph_title, graph_legend
            scope: Scope = Helper.safeDictCopy(data[graph_id]['graph'])

            # graph_axes = list(graph.keys())

            if not graph_initialized:
                #     graph_preview.set_xlim([graph_x_min, graph_x_max])
                #     graph_preview.set_ylim([graph_y_min - 3, graph_y_max + 3])

                scope.traceGraph(graph_preview, should_plot_original=True, should_plot_scaled=True, on_same_graph=False)

                graph_title = scope.getAxis('wrapper').text(
                    0.29, 0.9, '', fontsize=14, transform=plt.gcf().transFigure)
                if 'threshold' in self.corrected_values:
                    thtop = scope.getAxis('og').axhline(color='red', linestyle='-')
                    thbot = scope.getAxis('og').axhline(color='red', linestyle='-')

                if expression:
                    graph_legend = scope.getAxis('wrapper').text(0.310, 0.035,
                                                                 self.formatExpression(expression, data),
                                                                 ha='center',
                                                                 va='center',
                                                                 transform=plt.gcf().transFigure)

                graph_initialized = True

            if 'threshold' in self.corrected_values and 'threshold' in data[graph_id]:
                offset = data[graph_id]['offset'] if 'offset' in data[graph_id] else 0
                thtop.set_ydata(-offset + data[graph_id]['threshold'])
                thbot.set_ydata(-offset - data[graph_id]['threshold'])

            scope.updateGraph(lambda x, threshold, offset: (x + offset if abs(x + offset) >
                                                            threshold else 0), [data[graph_id]['threshold'], data[graph_id]['offset']])

            if graph_legend:
                graph_legend.set_text(self.formatExpression(expression, data))

        curve_line = None
        annotations = {}

        def updateCurve(id: str):
            """Retraces the rightmost graph."""
            nonlocal curve_initialized
            nonlocal curve_line, annotations

            # if not curve.hasData:
            #     curve_plot = curve.populate(data, {})
            # else:
            #     curve_plot = curve.update(id, data[id], {})

            if not curve_initialized:
                #     curve_preview.set_xlabel(curve_plot['x_label'])
                #     curve_preview.set_ylabel(curve_plot['y_label'])
                #     curve_line = curve_preview.scatter(
                #         curve_plot['x'], curve_plot['y'], c=colors, s=sizes)
                #     curve_preview.autoscale_view()
                #     for j, id in enumerate(data.keys()):
                #         # annotations[id] = curve_preview.annotate(id, (curve_plot['x'][j], curve_plot['y'][j]), textcoords='offset points', xytext=(0, 5), ha='center')
                #         annotations[id] = curve_preview.annotate(
                #             id, (curve_plot['x'][j], curve_plot['y'][j]))
                curve.traceCurve(curve_preview, data)

                curve.getAxis()

                curve_initialized = True

            # curve_line.set_offsets(list(zip(curve_plot['x'], curve_plot['y'])))
            # curve_line.set_color(colors)
            # curve_line.set_sizes(sizes)
            # curve_preview.relim()

            # annotations[id].set_y(curve_plot['y'][self.indexes[id]])

        def updateSliders(key: str):
            """Redraws sliders with current values."""
            for item, slider in sliders.items():
                # print(data[key][item])
                # slider.set_slider_value(data[key][item])
                slider.set_val(data[key][item])

        def updatePage(increment: int):
            """Redraws everything on screen and handles page change."""
            curr, _ = updateCurrent(increment)

            updateGraph(curr)
            updateCurve(curr)
            updateSliders(curr)

            fig.canvas.draw_idle()

        def resetCurrent():
            """Resets values of the current entry to previous ones."""
            key = self.ids[self.curr_index]
            data[key] = self.data[key].copy()
            updatePage(0)

        def apply():
            """Saves the changes and updates the file."""
            self.data = Helper.safeDictCopy(data)
            self.writeToFile()

        axnext = plt.axes([0.445, 0.01, 0.05, 0.050])
        axprev = plt.axes([0.125, 0.01, 0.05, 0.050])
        axapply = plt.axes([0.850, 0.01, 0.05, 0.050])
        axdefault = plt.axes([0.775, 0.01, 0.05, 0.050])
        bnext = Button(axnext, '>>')
        bnext.on_clicked(lambda _: updatePage(1))
        bprev = Button(axprev, '<<')
        bprev.on_clicked(lambda _: updatePage(-1))
        bapply = Button(axapply, 'Apply')
        bapply.on_clicked(lambda _: apply())
        bdefault = Button(axdefault, 'Default')
        bdefault.on_clicked(lambda _: resetCurrent())

        updatePage(0)

        plt.show()
