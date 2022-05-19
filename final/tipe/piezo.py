from datetime import datetime
import os
from inspect import signature
from math import ceil, floor, log10, pi, sqrt, tan
from re import L
from time import time
from typing import Callable, Literal

import matplotlib as mpl
import numpy as np
import serial
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy import integrate

# ====================================
#          Matplotlib Settings
# ====================================


mpl.use('pgf')
mpl.rcParams.update({
    'pgf.texsystem': 'xelatex',
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# ====================================
#                Types
# ====================================


ScopeData = dict[str, list[float]]
ExperienceData = dict[str, dict[str, float | ScopeData]]


# ====================================
#                Value
# ====================================


class Value:
    def __init__(self, value: float, uncert: float = None):
        assert isinstance(value, (float, int)), f'value must be float, found "{type(value)}"'
        assert uncert is None or isinstance(uncert, (float, int)), f'uncert must be float, found "{type(uncert)}"'
        self.value = value
        if uncert:
            assert uncert >= 0, 'uncertainty must be positive'
        self.uncert = uncert

    def __str__(self):
        if self.uncert is None or self.uncert <= 0:
            return f'{self.value}'

        precision = - floor(log10(self.uncert))
        if precision <= 0:
            return f'{self.value} ± {self.uncert}'
        return f'{self.value:.{precision}f} ± {self.uncert:.{precision}f}'

    def __repr__(self):
        return self.__str__()

    def tuple(self):
        return (self.value, self.uncert)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.value + other, self.uncert)
        elif isinstance(other, Value):
            return Value(
                self.value +
                other.value,
                (self.uncert if self.uncert is not None else .0) + (other.uncert if other.uncert is not None else .0))
        else:
            assert False, 'not implemented'

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.value + other, self.uncert)
        else:
            assert False, 'not implemented'

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return Value(other - self.value, self.uncert)

    def __neg__(self):
        return Value(-self.value, self.uncert)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.value * other, self.uncert * abs(other) if self.uncert is not None else None)
        elif isinstance(other, Value):
            return Value(self.value * other.value, abs(self.value * other.uncert) + abs(other.value *
                         self.uncert) if self.uncert is not None and other.uncert is not None else None)
        else:
            assert False, 'not implemented'

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.value * other, self.uncert * abs(other) if self.uncert is not None else None)
        elif isinstance(other, Value):
            return Value(self.value * other.value, abs(self.value * other.uncert) + abs(other.value *
                         self.uncert) if self.uncert is not None and other.uncert is not None else None)
        else:
            assert False, 'not implemented'

    def __round__(self):
        return round(self.value)

    def __float__(self):
        return self.value

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.value / other, self.uncert / other if self.uncert else None)
        elif isinstance(other, Value):
            return Value(self.value / other.value, (1 / other.value) * (self.uncert + (self.uncert / other.value)
                         * other.uncert) if self.uncert is not None and other.uncert is not None else None)
        else:
            assert False, 'not implemented'

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            if self.value == 0:
                return Value(0, 0)
            new_value = self.value ** other
            return Value(new_value, abs(new_value * (self.uncert / self.value) * other) if self.uncert is not None else None)
        else:
            assert False, 'not implemented'

    def __abs__(self):
        return Value(abs(self.value), self.uncert)

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.value >= other
        elif isinstance(other, Value):
            return self.value >= other.value
        else:
            assert False, 'not implemented'

    def __le__(self, other):
        return Value.__ge__(-self, -other)

    def __lt__(self, other):
        return Value.__gt__(-self, -other)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.value > other
        elif isinstance(other, Value):
            return self.value > other.value
        else:
            assert False, 'not implemented'


# ====================================
#              Constants
# ====================================


class CONSTANTS:
    # Accélération terrestre.
    g: Value = Value(9.80665, 0.00001)      # m.s-2
    # Masse par défaut d'une bille.
    M: Value = Value(6.9E-3, 0.1E-3)       # kg
    # Hauteur du capteur.
    Hc: Value = Value(4.5E-3, 0.1E-3)      # m
    # Pi
    pi: Value = Value(3.141592653589793, 1E-15)

    u_M: float = 0.1E-3
    # Incertitude sur la mesure de la hauteur.
    u_h: float = .5E-3       # m
    # # Incertitude sur la valeur de la résistance.
    u_r: float = 0.01       # % (relative)


# ====================================
#             Expression
# ====================================


class Expression:
    function: Callable[..., Value]
    defaults: dict[str, Value]
    arguments: list[str]

    def __init__(self, f: Callable[..., Value], defaults: dict[str, Value] = None):
        self.function = f
        self.defaults = defaults or {}
        self.arguments = signature(f).parameters

    def __call__(self, data: dict[str, Value]) -> Value:
        args = []
        value = None
        for arg in self.arguments:
            if arg in data:
                value = data.get(arg)
            else:
                value = self.defaults.get(arg)

            if value is None:
                assert False, f'Unknown key {arg}. Available keys: {list(data.keys())}'

            args.append(value)

        return self.function(*args)


# ====================================
#           Custom Expressions
# ====================================


class Expressions:
    '''Expressions'''

    Ee = Expression(lambda scope, r: Helper.computeIntegral(scope.x_data, scope.y_data, scaling=lambda x: (x ** 2) / r))
    '''Electric Energy'''

    Ep = Expression(lambda h: CONSTANTS.g * CONSTANTS.M * h)
    '''Potential Energy'''

    T = Expression(lambda h: sqrt(2 * h / CONSTANTS.g))
    '''Falling time'''

    EeOverEp = Expression(lambda Ee, Ep: Ee / Ep)
    '''Piezoelectric constant: k^2'''

    ''''''


# ====================================
#                Scope
# ====================================


class Scope:
    id: str

    ax_name_x: str
    ax_name_y: str

    x_data: list[float]
    y_data: list[float]

    ax: plt.Axes

    def __init__(self, id: str, scope_data: ScopeData) -> None:
        keys = list(scope_data.keys())

        self.id = id

        self.ax_name_x = keys[0]
        self.ax_name_y = keys[1]
        self.x_data = scope_data[self.ax_name_x]
        self.y_data = scope_data[self.ax_name_y]

    def trace(self, spec: gridspec.SubplotSpec):
        if spec is None:
            assert False, 'Missing plotting spec.'

        self.ax = plt.subplot(spec)
        self.ax.plot(self.x_data, self.y_data)


class Helper:
    @staticmethod
    def defaultUncertainty(key: str, value: float):
        UNCERTAINTIES = {
            'h': CONSTANTS.u_h,
            'm': CONSTANTS.u_M,
            'r': lambda val: abs(CONSTANTS.u_r * val),
        }

        if key not in UNCERTAINTIES:
            return None

        return UNCERTAINTIES[key](value) if hasattr(UNCERTAINTIES[key], '__call__') else UNCERTAINTIES[key]

    @staticmethod
    def fileToDict(path: str, line_offset: int = 0, prefix: str = 'scope_', measurements_title_line: int = 1) -> 'ExperienceData':
        with open(path, 'r') as file:
            data = {}
            lines = file.readlines()
            entries = [x.strip().lower() for x in lines[line_offset].split(',')]

            try:
                id_index = entries.index('id')
            except ValueError:
                id_index = None

            for i, line in enumerate(lines[line_offset + 1:]):
                if line.strip().startswith('#'):
                    continue

                values = line.split(',')
                assert len(entries) == len(values), 'Titles and values counts do not match.'

                if id_index is None:
                    _id = i
                else:
                    _id = values[id_index].strip()

                local_data = {}
                for j, entry in enumerate(entries):
                    if entry == 'id':
                        continue
                    val = float(values[j])
                    local_data[entry] = Value(val, Helper.defaultUncertainty(entry, val))

                    if id_index is not None:
                        scopes = Helper.measurementsToScope(
                            f'{path.strip().removesuffix(".csv")}/{prefix}{_id}.csv', measurements_title_line)
                        count = len(scopes)
                        if count > 1:
                            for k in range(count):
                                local_data[f'scope{k}'] = scopes[k]
                        else:
                            local_data['scope'] = scopes[0]

                data[_id] = local_data

        return data

    @staticmethod
    def fileNameFromPath(path: str):
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def measurementsToScope(path: str, id: str = None, measurements_title_line: int = 1):
        with open(path, 'r') as f:
            lines = f.readlines()
            entries = [x.strip().lower() for x in lines[measurements_title_line].split(',')]
            columns_count = len(entries)
            temp_data: dict[str, list[Value]] = {}
            for i in range(columns_count):
                temp_data[entries[i]] = []
            for line in lines[measurements_title_line + 1:]:
                if line.strip().startswith('#'):
                    continue

                try:
                    values = [float(x) for x in line.split(',')]
                    for i in range(columns_count):
                        temp_data[entries[i]].append(Value(values[i]))
                except ValueError:
                    pass
        _id = id or Helper.fileNameFromPath(path)

        return [Scope(_id, {
            entries[0]: temp_data[entries[0]],
            entries[i]: temp_data[entries[i]]
        }) for i in range(1, columns_count)]

    @staticmethod
    def computeIntegral(x: list, y: list, scaling: Callable[..., any] = None):
        use_uncert = True
        if (not isinstance(x[0], Value)):
            X = x
            use_uncert = False
        else:
            X, X_uncert = Helper.unpackValueList(x)

        if (isinstance(x[1], Value)):
            Y = y
            use_uncert = False
        else:
            Y, Y_uncert = Helper.unpackValueList([scaling(_y) for _y in y])

        integral: float | Value = integrate.simpson(Y, X)
        uncert: float = sum([abs(_x) * u_y + abs(_y) * u_x for _x, _y, u_x, u_y in zip(X, Y, X_uncert, Y_uncert)]) if use_uncert else None

        if isinstance(integral, Value):
            return Value(integral.value, uncert)

        return Value(integral, uncert)

    @staticmethod
    def computeExpressions(expressions: dict[str, Expression], data: ExperienceData, id=None):
        for expr_key, expression in expressions.items():
            if id is not None:
                data[id][expr_key] = expression(data[id])
            else:
                for _id in data.keys():
                    assert expr_key not in data[_id].keys(), f'Key {expr_key} already exists in the dataset, cannot override.'
                    data[_id][expr_key] = expression(data[_id])

    @staticmethod
    def trace(data: dict[str, list[Value]], scaling: Callable[..., Value] = None, save_name: str = None):
        x_label, y_label = list(data.keys())
        x = data[x_label]
        y = [scaling(_y) for _y in data[y_label]]

        if (not isinstance(x[0], Value)):
            X, X_uncert = x, None
        else:
            X, X_uncert = Helper.unpackValueList(x)

        if (isinstance(x[1], Value)):
            Y, Y_uncert = y, None
        else:
            Y, Y_uncert = Helper.unpackValueList([scaling(_y) for _y in y])

        plt.errorbar(X, Y, xerr=X_uncert, yerr=Y_uncert)

        if save_name is not None:
            Helper.save(save_name)

    @staticmethod
    def initializeGraph(count: int):
        cols = ceil(sqrt(count))
        rows = ceil(count / cols)
        is_2D = rows > 1
        return gridspec.GridSpec(rows, cols), is_2D, cols, rows

    @staticmethod
    def angle(angle: float, src: Literal['deg', 'rad'], dst: Literal['deg', 'rad']):
        if not (src in ('deg', 'rad') and dst in ('deg', 'rad')):
            assert False, 'Unknown value for source or destination unit.'

        part = {'deg': 180, 'rad': pi}

        # [-360, 360] -> [-1, 1]
        # [-2pi, 2pi] -> [-1, 1]
        norm: float = angle / (2 * part[src])

        if -1 < norm < -0.5:
            norm += 1
        elif 0.5 < norm < 1:
            norm -= 1

        return norm * part[dst]

    @staticmethod
    def impedenceFromScope(path: str, R: Value, f_min: float, f_max: float):
        # Z = U / I

        scopes = Helper.measurementsToScope(path)

        assert len(scopes) == 2, 'Two scopes measurements are required to compute impedence.'
        assert len(scopes[0].y_data) == len(scopes[1].y_data), 'Data sizes don\'t match.'

        # abs(e - u) / abs(u) * r
        # e: generator
        # u: resistor
        U = scopes[0].y_data
        E = scopes[1].y_data

        F = np.linspace(f_min, f_max, len(scopes[0].y_data))
        Z = [abs(E[i] - U[i]) / abs(U[i]) * R for i in range(len(U))]

        return Scope(Helper.fileNameFromPath(path), {'f': np.array(F), 'z': np.array(Z)})

    @staticmethod
    def coefficient(coeff: str, fm: float, fM: float):
        if coeff == 'k31':
            fMOverfm = fM / fm
            piOverTwo = pi / 2
            T = piOverTwo * fMOverfm * tan(piOverTwo * (fMOverfm - 1))
            return sqrt(T / (1 + T))
        else:
            assert False, f'Unknown coefficient: "{coeff}".'

    @staticmethod
    def expect(result: float, expected: float, unit: str):
        def appendUnit(s: str):
            return s + (f' {unit}' if unit else '')

        print(appendUnit(f'Expected: {expected}'))
        print(appendUnit(f'Got: {result}'))
        err = round((abs(expected - result) / expected if expected !=
                    0 else abs(expected - result) / result if result != 0 else -1) * 1000) / 10
        print(f'Relative error: {err}%')

    @staticmethod
    def formatFileName(exp_name: str, args: list[tuple[str, str]] | str = None):
        s = exp_name
        if args is not None:
            if isinstance(args, list):
                for x in args:
                    s += f'_({x[0]}, {x[1]})'
            elif isinstance(args, str):
                s += f'_{args}'
            else:
                assert False, f'Unknown type: {type(args)}'

        return s

    @staticmethod
    def save(exp_name: str, args: list[tuple[str, str]] | str = None, clear: bool = True):
        fname = Helper.formatFileName(exp_name, args)
        extensions = [['images', 'png'], ['svgs', 'svg'], ['latex', 'pgf']]
        for folder, ext in extensions:
            print(f'\nSaving "{fname}.{ext}".')
            plt.savefig(f'graphs/{folder}/{fname}.{ext}')
            print(f'Done saving "{fname}.{ext}".\n')
        if clear:
            plt.clf()

    @staticmethod
    def unpackValueList(x: list[Value]):
        assert isinstance(x[0], Value), f'{x} is not a [Value] list.'
        values: list[float] = []
        uncerts: list[float] = []
        for _x in x:
            values.append(_x.value)
            uncerts.append(_x.uncert or 0)
        return values, uncerts


# ====================================
#              Voltmeter
# ====================================


class Voltmeter:
    port: str
    C: float = 1000e-6              # F
    R: float = 1.012e6              # Ohm
    Ri: float = 319.3e3             # Ohm
    Req: float (R * Ri) / (R + Ri)  # Ohm
    ratio: float = 3.3 / 1024

    logging_file: str
    image_file: str
    latex_file: str

    def __init__(self, port: str = None, logging_file: str = 'voltmeter/log-{}.csv', image_file: str = 'voltmeter/graph-{}.svg', latex_file: str = 'voltmeter/latex-{}.pgf') -> None:
        available_ports = self.listPorts()
        if ((port and port not in available_ports) or len(available_ports) != 1):
            assert False, f'Available ports: {available_ports}'
        
        self.port = port or available_ports[0]
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.logging_file = logging_file.format(timestamp)
        self.image_file = logging_file.format(timestamp)
        self.latex_file = latex_file.format(timestamp)

    def live(self, average_on: int = 30, points: int = 500):
        start = time()

        def get_n_points(L1: list[float], L2: list[float], n: int = points):
            l = len(L1)

            if l < 2 * n:
                return L1, L2
            else:
                step = l // n
                x = 0
                new_L1 = []
                new_L2 = []
                for i in range(0, l - n // 10, step):
                    new_L1.append(L1[i])
                    new_L2.append(L2[i])
                for i in range(l - n // 10, l):
                    new_L1.append(L1[i])
                    new_L2.append(L2[i])
                return new_L1, new_L2

        def close(_=None):
            fig.savefig(self.image_file)
            fig.savefig(self.latex_file)
            plt.close(fig)
        
        plt.ion()

        fig, ax = plt.subplots()

        fig.canvas.mpl_connect('close_event', close)
        ax.set_xlabel('Temps (s)')
        ax.set_ylabel('Tension (V)')

        ln, = ax.plot([], [], c='royalblue', marker='o', linestyle='', label='voltage', markersize=0.8)

        def update_graph(X, Y):
            ln.set_data(X, Y)
            ax.relim()
            ax.autoscale()
            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.show()

        T = []
        V = []

        correction = 0.08

        try:
            with open(self.logging_file, 'w') as f:
                f.write('t, u, e\n')
                with serial.Serial(self.port, 9600) as arduino:
                    L: list[float] = []
                    count = 0
                    while True:
                        out = arduino.read_all()[:-2]
                        if out:
                            data = [int(x) for x in out.decode().split('\r\n') if x.strip() != ''][1:-1]
                            L += data
                            count += len(data)
                            if count > average_on:
                                avg_analog = sum(L) / count
                                voltage = avg_analog * self.ratio * (self.R + self.Req) / self.Req - correction
                                energy = 1 / 2 * self.C * voltage ** 2
                                time_offset = time() - start
                                f.write(f'{time_offset}, {voltage}, {energy}\n')

                                T.append(time_offset)
                                V.append(voltage)
                                X, Y = get_n_points(T, V)

                                update_graph(X, Y)

                                count = 0
                                L = []

        except KeyboardInterrupt:
            close()

    @staticmethod
    def listPorts():
        ports = ['COM%s' % (i + 1) for i in range(256)]
        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result


# ====================================
#             Experiment
# ====================================


class Experiment:
    path: str
    expressions: dict[str, Expression]
    data: dict[str, dict[str, Value | Scope]]

    def __init__(self, path: str, expressions: dict[str, Expression] = None) -> None:
        self.path = path
        self.expressions = expressions or {}
        self.data = Helper.fileToDict(path)

        Helper.computeExpressions(self.expressions, self.data)

    def saveMeasurementScopes(self):
        count = len(self.data)
        grid, is_2D, cols, _ = Helper.initializeGraph(count)
        ids = list(self.data.keys())

        for i in range(count):
            spec = grid[i // cols, i % cols] if is_2D else grid[i]
            scope = self.data[ids[i]]['scope']
            if isinstance(scope, Scope):
                scope.trace(spec)
                scope.ax.set_title(ids[i])
            
        grid.tight_layout(plt.gcf(), h_pad=0.3, w_pad=0.2)

        filename = Helper.fileNameFromPath(self.path)
        args = 'measurements'

        Helper.save(filename, args)

    def trace(self, curves: list['Curve'], show_ax_labels = True, show_pt_labels = False, ignore_ids: list[str] = None):
        count = len(curves)
        grid, is_2D, cols, _ = Helper.initializeGraph(count)
        for curve in curves:
            curve.populate(self.data)

        for i in range(count):
            spec = grid[i // cols, i % cols] if is_2D else grid[i]
            curves[i].trace(spec, show_ax_labels, show_pt_labels, ignore_ids)

        filename = Helper.fileNameFromPath(self.path)
        args = [(curve.x_axis_entry, curve.y_axis_entry) for curve in curves]
        Helper.save(filename, args)

# ====================================
#               Curve
# ====================================


class Curve:
    raw_data: dict[str, dict[str, Value | Scope]]

    x_data: list[Value]
    y_data: list[Value]

    x_axis_entry: str
    y_axis_entry: str

    x_axis_label: str
    y_axis_label: str

    ax: plt.Axes

    point_labels: list[str]
    ids_to_index: dict[str, int]

    use_uncert: bool

    show_ids: bool

    def __init__(self, x_entry: str, y_entry: str, use_uncert = True, x_label: str = None, y_label: str = None, show_pt_ids = False):
        self.raw_data = None

        self.x_axis_entry = x_entry
        self.y_axis_entry = y_entry

        self.x_axis_label = x_label
        self.y_axis_label = y_label

        self.use_uncert = use_uncert

        self.point_labels = []
        self.ids_to_index = {}

        self.show_ids = show_pt_ids

        self.x_data = []
        self.y_data = []

    def populate(self, data: ExperienceData):
        '''Populate axes with `data`.'''
        for var in [self.x_axis_entry, self.y_axis_entry]:
            if var == 'graph':
                assert False, 'Use `Helper.traceMeasurementGraphs` to trace measurement graphs.'
            elif var not in list(data.values())[0]:
                assert False, f'Error while plotting the graph, unknown variable: "{var}"'

        self.x_data = [data[point][self.x_axis_entry] for point in data.keys()]
        self.y_data = [data[point][self.y_axis_entry] for point in data.keys()]
        self.raw_data = data

    def update(self, key: str, data: dict[str, Value]):
        '''Updates `data` of a populated graph.'''
        assert key in self.ids, f'Unknown key: {key}.'
        index = self.ids_to_index[key]
        self.x_data[index] = data[self.x_axis_entry]
        self.y_data[index] = data[self.y_axis_entry]

    def trace(self, spec: gridspec.SubplotSpec, show_ax_label = True, show_pt_label = False, ignore_ids: list[str] = None):
        self.ax = plt.subplot(spec)

        if show_ax_label:
            self.ax.set_xlabel(self.x_axis_label or self.x_axis_entry)
            self.ax.set_ylabel(self.y_axis_label or self.y_axis_entry)

        if ignore_ids is not None:
                sorted_x_data: list[float] = []
                sorted_x_uncert: list[float] = []
                sorted_y_data: list[float] = []
                sorted_y_uncert: list[float] = []

                for i, key in enumerate(self.raw_data):
                    if key in ignore_ids:
                        pass
                    else:
                        x: Value = self.raw_data[key][self.x_axis_entry]
                        y: Value = self.raw_data[key][self.y_axis_entry]
                        sorted_x_data.append(x.value)
                        sorted_x_uncert.append(x.uncert or 0)
                        sorted_y_data.append(y.value)
                        sorted_y_uncert.append(y.uncert or 0)

                        if show_pt_label:
                            self.ax.annotate(key, (x.value, y.value), textcoords='offset points', xytext=(0, 5), ha='center')

                self.ax.errorbar(sorted_x_data, sorted_y_data, yerr=sorted_y_uncert, xerr=sorted_x_uncert,
                                ecolor='firebrick', ls='', marker='+', mfc='royalblue', mec='royalblue',)

        else:
            x_data, x_uncert = Helper.unpackValueList(self.x_data)
            y_data, y_uncert = Helper.unpackValueList(self.y_data)
            self.ax.errorbar(
                x_data,
                y_data,
                yerr=y_uncert,
                xerr=x_uncert,
                ecolor='firebrick',
                ls='',
                marker='+',
                mfc='royalblue',
                mec='royalblue',
            )
            if show_pt_label:
                for i, key in enumerate(self.raw_data):
                    self.ax.annotate(key, (x_data[i], y_data[i]), textcoords='offset points', xytext=(0, 5), ha='center')

    def axis(self):
        return self.ax