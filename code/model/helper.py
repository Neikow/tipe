import inspect
from math import ceil, sqrt
from typing import Callable, Literal, Union
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import simpson
from model.experience_data import ExperienceData
from model.scope import Scope, Data
from model.expression import Expression, ExpressionsList
from model.constants import Constants
from model.types import Uncertainty


class Helper:
    """Helper class handling useful calculations and functions."""
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
                            for k in range(len(scopes)):
                                local_data[f'graph{k}'] = scopes[k]
                        else:
                            local_data['graph'] = scopes[0]
                global_data[_id] = local_data

        return global_data

    @staticmethod
    def measurementsToScopeData(path: str, measurements_title_line: int = 1) -> list[Scope]:
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

        return [Scope({
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

        assert len(inspect.signature(scaling_function).parameters) == len(
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
    def getGlobalScopeMinMax(data: dict[str, dict[str, Union[float, Scope]]], axis: Literal['x', 'y'] = 'y'):
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
            copy.set_data_keys(obj.getDataKeys())
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
    def trace(data: dict[str, list[float]], scaling_function=lambda x: x):
        """Traces a graph using `matplolib.pyplot`"""

        x_label, y_label = list(data.keys())
        x_data = np.array(data[x_label])
        y_data = scaling_function(np.array(data[y_label]))

        plt.plot(x_data, y_data)

        plt.show()

    @staticmethod
    def initializeGraph(plots_count: int):
        """Computes the graph matrix size."""
        columns = ceil(sqrt(plots_count))
        rows = ceil(plots_count / columns)
        is_graph_2d = rows > 1
        return plt.subplots(rows, columns)[1], is_graph_2d, columns, rows

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

        return Scope({'f': np.array(F), 'Z': np.array(Z)})
