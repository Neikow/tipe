import inspect
from math import ceil, sqrt
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import simpson
from typing import Callable, Dict, List, Literal, Union
from model.expression import Expression
from model.constants import Constants
from model.types import Data, ExpressionsList, Uncert

class Helper:
    """Helper class handling useful calculations and functions."""
    @staticmethod
    def fileToData(file: str, data_title_line: int = 0, measurements_file_prefix: str = 'scope_', measurements_title_line: int = 1) -> dict[str, dict[str, Union[float, dict]]]:
        """Convers a measurments data file to python dictionary."""
        with open(file) as f:
            global_data = {}
            lines = f.readlines()
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

                if id_index == None:
                    _id = i
                else:
                    _id = values[id_index].strip()
                
                local_data = {}
                for j, entry in enumerate(entries):
                    if entry == 'id': continue
                    local_data[entry] = float(values[j])
                    if id_index != None:
                        local_data['graph'] = Helper.measurementsToData(f"{file.strip().removesuffix('.csv')}/{measurements_file_prefix}{_id}.csv", measurements_title_line)

                global_data[_id] = local_data

        # print(global_data)
        return global_data
    
    @staticmethod
    def measurementsToData(file: str, measurements_title_line: int = 1):
        """Converts measurments from a Scope to readable dict."""
        with open(file) as f:
            lines = f.readlines()
            entries = [x.strip().lower() for x in lines[measurements_title_line].split(',')]
            n = len(entries)
            temp: dict[str, list[float]] = {}
            for i in range(n):
                temp[entries[i]] = []
            for line in lines[measurements_title_line + 1:]:
                if line.strip().startswith('#'): continue
                try:
                    values = [float(x) for x in line.split(',')]
                    for i in range(n):
                        temp[entries[i]].append(values[i])
                except:
                    pass

            data: dict[str, list[float]] = {}
            for i in range(n):
                data[entries[i]] = np.array(temp[entries[i]]) 

        return data            

    @staticmethod
    def computeIntegral(X: list[float], Y: list[float], threshold: float = 0.0, scaling_function: Callable[..., float] = lambda y: y, scaling_aguments: list[float] = []):
        """Calculates the integral for a given graph.

        `Y` axis values can be scaled using the `scaling_function` attribute.
        """

        def aboveThreshhold(val: float):
            return abs(val) >= abs(threshold)

        assert len(X) == len(Y), "The two lists lengths don't match."

        assert len(inspect.signature(scaling_function).parameters) == len(scaling_aguments) + 1, "The argument count for the scaling function is wrong."
        
        for i in range(len(Y)):
            if not aboveThreshhold(Y[i]):
                Y[i] = 0

        Y = scaling_function(Y, *scaling_aguments)

        integral: float = simpson(Y, X)
        return integral

    @staticmethod
    def computeExpressions(type: Literal['data', 'uncert'], expressions: ExpressionsList, data: Data, uncert: Uncert = None, id: str = None):
        """Calculates the value (or uncertainty) of a dataset entry using data from the dataset and a given expression."""
        
        assert (type == 'uncert' and uncert) or not uncert, 'Uncertainties are required.'
        
        _data = uncert if type == 'uncert' else data

        for (expr_key, expression) in expressions.items():
            if (id != None):
                if type == 'uncert':
                    try:
                        _data[id][expr_key] = expression(data[id], uncert[id])
                    except:
                        pass
                else:        
                    _data[id][expr_key] = expression(data[id])
            else:
                for (_id) in data.keys():
                    assert not expr_key in _data[_id].keys(), f'The key "{expr_key}" already exists in the provided data.'
                    _data[_id][expr_key] = expression(data[_id], uncert[_id] if (type == 'uncert') else None)
    
    @staticmethod
    def defaultUncertainties(data: Data, uncert: Uncert):
        """Returns default uncertainties for certain dataset entries."""

        DefaultUncertainties = [
            ('h', Expression(lambda: Constants.u_h)),
            ('r', Expression(lambda: Constants.u_r))
        ]
        for (expr_key, expression) in DefaultUncertainties:
            variables = list(list(data.values())[0].keys())
            if expr_key in variables:
                for (id) in data.keys():
                    uncert[id][expr_key] = expression(data[id], uncert[id])

    @staticmethod
    def getGlobalGraphMinMax(data: dict[str, dict[str, Union[float, dict]]], ax: Literal['x', 'y'] = 'y'):
        """Computes graph's extremum values for X and Y axes on the whole dataset."""
        
        ax_key = list(data[list(data.keys())[0]]['graph'].keys())[{'x': 0, 'y': 1}[ax]]
        min: float = None
        max: float = None
        
        def localMinMax(list: list[float]):
            _min: float = list[0]
            _max: float = list[0]

            for x in list:
                if x < _min:
                    _min = x
                if x > _max:
                    _max = x

            return (_min, _max) 

        for _data in data.values():
            local_min, local_max = localMinMax(_data['graph'][ax_key])
            if min == None or local_min < min:
                min = local_min
            if max == None or local_max > max:
                max = local_max
        return (min, max)

    def safeListCopy(o: list) -> list:
        """Returns a shallow copy of a python `list`."""
        copy: list = []
        for value in o:
            if isinstance(value, Dict):
                copy.append(Helper.safeDictCopy(o))
            elif isinstance(value, List):
                copy.append(Helper.safeListCopy(o))
            else:
                copy.append(value)
        return copy
            
    @staticmethod
    def safeDictCopy(o: dict) -> dict:
        """Returns a shallow copy of a python `dict`."""
        copy: dict = {}
        for key, value in o.items():
            if isinstance(value, Dict):
                copy[key] = Helper.safeDictCopy(value)
            elif isinstance(value, List):
                copy[key] = Helper.safeListCopy(value)
            else:
                copy[key] = value
        return copy

    @staticmethod
    def trace(data: dict[str, list[float]], scaling_function = lambda x: x):
        """Traces a graph using `matplolib.pyplot`"""

        x, y = list(data.keys())
        X = np.array(data[x])
        Y = scaling_function(np.array(data[y]))

        plt.plot(X, Y)

        plt.show()

    @staticmethod
    def initializeGraph(n: int):
        """Computes the graph matrix size."""
        columns = ceil(sqrt(n))
        rows = ceil(n / columns)
        is2D = rows > 1
        return plt.subplots(rows, columns)[1], is2D, columns, rows



