from inspect import signature
from typing import Callable

# pylint: disable = too-few-public-methods


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
