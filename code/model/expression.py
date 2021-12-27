from inspect import signature
from typing import Callable

class Expression:
    function: Callable[..., float]
    defaults: dict[str, float]
    arguments: list[str]
    def __init__(self, function: Callable[..., float], defaults: dict[str, float] = {}) -> None:
        self.function = function
        self.defaults = defaults
        self.arguments = [arg for arg in signature(function).parameters]

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
            assert value != None, f'Unknown variable: "{arg}"'
            args.append(value)
        return self.function(*args)