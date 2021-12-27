from typing import Union, Callable
from model.expression import Expression

Data = dict[str, dict[str, Union[float, dict]]]
Uncert = dict[str, dict[str, float]]
ExpressionsList = dict[str, Expression]
ScaleFunction = Callable[..., float]