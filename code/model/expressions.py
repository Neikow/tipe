from model.helper import Helper
from model.uncertainties import Uncertainties
from model.expression import Expression
from model.constants import Constants
from math import sqrt

class Expressions:
    """Useful expressions."""
    # Energie éléctrique aux bornes de la résistance.
    Ee = Expression(lambda graph, r, threshold, offset: Helper.computeIntegral(graph['second'], graph['volt'], scaling_function=lambda x: (x + offset) ** 2, threshold=threshold) / r, {'threshold': 0, 'offset': 0})
    
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

    u_T = Expression(lambda T, u_h: Uncertainties.power(T ** 2, 1/2, u_h))

    # Rapport Ee / Ep.
    Ee_Ep = Expression(lambda Ee, Ep: Ee / Ep)

    # Incertitude rapport Ee/Ep.
    u_Ee_Ep = Expression(lambda Ee, Ep, u_Ee, u_Ep: Uncertainties.quotient(Ee, Ep, u_Ee, u_Ep))

    # Incertitude sur R:
    u_r = Expression(lambda: Constants.u_r)