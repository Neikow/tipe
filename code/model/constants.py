# pylint: disable = too-few-public-methods
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

