class Uncertainties:
    """Wrapper class for uncertainties calculation."""
    '''Wikipedia: https://fr.wikipedia.org/wiki/Propagation_des_incertitudes'''
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