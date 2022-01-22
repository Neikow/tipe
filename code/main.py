# from numpy import cos, exp, inf
from xmlrpc.client import boolean
from model.helper import Helper
# from model.constants import Constants
from model.expression import Expression
from model.expressions import Expressions
from model.curve import Curve
from model.experiment import Experiment


class Tests:
    '''Various tests for this program.'''
    @staticmethod
    def Empty():
        '''Measuring the impact of noise on the integral by applying no tension on the resistor.'''
        R = 1
        D = Helper.measurementsToScopeData('measures/tests/test_a_vide.csv')[0]

        Helper.expect(Helper.computeIntegral(D.x_data, D.y_data, scaling_function=lambda x: x * x / R), 0, 'J')
        D.plot()

    @staticmethod
    def Square():
        '''Generating a single square pulse (1.0V, 0.1s) and calculating it's energy to test the integration algorithm.'''
        R = 1
        D = Helper.measurementsToScopeData('measures/tests/test_creneaux.csv')[0]

        Helper.expect(Helper.computeIntegral(D.x_data, D.y_data, scaling_function=lambda x: x * x / R), 0.1, 'J')
        D.plot()


class Calibrations:
    '''Various calibrations to learn more about our equipment.'''
    @staticmethod
    def Calibration1():
        '''We measured tension on the generator and a resistor to calculate the impedence.'''
        E = Experiment('measures/etalonage_piezo1.csv', {
            'Z': Expression(lambda e, u, r: abs(e - u) / abs(u) * r),
            'phi': Expression(lambda phi_deg: Helper.computeAngle(phi_deg, 'deg', 'rad')),
        }, {})

        E.trace([Curve('f', 'Z')])

    @staticmethod
    def Calibration2():
        '''We measured tension on the generator and a resistor to calculate the impedence.'''
        E = Experiment('measures/etalonage_piezo2.csv', {
            'Z': Expression(lambda e, u, r: abs(e - u) / abs(u) * r),
            'phi': Expression(lambda phi_deg: Helper.computeAngle(phi_deg, 'deg', 'rad')),
        }, {})

        E.trace([Curve('f', 'Z')])

    @staticmethod
    def Calibration3(scope_id: int):
        '''We created an impedence-meter from a resistor (200 ohm), a generator
           and two low-pass filters.

           The measures were taken automatically over a period of 2s.

           The filter is an envelope detector, made with a 1k ohm resistor,
           470nF capactitor and a diode.
        '''
        R = 200

        freq_table = {
            1: [10e3, 750e3],
            2: [5, 750e3],
            3: [10e3, 1e6],
            4: [10e3, 500e3]
        }
        S = Helper.impedenceFromGraph(f'measures/calibrations/scope_{scope_id}.csv', R, *freq_table[scope_id])

        # scope_1 coefficients
        print(Helper.coefficient('k31', 1.33e5, 1.47e5))
        print(Helper.coefficient('k31', 1.96e5, 2.10e5))

        # scope_2 coefficients
        print(Helper.coefficient('k31', 1.33e5, 1.47e5))
        print(Helper.coefficient('k31', 1.96e5, 2.13e5))
        print(Helper.coefficient('k31', 2.85e5, 3.10e5))

        S.plot(False, True)


class Experiments:
    '''Experiments conducted during the project.'''
    @staticmethod
    def Experiment1(default: boolean = True):
        '''???'''
        E = Experiment('measures/chutes.csv', {})

        if default:
            E.traceMeasurementGraphs()

        return E

    @staticmethod
    def Experiment2(default: boolean = True):
        '''???'''
        E = Experiment('measures/chutes2.csv', {})

        if default:
            E.traceMeasurementGraphs()

        return E

    @staticmethod
    def Experiment3(default: boolean = True):
        '''Dropping a metal ball on a piezo from a chaning height but measuring the impact with a fixed resistance.'''
        magnet_height = 27.9E-2

        E = Experiment('measures/chutes3.csv', {
            'h': Expression(lambda h_offset: magnet_height - h_offset),
            'Ee': Expressions.Ee,
            'Ep': Expressions.Ep,
            'Ee_Ep': Expressions.Ee_Ep,
        },
            {
            'Ee': Expressions.u_Ee,
            'Ep': Expressions.u_Ep,
            'Ee_Ep': Expressions.u_Ee_Ep,
        })

        if default:
            E.trace([Curve('h', 'Ee'), Curve('h', 'Ee_Ep')])

        return E

    @staticmethod
    def Experiment4(default: boolean = True):
        '''Dropping a metal ball on a piezo from a fixed height but measuring the impact with a changing resistance.'''
        E = Experiment('measures/chutes4.csv', {
            'Ee': Expressions.Ee,
            'Ep': Expressions.Ep,
            'Ee_Ep': Expressions.Ee_Ep,
        },
            {
            'Ee': Expressions.u_Ee,
            'Ep': Expressions.u_Ep,
            'Ee_Ep': Expressions.u_Ee_Ep,
        })

        if default:
            E.trace([Curve('r', 'Ee'), Curve('r', 'Ee_Ep')])

        return E


if __name__ == '__main__':
    # ================================
    #              Tests
    # ================================

    # Tests.Empty()
    Tests.Square()

    # ================================
    #          Calibrations
    # ================================

    # Calibrations.Calibration1()
    # Calibrations.Calibration2()
    # Calibrations.Calibration3(1)

    # ================================
    #           Experiments
    # ================================

    # Experiments.Experiment1()
    # Experiments.Experiment2()
    # Experiments.Experiment3()
    # Experiments.Experiment4()

    ...
