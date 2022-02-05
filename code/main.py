from matplotlib.pyplot import xlabel
import numpy as np
from newton import *

class Tests:
    '''Various tests for this program.'''
    @staticmethod
    def FingerShock():
        '''Testing the plotting algorithm.'''
        D = Helper.measurementsToScopeData('measures/tests/finger_shock.csv')[0]

        D.plot()

    @staticmethod
    def Empty():
        '''Measuring the impact of noise on the integral by applying no tension on the resistor.'''
        R = 1
        D = Helper.measurementsToScopeData('measures/tests/empty_noise_test.csv')[0]

        Helper.expect(Helper.computeIntegral(D.x_data, D.y_data, scaling_function=lambda x: x * x / R), 0, 'J')
        D.plot()

    @staticmethod
    def Square():
        '''Generating a single square pulse (1.0V, 0.1s) and calculating it's energy to test the integration algorithm.'''
        R = 1
        D = Helper.measurementsToScopeData('measures/tests/square_signal.csv')[0]

        Helper.expect(Helper.computeIntegral(D.x_data, D.y_data, scaling_function=lambda x: x * x / R), 0.1, 'J')
        D.plot()


class Calibrations:
    '''Various calibrations to learn more about our equipment.'''
    @staticmethod
    def Calibration1():
        '''We measured tension on the generator and a resistor to calculate the impedence.'''
        E = Experiment('measures/piezo_calibration1.csv', {
            'Z': Expression(lambda e, u, r: abs(e - u) / abs(u) * r),
            'phi': Expression(lambda phi_deg: Helper.computeAngle(phi_deg, 'deg', 'rad')),
        }, {})

        E.trace([Curve('f', 'Z')])

    @staticmethod
    def Calibration2():
        '''We measured tension on the generator and a resistor to calculate the impedence.'''
        E = Experiment('measures/piezo_calibration2.csv', {
            'Z': Expression(lambda e, u, r: abs(e - u) / abs(u) * r),
            'phi': Expression(lambda phi_deg: Helper.computeAngle(phi_deg, 'deg', 'rad')),
        }, {})

        curve = Curve('f', 'Z')
        E.trace([curve])

        print(curve.x_data, curve.y_data)

    @staticmethod
    def Calibration3(scope_id: int):
        '''We created an impedence-meter from a resistor (200 ohm), a generator
           and two low-pass filters.

           The measures were taken automatically over a period of 2s with a frequency
           varying linearly between two values.

           The filter is an [envelope detector](https://fr.wikipedia.org/wiki/Circuit_d%C3%A9tecteur_d%27enveloppe),
           made with a 1k ohm resistor, 470nF capactitor and a diode.
        '''
        R = 200

        freq_table = {
            1: [10e3, 750e3],
            2: [5, 750e3],
            3: [10e3, 1e6],
            4: [10e3, 500e3]
        }
        S = Helper.impedenceFromGraph(f'measures/calibrations/calib_{scope_id}.csv', R, *freq_table[scope_id])

        # print(list(S.y_data), list(S.x_data))

        # scope_1 coefficients
        print(Helper.coefficient('k31', 1.33e5, 1.47e5))
        print(Helper.coefficient('k31', 1.96e5, 2.10e5))

        # scope_2 coefficients
        print(Helper.coefficient('k31', 1.33e5, 1.47e5))
        print(Helper.coefficient('k31', 1.96e5, 2.13e5))
        print(Helper.coefficient('k31', 2.85e5, 3.10e5))

        # plt.plot(S.x_data, 1 / S.y_data)
        S.plot(False, True)




class Experiments:
    '''Experiments conducted during the project.'''
    @staticmethod
    def Experiment1(default: bool = True):
        '''???'''
        E = Experiment('measures/exp1.csv', {})

        if default:
            E.traceMeasurementGraphs()

        return E

    @staticmethod
    def Experiment2(default: bool = True):
        '''???'''
        E = Experiment('measures/exp2.csv', {})

        if default:
            E.traceMeasurementGraphs()

        return E

    @staticmethod
    def Experiment3(default: bool = True):
        '''Dropping a metal ball on a piezo from a chaning height but measuring the impact with a fixed resistance.'''
        magnet_height = 27.9E-2

        E = Experiment('measures/exp3.csv', {
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
            E.traceMeasurementGraphs()
            E.trace([Curve('h', 'Ee'), Curve('h', 'Ee_Ep')])

        return E

    @staticmethod
    def Experiment4(default: bool = True):
        '''Dropping a metal ball on a piezo from a fixed height but measuring the impact with a changing resistance.'''
        E = Experiment('measures/exp4.csv', {
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
            E.traceMeasurementGraphs()
            E.trace([Curve('r', 'Ee'), Curve('r', 'Ee_Ep')])

        return E

    @staticmethod
    def Experiment5(default: bool = True):
        E = Experiment('measures/exp5.csv', {
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
            # E.traceMeasurementGraphs()
            h_Ee = Curve('h', 'Ee')
            h_Ee.populate(E.data)
            h_Ee_Ep = Curve('h', 'Ee_Ep')
            h_Ee_Ep.populate(E.data)
            # E.trace([h_Ee, h_Ee_Ep])

            X1 = h_Ee_Ep.x_data
            Y1 = h_Ee_Ep.y_data
            X2 = h_Ee.x_data
            Y2 = h_Ee.y_data

            X1_label = h_Ee_Ep.x_label
            Y1_label = h_Ee_Ep.y_label
            X2_label = h_Ee.x_label
            Y2_label = h_Ee.y_label

            fig, axs = plt.subplots(2)

            coef1 = np.polyfit(X1,Y1,1)
            poly1d_fn1 = np.poly1d(coef1)
            
            axs[0].plot(X1,Y1, 'yo', X1, poly1d_fn1(Y1), '--k')
            axs[0].set(ylabel=Y1_label, xlabel=X1_label)

            coef2 = np.polyfit(X2,Y2,1)
            poly1d_fn2 = np.poly1d(coef2)
            
            axs[1].plot(X2,Y2, 'yo', X2, poly1d_fn2(X2), '--k')
            axs[1].set(ylabel=Y2_label, xlabel=X2_label)

            plt.show()

        E.trace([Curve('Ee', 'Ee_Ep'), Curve('Ep', 'Ee_Ep')])

            


if __name__ == '__main__':
    Helper.init()
    # ================================
    #              Tests
    # ================================

    # Tests.Empty()
    # Tests.Square()

    # ================================
    #          Calibrations
    # ================================

    # Calibrations.Calibration1()
    # Calibrations.Calibration2()
    # Calibrations.Calibration3(1)
    # Calibrations.Calibration3(2)
    # Calibrations.Calibration3(3)
    # Calibrations.Calibration3(4)

    # ================================
    #           Experiments
    # ================================

    # Experiments.Experiment1()
    # Experiments.Experiment2()
    # Experiments.Experiment3()
    # Experiments.Experiment4()
    # Experiments.Experiment5(False)

    D = Helper.fileToData('measures/standalone.csv')
    Helper.computeExpressions('data', {
        'Ee': Expressions.Ee, 'Ep': Expressions.Ep, 'Ee_Ep': Expressions.Ee_Ep
    }, D)
    
    print(D)