from tipe.piezo import *
os.chdir('E:\OneDrive\.school\.tipe\code')


class Tests:
    '''Various tests for this program.'''
    @staticmethod
    def FingerShock():
        '''Testing the plotting algorithm.'''
        D = Helper.measurementsToScope('measures/tests/finger_shock.csv')[0]
        D.trace(Helper.initializeGraph(1)[0][0])
        Helper.save('finger_shock')

    @staticmethod
    def Empty():
        '''Measuring the impact of noise on the integral by applying no tension on the resistor.'''
        R = 1
        D = Helper.measurementsToScope('measures/tests/empty_noise_test.csv')[0]

        Helper.expect(Helper.computeIntegral(D.x_data, D.y_data, scaling=lambda x: x * x / R), 0, 'J')
        D.trace(Helper.initializeGraph(1)[0][0])
        Helper.save('empty_noise')

    @staticmethod
    def Square():
        '''Generating a single square pulse (1.0V, 0.1s) and calculating it's energy to test the integration algorithm.'''
        R = 1
        D = Helper.measurementsToScope('measures/tests/square_signal.csv')[0]

        Helper.expect(Helper.computeIntegral(D.x_data, D.y_data, scaling=lambda x: x * x / R), 0.1, 'J')
        D.trace(Helper.initializeGraph(1)[0][0])
        Helper.save('square')

class Calibrations:
    '''Various calibrations to learn more about our equipment.'''
    @staticmethod
    def Calibration1():
        '''I measured the tension of the generator and a resistor to calculate the impedence.'''
        E = Experiment('measures/piezo_calibration1.csv', {
            'Z': Expression(lambda e, u, r: abs(e - u) / abs(u) * r),
            'phi': Expression(lambda phi_deg: Helper.angle(phi_deg, 'deg', 'rad')),
        })

        E.trace([Curve('f', 'Z')])

    @staticmethod
    def Calibration2():
        '''I measured the tension on the generator and a resistor to calculate the impedence.'''
        E = Experiment('measures/piezo_calibration2.csv', {
            'Z': Expression(lambda e, u, r: abs(e - u) / abs(u) * r),
            'phi': Expression(lambda phi_deg: Helper.angle(phi_deg, 'deg', 'rad')),
        })

        E.trace([Curve('f', 'Z')])

    @staticmethod
    def Calibration3(scope_id: int):
        '''I created an impedence-meter from a resistor (200 ohm), a generator
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
        S = Helper.impedenceFromScope(f'measures/calibrations/calib_{scope_id}.csv', R, *freq_table[scope_id])

        # scope_1 coefficients
        print(Helper.coefficient('k31', 1.33e5, 1.47e5))
        print(Helper.coefficient('k31', 1.96e5, 2.10e5))

        # scope_2 coefficients
        print(Helper.coefficient('k31', 1.33e5, 1.47e5))
        print(Helper.coefficient('k31', 1.96e5, 2.13e5))
        print(Helper.coefficient('k31', 2.85e5, 3.10e5))

        S.trace(Helper.initializeGraph(1)[0][0])
        Helper.save(f'piezo_calibration3-{scope_id}')

class Experiments:
    '''Experiments conducted during the project.'''
    @staticmethod
    def Experiment1():
        '''???'''
        E = Experiment('measures/exp1.csv', {})
        E.saveMeasurementScopes()

    @staticmethod
    def Experiment2():
        '''???'''
        E = Experiment('measures/exp2.csv', {})
        E.saveMeasurementScopes()    

    @staticmethod
    def Experiment3():
        '''Dropping a metal ball on a piezo from a chaning height but measuring the impact with a fixed resistance.'''
        magnet_height = 27.9E-2

        E = Experiment('measures/exp3.csv', {
            'h': Expression(lambda h_offset: magnet_height - h_offset),
            'Ee': Expressions.Ee,
            'Ep': Expressions.Ep,
            'Ee_Ep': Expressions.EeOverEp,
        })

        E.trace([Curve('h', 'Ee'), Curve('h', 'Ee_Ep')])

    @staticmethod
    def Experiment4():
        '''Dropping a metal ball on a piezo from a fixed height but measuring the impact with a changing resistance.'''
        E = Experiment('measures/exp4.csv', {
            'Ee': Expressions.Ee,
            'Ep': Expressions.Ep,
            'Ee_Ep': Expressions.EeOverEp,
        })

        E.trace([Curve('r', 'Ee'), Curve('r', 'Ee_Ep')], show_pt_labels=False, ignore_ids=['62', '63', '64', '65', '66', '67', '68', '69', '70', '71'])

    @staticmethod
    def Experiment5(default: bool = True):
        E = Experiment('measures/exp5.csv', {
            'Ee': Expressions.Ee,
            'Ep': Expressions.Ep,
            'Ee_Ep': Expressions.EeOverEp,
        })

        E.saveMeasurementScopes()
        E.trace([Curve('Ee', 'Ee_Ep'), Curve('h', 'Ee_Ep')])


if __name__ == '__main__':
    # ================================
    #              Tests
    # ================================

    Tests.Empty()
    Tests.Square()

    # ================================
    #          Calibrations
    # ================================

    Calibrations.Calibration1()
    Calibrations.Calibration2()
    Calibrations.Calibration3(1)
    Calibrations.Calibration3(2)
    Calibrations.Calibration3(3)
    Calibrations.Calibration3(4)

    # ================================
    #           Experiments
    # ================================

    Experiments.Experiment1()
    Experiments.Experiment2()
    Experiments.Experiment3()
    Experiments.Experiment4()
    Experiments.Experiment5()

    # D = Helper.fileToData('measures/standalone.csv')
    # Helper.computeExpressions('data', {
    #     'Ee': Expressions.Ee, 'Ep': Expressions.Ep, 'Ee_Ep': Expressions.Ee_Ep
    # }, D)

    # ================================
    #           Charging
    # ================================
    # Voltmeter('COM4').graph(average_on=10)