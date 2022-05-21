from matplotlib.pyplot import xlabel, ylabel
from tipe.piezo import *
os.chdir('E:\\OneDrive\\.school\\.tipe\\code')


class Tests:
    '''Various tests for this program.'''
    @staticmethod
    def FingerShock():
        '''Testing the plotting algorithm.'''
        S = Helper.measurementsToScope('measures/tests/finger_shock.csv')[0]
        S.trace(Helper.initializeGraph(1)[0][0], x_label=r'$T$ (s)', y_label=r'$U$ (V)')
        Helper.save('finger_shock')

    @staticmethod
    def Empty():
        '''Measuring the impact of noise on the integral by applying no tension on the resistor.'''
        R = 1
        S = Helper.measurementsToScope('measures/tests/empty_noise_test.csv')[0]

        Helper.expect(Helper.computeIntegral(S.x_data, S.y_data, scaling=lambda x: x * x / R), 0, 'J')
        S.trace(Helper.initializeGraph(1)[0][0], x_label=r'$T$ (s)', y_label=r'$U$ (V)')
        Helper.save('empty_noise')

    @staticmethod
    def Square():
        '''Generating a single square pulse (1.0V, 0.1s) and calculating it's energy to test the integration algorithm.'''
        R = 1
        S = Helper.measurementsToScope('measures/tests/square_signal.csv')[0]

        Helper.expect(Helper.computeIntegral(S.x_data, S.y_data, scaling=lambda x: x ** 2 / R), 0.1, 'J')
        S.trace(Helper.initializeGraph(1)[0][0], x_label=r'$T$ (s)', y_label=r'$U$ (V)')
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

        E.trace([Curve('f', 'Z', x_label=r'$f$ (Hz)', y_label=r'$Z$ ($\Omega$)')])

    @staticmethod
    def Calibration2():
        '''I measured the tension on the generator and a resistor to calculate the impedence.'''
        E = Experiment('measures/piezo_calibration2.csv', {
            'Z': Expression(lambda e, u, r: abs(e - u) / abs(u) * r),
            'phi': Expression(lambda phi_deg: Helper.angle(phi_deg, 'deg', 'rad')),
        })

        E.trace([Curve('f', 'Z', x_label=r'$f$ (Hz)', y_label=r'$Z$ ($\Omega$)')])

    @staticmethod
    def Calibration3(scope_id: int, show_coefficients=False):
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

        if show_coefficients:
            print('k33')
            # scope_1 coefficients
            print(Helper.coefficient('k33', 1.33e5, 1.47e5))
            print(Helper.coefficient('k33', 1.96e5, 2.10e5))

            # scope_2 coefficients
            print(Helper.coefficient('k33', 1.33e5, 1.47e5))
            print(Helper.coefficient('k33', 1.96e5, 2.13e5))
            print(Helper.coefficient('k33', 2.85e5, 3.10e5))

            print('k31')
            # scope_1 coefficients
            print(Helper.coefficient('k31', 1.33e5, 1.47e5))
            print(Helper.coefficient('k31', 1.96e5, 2.10e5))

            # scope_2 coefficients
            print(Helper.coefficient('k31', 1.33e5, 1.47e5))
            print(Helper.coefficient('k31', 1.96e5, 2.13e5))
            print(Helper.coefficient('k31', 2.85e5, 3.10e5))

        S.trace(Helper.initializeGraph(1)[0][0], x_label=r'$f$ (Hz)', y_label=r'$Z_\pi$ ($\Omega$)')
        Helper.save(f'piezo_calibration3_{scope_id}')

    def Calibration4(scope_id: int):
        E, U = Helper.measurementsToScope(f'measures/calibrations/calib_{scope_id}.csv')

        freq_table = {
            1: [10e3, 750e3],
            2: [5, 750e3],
            3: [10e3, 1e6],
            4: [10e3, 500e3]
        }

        spec = Helper.initializeGraph(1)[0][0]
        E.trace(spec, x_label='$f$ (Hz)', y_label='$E$,$U$ (V)', x_minmax=freq_table[scope_id], label='$E$')
        E.traceOther(U, color='crimson', label='$U$')
        E.addLegend()
        Helper.save(f'piezo_calibration4_{scope_id}')


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

        E.trace([Curve('h', 'Ee', x_label=r'$h$ (m)', y_label=r'$E_e$ (J)'), Curve(
            'h', 'Ee_Ep', x_label=r'$h$ (m)', y_label=r'$k^{2}_{33}$')])

    @staticmethod
    def Experiment4():
        '''Dropping a metal ball on a piezo from a fixed height but measuring the impact with a changing resistance.'''
        E = Experiment('measures/exp4.csv', {
            'Ee': Expressions.Ee,
            'Ep': Expressions.Ep,
            'Ee_Ep': Expressions.EeOverEp,
        })

        E.trace([
            Curve('r', 'Ee_Ep', x_label=r'$R$ ($\Omega$)', y_label=r'$k^{2}_{33}$'),
        ], show_pt_labels=True, ignore_ids=['62', '63', '64', '65', '66', '67', '68', '69', '70', '71'])


        E.trace([Curve('r', 'Ee', x_label=r'$R$ ($\Omega$)', y_label=r'$E_e$ (J)'), Curve('r', 'Ee_Ep', x_label=r'$R$ ($\Omega$)',
                y_label=r'$k^{2}_{33}$')], ignore_ids=['62', '63', '64', '65', '66', '67', '68', '69', '70', '71'])

    @staticmethod
    def Experiment5():
        E = Experiment('measures/exp5.csv', {
            'Ee': Expressions.Ee,
            'Ep': Expressions.Ep,
            'Ee_Ep': Expressions.EeOverEp,
        })

        E.saveMeasurementScopes(show_title=False, ticksize=5)

        ignore_ids = ['1', '12', '23', '24', '34', '35']

        E.trace([
            Curve('h', 'Ee', x_label=r'$h$ (m)', y_label=r'$E_e$ (J)'),
        ], ignore_ids=ignore_ids)
        E.trace([
            Curve('h', 'Ee_Ep', x_label=r'$h$ (m)', y_label=r'$k^{2}_{33}$'),
        ], ignore_ids=ignore_ids)
        E.trace([
            Curve('h', 'Ee', x_label=r'$h$ (m)', y_label=r'$E_e$ (J)'),
            Curve('h', 'Ee_Ep', x_label=r'$h$ (m)', y_label=r'$k^{2}_{33}$'),
        ], ignore_ids=ignore_ids)
        E.trace([
            Curve('Ee', 'Ee_Ep', x_label=r'$E_e$ (J)', y_label=r'$k^{2}_{33}$'),
            Curve('h', 'Ee_Ep', x_label=r'$h$ (m)', y_label=r'$k^{2}_{33}$')
        ], ignore_ids=ignore_ids)

    @staticmethod
    def Experiment6():
        V1, E1 = Helper.measurementsToScope('voltmeter\log-2022-05-21-15-57-53.csv', measurements_title_line=0)
        V1.trace(Helper.initializeGraph(1)[0][0], x_label=r'Temps (s)', y_label=r'Tension (V)')
        Helper.save('voltmeter voltage 5min (+10g)')

        E1.trace(Helper.initializeGraph(1)[0][0], x_label=r'Temps (s)', y_label=r'$E_{e}$ (J)')
        Helper.save('voltmeter energy 5min (+10g)')
        
        V2, E2 = Helper.measurementsToScope('voltmeter\log-2022-05-21-16-23-27.csv', measurements_title_line=0)
        V2.trace(Helper.initializeGraph(1)[0][0], x_label=r'Temps (s)', y_label=r'Tension (V)')
        Helper.save('voltmeter voltage 5min (+0g)')
        
        E2.trace(Helper.initializeGraph(1)[0][0], x_label=r'Temps (s)', y_label=r'$E_{e}$ (J)')
        Helper.save('voltmeter energy 5min (+0g)')

        V3, E3  =Helper.measurementsToScope('voltmeter/log-2022-05-21-16-56-13.csv', measurements_title_line=0)
        V3.trace(Helper.initializeGraph(1)[0][0], x_label=r'Temps (s)', y_label=r'Tension (V)')
        Helper.save('voltmeter voltage 5min (+5g)')
        
        E3.trace(Helper.initializeGraph(1)[0][0], x_label=r'Temps (s)', y_label=r'$E_{e}$ (J)')
        Helper.save('voltmeter energy 5min (+5g)')

        E1.trace(Helper.initializeGraph(1)[0][0], label=r'$10$ g', x_label=r'Temps (s)', y_label=r'$E_{e}$ (J)')
        E1.traceOther(E2, color='crimson', label=r'$0$ g')
        E1.addLegend()
        Helper.save('voltmeter energy 5 min both')

        E1.trace(Helper.initializeGraph(1)[0][0], label=r'$10$ g', x_label=r'Temps (s)', y_label=r'$E_{e}$ (J)')
        E1.traceOther(E3, color='forestgreen', label=r'$5$ g')
        E1.traceOther(E2, color='crimson', label=r'$0$ g')
        E1.addLegend()
        Helper.save('voltmeter energy 5 min all three')

if __name__ == '__main__':
    # ================================
    #              Tests
    # ================================

    # Tests.FingerShock()
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
    # Calibrations.Calibration4(1)
    # Calibrations.Calibration4(2)
    # Calibrations.Calibration4(3)
    # Calibrations.Calibration4(4)

    # ================================
    #           Experiments
    # ================================

    # Experiments.Experiment1()
    # Experiments.Experiment2()
    # Experiments.Experiment3()
    # Experiments.Experiment4()
    # Experiments.Experiment5()
    Experiments.Experiment6()

    # ================================
    #           Charging
    # ================================

    # Voltmeter().live(average_on=10)
