from model.helper import Helper
# from model.constants import Constants
from model.expression import Expression
# from model.expressions import Expressions
from model.curve import Curve
from model.experiment import Experiment

if __name__ == '__main__':
    # E = Experiment('measures/chutes5.csv', {
    #     'Ee': Expressions.Ee,
    #     'Ep': Expressions.Ep,
    #     'Ee_Ep': Expressions.Ee_Ep,
    # },
    #     {
    #         'Ee': Expressions.u_Ee,
    #         'Ep': Expressions.u_Ep,
    #         'Ee_Ep': Expressions.u_Ee_Ep,
    # })

    # E = Experiment('measures/etalonage_piezo2.csv', {
    #     'Z': Expression(lambda e, u, r: abs(e - u) / abs(u) * r),
    #     'phi': Expression(lambda phi_deg: Helper.compute_angle(phi_deg, 'deg', 'rad')),
    # }, {})

    Helper.impedenceFromGraph('measures/etalonage_piezo3.csv', 200).plot()


    # E.traceMeasurementGraphs()

    # D = Helper.measurementsToData('tests/test_creneaux.csv')
    # print(Helper.computeIntegral(D['second'], D['volt'], scaling_function=lambda x: x * x) / 1)
    # Helper.trace(D)

    # D = Helper.measurementsToData('../tests/test_a_vide.csv')
    # print(Helper.computeIntegral(D['second'], D['volt'], scaling_function=lambda x: x * x) / 1)

    # E.correct_measurements(Curve('r', 'Ee_Ep'), expression=Expressions.Ee)

    # E.trace([Curve('f', 'Z'), Curve('f', 'phi')])

    # Helper.compute_angle(32, 'rad', 'deg')
