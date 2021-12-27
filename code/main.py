from model.expressions import Expressions
from model.curve import Curve
from model.experiment import Experiment

if __name__ == '__main__':
    E = Experiment('measures/chutes5.csv', {
            'Ee': Expressions.Ee,
            'Ep': Expressions.Ep,
            'Ee_Ep': Expressions.Ee_Ep,
        },
        {
            'Ee': Expressions.u_Ee,
            'Ep': Expressions.u_Ep,
            'Ee_Ep': Expressions.u_Ee_Ep,
        })

    # E.traceMeasurementGraphs()

    # D = Helper.measurementsToData('tests/test_creneaux.csv')
    # print(Helper.computeIntegral(D['second'], D['volt'], scaling_function=lambda x: x * x) / 1)
    # Helper.trace(D)

    # D = Helper.measurementsToData('../tests/test_a_vide.csv')
    # print(Helper.computeIntegral(D['second'], D['volt'], scaling_function=lambda x: x * x) / 1)

    E.correctMeasurements(Curve('r', 'Ee_Ep'), expression = Expressions.Ee)
    
    # E.trace([Curve('r', 'Ee_Ep', True)])