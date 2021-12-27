from model.types import ExpressionsList, Data, Uncert
from model.corrector import Corrector
from model.expression import Expression
from model.curve import Curve
from model.helper import Helper
from matplotlib import pyplot as plt

class Experiment:
    """An experiment wrapper.
    
    Experiment related data and useful methods are packed here.
    """
    file: str
    expr_meas: ExpressionsList
    expr_uncert: ExpressionsList
    uncert: Uncert
    data: Data

    def __init__(self, file: str, measurements: ExpressionsList = {}, uncertainties: ExpressionsList = {}) -> None:
        self.file = file

        self.expr_meas = measurements
        self.expr_uncert = uncertainties

        self.data = Helper.fileToData(file)
        # print(self.data)
        self.uncert = dict([(x, {}) for x in self.data.keys()])

        Helper.computeExpressions('data', self.expr_meas, self.data)
        Helper.defaultUncertainties(self.data, self.uncert)
        Helper.computeExpressions('uncert', self.expr_uncert, self.data, self.uncert)

    def traceMeasurementGraphs(self, X: str = None, Y: str = None, ids: list[str] = []):
        """Traces Scope graphs of the current dataset."""
        try:
            list(self.data.values())[0]['graph']
        except:
            assert False, 'The current expriment doesn\'t have associated graph.'

        count = len(self.data) if ids == [] else len(ids)
        axs, is2D, c, r = Helper.initializeGraph(count)

        curves_ids = ids if ids != [] else list(self.data.keys())
        n = len(curves_ids)

        for i in range(c * r):
            ax = axs[i // c][i % c] if is2D else axs[i]

            if i < n:
                graph = self.data[curves_ids[i]]['graph']
                x_label, y_label = graph.keys()
                ax.plot(graph[X if X != None else x_label], graph[Y if Y != None else y_label])
                ax.set_title(curves_ids[i])
            else:
                ax.axis('off')

        plt.show()            
    
    def trace(self, curves: list[Curve]):
        """Traces `Curves` using the current dataset."""        
        n = len(curves)
        axs, is2D, c, r = Helper.initializeGraph(n)

        for curve in curves:
            curve.populate(self.data, self.uncert)

        l = len(self.data.keys())

        for i in range(c * r):
            ax = axs[i // c][i % c] if is2D else axs[i] if c * r > 1 else axs

            if i < n:
                curve = curves[i].plot()
                ax.scatter(curve['x'], curve['y'])
            
                ax.set_xlabel(curve['x_label'])
                ax.set_ylabel(curve['y_label'])

                valid_ux = len(curve['ux']) == l 
                valid_uy = len(curve['uy']) == l

                for j, id in enumerate(self.data.keys()):
                    ax.annotate(id, (curve['x'][j], curve['y'][j]), textcoords='offset points', xytext=(0, 5), ha='center')
                    if curve['unc'] and (valid_ux or valid_uy):
                        ax.errorbar(curve['x'][j], curve['y'][j], curve['uy'][j] if valid_uy else 0, curve['ux'][j] if valid_ux else 0, ecolor='firebrick')
            else:
                ax.axis('off')
            
        plt.show()

    def correctMeasurements(self, curve: Curve, corrected_values: list[str] = ['threshold', 'offset'], expression: Expression = None):
        """Starts a correction session for given values."""
        corrector = Corrector(self.file, self.data, expressions=self.expr_meas, corrected_values=corrected_values)
        corrector.start(curve, expression)
        