from model.scope import Scope
from model.experience_data import ExperienceData
from model.expression import ExpressionsList
from model.types import Uncertainty
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
    uncert: Uncertainty
    data: ExperienceData

    def __init__(self, file: str, measurements_expressions: ExpressionsList = None,
                 uncertainties_expressions: ExpressionsList = None) -> None:
        self.file = file

        self.expr_meas = measurements_expressions or {}
        self.expr_uncert = uncertainties_expressions or {}

        self.data = Helper.fileToData(file)

        self.uncert = {x: {} for x in self.data}

        Helper.computeExpressions('data', self.expr_meas, self.data)
        Helper.defaultUncertainties(self.data, self.uncert)
        Helper.computeExpressions('uncert', self.expr_uncert, self.data, self.uncert)

    def traceMeasurementGraphs(self, ids: list[str] = None):
        """Traces Scope graphs of the current dataset."""
        if 'graph' not in self.data.getDataKeys():
            assert False, 'The current exprimentdoesn\'t have associated graph.'

        count = len(self.data) if ids is None else len(ids)
        gris_spec, is_fig_2d, col_count = Helper.initializeGraph(count)[:3]
        curves_ids = ids or list(self.data.keys())
        scopes_count = len(curves_ids)

        for i in range(count):
            spec = gris_spec[i // col_count, i % col_count] if is_fig_2d else gris_spec[i]

            if i < scopes_count:
                graph = self.data[curves_ids[i]]['graph']
                if isinstance(graph, Scope):
                    graph.traceGraph(spec, should_plot_scaled=False)
                    graph.ax_og.set_title(curves_ids[i])

        gris_spec.tight_layout(plt.gcf(), h_pad=0.3, w_pad=0.2)

        Helper.show(Helper.fileNameFromPath(self.file), 'measurements')

        # plt.show()

    def trace(self, curves: list[Curve], ignore: list[str] = None):
        """Traces `Curves` using the current dataset."""
        curves_count = len(curves)
        specs, is_fig_2d, col_count = Helper.initializeGraph(curves_count)[:3]

        for curve in curves:
            curve.populate(self.data, self.uncert, ignore)

        data_len = len(self.data.keys())

        for i in range(curves_count):
            spec = specs[i // col_count, i % col_count] if is_fig_2d else specs[i]

            if i < curves_count:
                ax = plt.subplot(spec)
                data = curves[i].getData()
                ax.scatter(data['x'], data['y'])

                ax.set_xlabel(data['x_label'])
                ax.set_ylabel(data['y_label'])

                valid_ux = len(data['ux']) == data_len
                valid_uy = len(data['uy']) == data_len

                for j, _id in enumerate(self.data.keys()):
                    x_pos: float = data['x'][j]
                    y_pos: float = data['y'][j]
                    ax.annotate(_id, (x_pos, y_pos),
                                textcoords='offset points', xytext=(0, 5), ha='center')
                    if data['unc'] and (valid_ux or valid_uy):
                        ax.errorbar(
                            data['x'][j],
                            data['y'][j],
                            data['uy'][j] if valid_uy else 0,
                            data['ux'][j] if valid_ux else 0,
                            ecolor='firebrick')
            else:
                ax.axis('off')

        Helper.show(Helper.fileNameFromPath(self.file), [(curve.x_label, curve.y_label) for curve in curves])

        # plt.show()

    def correctMeasurements(
            self,
            curve: Curve,
            corrected_values: list[str] = None,
            expression: Expression = None):
        """Starts a correction session for given values."""
        corrector = Corrector(self.file,
                              self.data,
                              expressions=self.expr_meas,
                              corrected_values=corrected_values or ['threshold', 'offset'])
        corrector.start(curve, expression)
