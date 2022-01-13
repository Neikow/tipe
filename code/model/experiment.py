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

        self.data = Helper.file_to_data(file)

        self.uncert = {x: {} for x in self.data}

        Helper.compute_expressions('data', self.expr_meas, self.data)
        Helper.default_uncertainties(self.data, self.uncert)
        Helper.compute_expressions('uncert', self.expr_uncert, self.data, self.uncert)

    def trace_measurement_graphs(self, custom_x_axis: str = None, custom_y_axis: str = None, ids: list[str] = None):
        """Traces Scope graphs of the current dataset."""
        if 'graph' not in self.data.get_data_keys():
            assert False, 'The current exprimentdoesn\'t have associated graph.'

        count = len(self.data) if ids is None else len(ids)
        axs, is_fig_2d, col_count = Helper.initialize_graph(count)[:3]

        curves_ids = ids or list(self.data.keys())
        scopes_count = len(curves_ids)

        for i in range(count):
            axis = axs[i // col_count][i % col_count] if is_fig_2d else axs[i]

            if i < scopes_count:
                graph = self.data[curves_ids[i]]['graph']
                if isinstance(graph, dict):
                    x_label, y_label = graph.keys()
                    axis.plot(graph[custom_x_axis or x_label],
                              graph[custom_y_axis or y_label])
                    axis.set_title(curves_ids[i])
            else:
                axis.axis('off')

        plt.show()

    def trace(self, curves: list[Curve]):
        """Traces `Curves` using the current dataset."""
        curves_count = len(curves)
        axs, is_fig_2d, col_count = Helper.initialize_graph(curves_count)[:3]

        for curve in curves:
            curve.populate(self.data, self.uncert)

        data_len = len(self.data.keys())

        for i in range(curves_count):
            axis = axs[i // col_count][i % col_count] if is_fig_2d else axs[i] if curves_count > 1 else axs

            if i < curves_count:
                data = curves[i].plot()
                axis.scatter(data['x'], data['y'])

                axis.set_xlabel(data['x_label'])
                axis.set_ylabel(data['y_label'])

                valid_ux = len(data['ux']) == data_len
                valid_uy = len(data['uy']) == data_len

                for j, _id in enumerate(self.data.keys()):
                    x_pos: float = data['x'][j]
                    y_pos: float = data['y'][j]
                    axis.annotate(_id, (x_pos, y_pos),
                                  textcoords='offset points', xytext=(0, 5), ha='center')
                    if data['unc'] and (valid_ux or valid_uy):
                        axis.errorbar(
                            data['x'][j],
                            data['y'][j],
                            data['uy'][j] if valid_uy else 0,
                            data['ux'][j] if valid_ux else 0,
                            ecolor='firebrick')
            else:
                axis.axis('off')

        plt.show()

    def correct_measurements(
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
