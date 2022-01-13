from typing import Union
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from model.experience_data import ExperienceData
from model.scope import Scope, Data
from model.value_slider import ValueSlider
from model.expression import Expression, ExpressionsList
from model.curve import Curve
from model.helper import Helper


class Corrector:
    """A wrapper object for handling correction on given dataset values."""
    curr_index: int
    prev_index: int
    correction_index: list[int]
    corrected_values: list[str]
    data: Data
    expressions: ExpressionsList
    ids: list[str]
    indexes: dict[str, int]
    columns: list[str]
    file: str

    def __init__(self,
                 path: str, data: Data,
                 expressions: ExpressionsList,
                 corrected_values: list[str] = None,
                 title_line: int = 0) -> None:
        self.data = data
        self.file = path
        self.expressions = expressions

        self.ids = list(data.keys())
        self.indexes = {k: i for i, k in enumerate(data.keys())}
        self.correction_index = []
        self.curr_index = 0
        self.prev_index = None

        self.corrected_values = corrected_values or ['threshold', 'offset']

        with open(path) as file:
            self.columns = [x.strip().lower()
                            for x in file.readlines()[title_line].split(',')]

        needs_update = False
        for col in corrected_values:
            if col not in self.columns:
                self.columns.append(col)
                self.correction_index.append(len(self.columns) - 1)
                for key in self.data.keys():
                    if col == 'offset':
                        x_data = data[key]['graph'].x_data
                        self.data[key][col] = - sum(x_data) / len(x_data)
                    else:
                        self.data[key][col] = .0
                needs_update = True
            else:
                self.correction_index.append(self.columns.index(col))

        if needs_update:
            self.write_to_file()

    def write_to_file(self):
        """Writes current experiment settings to file."""
        # pylint: disable = unsubscriptable-object
        file_buffer: list[list[Union[str, float]]] = []
        columns = self.columns

        file_buffer.append(columns)

        for _id in self.data.keys():
            row_buffer = []
            for col in columns:
                if col == 'id':
                    row_buffer.append(_id)
                else:
                    row_buffer.append(self.data[_id][col])

            file_buffer.append(row_buffer)

        with open(self.file, 'w', newline='') as file:
            file.writelines('\n'.join([', '.join([str(value) for value in row_buffer]) for row_buffer in file_buffer]))

    # pylint: disable = unsubscriptable-object
    def format_expression(self, expression: Expression, data: ExperienceData):
        """Returns a formatted string of an expression evaluation."""
        res = expression(data[self.ids[self.curr_index]])
        return f'Expression: {res:.3e}'

    def start(self, curve: Curve, expression: Expression = None):
        """Begins the correction session."""

        # pylint: disable = invalid-name
        DEFAULT_COLOR = 'tab:blue'
        DEFAULT_SIZE = 20
        SELECTED_COLOR = 'tab:red'
        SELECTED_SIZE = 50

        graph_initialized: float = False
        curve_initialized = False

        data: Data = Helper.safe_dict_copy(self.data)

        fig = plt.figure(figsize=(10, 8))

        graph_preview, curve_preview = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

        # graph_preview = plt.subplot2grid(
        #     shape=(4, 4), loc=(0, 0), colspan=2, rowspan=4)
        # curve_preview = plt.subplot2grid(shape=(4, 4), loc=(0, 2), colspan=2, rowspan=3)

        colors = ([DEFAULT_COLOR] * len(self.ids))
        sizes = [DEFAULT_SIZE] * len(self.ids)

        # graph_y_min, graph_y_max = Helper.get_global_scope_min_max(data, 'y')
        # graph_x_min, graph_x_max = Helper.get_global_scope_min_max(data, 'x')

        def handle_slider_update(key: str, value: float):
            """Handles modifications on sliders."""
            curr = self.ids[self.curr_index]
            data[curr][key] = value
            # print(data[curr][key], self.data[curr][key])
            Helper.compute_expressions('data', self.expressions, data, element_id=curr)

            updateGraph(curr)
            updateCurve(curr)

        def handle_toggle(key: str, value: bool):
            """Handles changes to the slider toggles."""
            curr = self.ids[self.curr_index]
            ...

        sliders: dict[str, ValueSlider] = {}
        for i, key in enumerate(self.corrected_values):
            # sliders[key] = ValueSlider(key,
            #                            plt.axes([0.600, 0.1 + i * 0.05, 0.250, 0.050]),
            #                            plt.axes([0.900, 0.1 + i * 0.05, 0.050, 0.050]),
            #                            handle_slider_update,
            #                            handle_toggle,
            #                            min_max=1)
            sliders[key] = Slider(ax=plt.axes([0.600, 0.1 + i * 0.05, 0.250, 0.050]), label=key, valmin=-1, valmax=1)

        def updateCurrent(incr: int) -> tuple[str, str]:
            """Updates the current indexes and related stuff."""
            self.prev_index = self.curr_index
            self.curr_index += incr
            self.curr_index %= len(self.ids)
            prev = self.ids[self.prev_index]
            curr = self.ids[self.curr_index]

            colors[self.prev_index] = DEFAULT_COLOR
            sizes[self.prev_index] = DEFAULT_SIZE
            colors[self.curr_index] = SELECTED_COLOR
            sizes[self.curr_index] = SELECTED_SIZE

            return (curr, prev)

        graph_line = None
        thtop = None
        thbot = None
        graph_title = None
        graph_legend = None

        def updateGraph(graph_id: str):
            """Redraws the leftmost graph."""
            nonlocal graph_initialized
            nonlocal graph_line, thtop, thbot, graph_title, graph_legend
            scope: Scope = Helper.safe_dict_copy(data[graph_id]['graph'])

            # graph_axes = list(graph.keys())

            if not graph_initialized:
                #     graph_preview.set_xlim([graph_x_min, graph_x_max])
                #     graph_preview.set_ylim([graph_y_min - 3, graph_y_max + 3])

                scope.trace_graph(graph_preview, should_plot_original=True, should_plot_scaled=True, on_same_graph=False)

                graph_title = scope.get_axis('wrapper').text(
                    0.29, 0.9, '', fontsize=14, transform=plt.gcf().transFigure)
                if 'threshold' in self.corrected_values:
                    thtop = scope.get_axis('og').axhline(color='red', linestyle='-')
                    thbot = scope.get_axis('og').axhline(color='red', linestyle='-')

                if expression:
                    graph_legend = scope.get_axis('wrapper').text(0.310, 0.035,
                                                                  self.format_expression(expression, data),
                                                                  ha='center',
                                                                  va='center',
                                                                  transform=plt.gcf().transFigure)

                graph_initialized = True

            if 'threshold' in self.corrected_values and 'threshold' in data[graph_id]:
                offset = data[graph_id]['offset'] if 'offset' in data[graph_id] else 0
                thtop.set_ydata(-offset + data[graph_id]['threshold'])
                thbot.set_ydata(-offset - data[graph_id]['threshold'])

            scope.update_graph(lambda x, threshold, offset: (x + offset if abs(x + offset) >
                                                             threshold else 0), [data[graph_id]['threshold'], data[graph_id]['offset']])

            if graph_legend:
                graph_legend.set_text(self.format_expression(expression, data))

        curve_line = None
        annotations = {}

        def updateCurve(id: str):
            """Retraces the rightmost graph."""
            nonlocal curve_initialized
            nonlocal curve_line, annotations

            # if not curve.hasData:
            #     curve_plot = curve.populate(data, {})
            # else:
            #     curve_plot = curve.update(id, data[id], {})

            if not curve_initialized:
                #     curve_preview.set_xlabel(curve_plot['x_label'])
                #     curve_preview.set_ylabel(curve_plot['y_label'])
                #     curve_line = curve_preview.scatter(
                #         curve_plot['x'], curve_plot['y'], c=colors, s=sizes)
                #     curve_preview.autoscale_view()
                #     for j, id in enumerate(data.keys()):
                #         # annotations[id] = curve_preview.annotate(id, (curve_plot['x'][j], curve_plot['y'][j]), textcoords='offset points', xytext=(0, 5), ha='center')
                #         annotations[id] = curve_preview.annotate(
                #             id, (curve_plot['x'][j], curve_plot['y'][j]))
                curve.trace_curve(curve_preview, data)

                curve.get_axis()

                curve_initialized = True

            # curve_line.set_offsets(list(zip(curve_plot['x'], curve_plot['y'])))
            # curve_line.set_color(colors)
            # curve_line.set_sizes(sizes)
            # curve_preview.relim()

            # annotations[id].set_y(curve_plot['y'][self.indexes[id]])

        def updateSliders(key: str):
            """Redraws sliders with current values."""
            for item, slider in sliders.items():
                # print(data[key][item])
                # slider.set_slider_value(data[key][item])
                slider.set_val(data[key][item])

        def updatePage(increment: int):
            """Redraws everything on screen and handles page change."""
            curr, _ = updateCurrent(increment)

            
            updateGraph(curr)
            updateCurve(curr)
            updateSliders(curr)

            fig.canvas.draw_idle()

        def resetCurrent():
            """Resets values of the current entry to previous ones."""
            key = self.ids[self.curr_index]
            data[key] = self.data[key].copy()
            updatePage(0)

        def apply():
            """Saves the changes and updates the file."""
            self.data = Helper.safe_dict_copy(data)
            self.write_to_file()

        axnext = plt.axes([0.445, 0.01, 0.05, 0.050])
        axprev = plt.axes([0.125, 0.01, 0.05, 0.050])
        axapply = plt.axes([0.850, 0.01, 0.05, 0.050])
        axdefault = plt.axes([0.775, 0.01, 0.05, 0.050])
        bnext = Button(axnext, '>>')
        bnext.on_clicked(lambda _: updatePage(1))
        bprev = Button(axprev, '<<')
        bprev.on_clicked(lambda _: updatePage(-1))
        bapply = Button(axapply, 'Apply')
        bapply.on_clicked(lambda _: apply())
        bdefault = Button(axdefault, 'Default')
        bdefault.on_clicked(lambda _: resetCurrent())

        updatePage(0)

        plt.show()
