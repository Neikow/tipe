from typing import Union
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from model.value_slider import ValueSlider
from model.types import ExpressionsList
from model.expression import Expression
from model.curve import Curve
from model.helper import Helper

from pprint import pprint

id_fun = id

class Corrector:
    """A wrapper object for handling correction on given dataset values."""
    curr_index: int
    prev_index: int
    correction_index: list[int]
    corrected_values: list[str]
    data: dict[str, dict[str, Union[float, dict]]]
    expressions: ExpressionsList
    ids: list[str]
    indexes: dict[str, int]
    columns: list[str]
    file: str

    def __init__(self, file: str, data: dict[str, dict[str, Union[float, dict]]], expressions: ExpressionsList, corrected_values: list[str] = ['threshold', 'offset'], title_line: int = 0) -> None:
        self.data = data
        self.file = file
        self.expressions = expressions

        self.ids = list(data.keys())
        self.indexes = {k: i for i, k in enumerate(data.keys())}
        self.correction_index = []
        self.curr_index = 0
        self.prev_index = None

        self.corrected_values = corrected_values

        with open(file) as f:
            self.columns = [x.strip().lower() for x in f.readlines()[title_line].split(',')]

        needs_update = False
        for col in corrected_values:
            if not col in self.columns:
                self.columns.append(col)
                self.correction_index.append(len(self.columns) - 1)
                for key in self.data.keys():
                    if col == 'offset':
                        X = data[key]['graph'][list(data[key]['graph'].keys())[1]]
                        self.data[key][col] = - sum(X) / len(X)
                    else:
                        self.data[key][col] = .0
                needs_update = True
            else:
                self.correction_index.append(self.columns.index(col))

        if needs_update: self.writeToFile()
        

    def writeToFile(self):
        """Writes current experiment settings to file."""
        file_buffer: list[list[Union[str,float]]] = []
        columns = self.columns

        file_buffer.append(columns)

        for id in self.data.keys():
            row_buffer = []
            for col in columns:
                if col == 'id':
                    row_buffer.append(id)
                else:
                    row_buffer.append(self.data[id][col])

            file_buffer.append(row_buffer)
        
        with open(self.file, 'w', newline='') as f:
            f.writelines('\n'.join([', '.join([str(value) for value in row_buffer]) for row_buffer in file_buffer]))

    def formatExpression(self, expression: Expression, data: dict[str, Union[float, dict]]):
        res = expression(data[self.ids[self.curr_index]])
        return f'Expression: {res:.3e}'

    def start(self, curve: Curve, expression: Expression = None):
        """Begins the correction session."""

        DEFAULT_COLOR = 'tab:blue'
        DEFAULT_SIZE = 20
        SELECTED_COLOR = 'tab:red'
        SELECTED_SIZE = 50

        graph_initialized: float = False
        curve_initialized = False

        data = Helper.safeDictCopy(self.data)

        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(8)

        graph_preview = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=2, rowspan=4)
        curve_preview = plt.subplot2grid(shape=(4, 4), loc=(0, 2), colspan=2, rowspan=3)

        colors = ([DEFAULT_COLOR] * len(self.ids))
        sizes = [DEFAULT_SIZE] * len(self.ids)

        graph_y_min, graph_y_max = Helper.getGlobalGraphMinMax(data, 'y')
        graph_x_min, graph_x_max = Helper.getGlobalGraphMinMax(data, 'x')

        def handleSliderUpdate(key: str, value: float):
            """Handles modifications on sliders."""
            curr = self.ids[self.curr_index]
            data[curr][key] = value
            # print(data[curr][key], self.data[curr][key])
            Helper.computeExpressions('data', self.expressions, data, id=curr)

            updateGraph(curr)
            updateCurve(curr)

        def handleToggle(key: str, value: bool):
            """Handles changes to the slider toggles."""
            curr = self.ids[self.curr_index]


        sliders: dict[str, ValueSlider] = {}
        for i, val in enumerate(self.corrected_values):
            slider = ValueSlider(val, plt.axes([0.600, 0.1 + i * 0.05, 0.250, 0.050]), plt.axes([0.900, 0.1 + i * 0.05, 0.050, 0.050]), handleSliderUpdate, handleToggle, min_max=1)
            sliders[val] = slider

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
        def updateGraph(id: str):
            """Redraws the leftmost graph."""
            nonlocal graph_initialized
            nonlocal graph_line, thtop, thbot, graph_title, graph_legend
            graph: dict[str, list[float]] = Helper.safeDictCopy(data[id]['graph'])
            graph_axes = list(graph.keys())
            
            if not graph_initialized:
                graph_preview.set_xlim([graph_x_min, graph_x_max])
                graph_preview.set_ylim([graph_y_min - 3, graph_y_max + 3])
                graph_line, = graph_preview.plot([], [])
                graph_title = graph_preview.text(0.29, 0.9, '', fontsize=14, transform=plt.gcf().transFigure)
                if 'threshold' in self.corrected_values:
                    thtop = graph_preview.axhline(color='red', linestyle='-')
                    thbot = graph_preview.axhline(color='red', linestyle='-')

                if expression:
                    graph_legend = graph_preview.text(0.310, 0.035,
                                                    self.formatExpression(expression, data),
                                                    ha='center', 
                                                    va='center', 
                                                    transform=plt.gcf().transFigure)

                graph_initialized = True                    

            Y = [np.array(Helper.safeListCopy(graph[graph_axes[1]]))]
            
            if 'offset' in self.corrected_values and 'offset' in data[id]:
                Y.append(Y[-1] + data[id]['offset'])

            if 'threshold' in self.corrected_values and 'threshold' in data[id]:
                thtop.set_ydata(data[id]['threshold'])
                thbot.set_ydata(-data[id]['threshold'])
                
                Y.append(Y[-1])
                for i in range(len(Y[-1])):
                    if abs(Y[-1][i]) < abs(data[id]['threshold']):
                        Y[-1][i] = 0
                

            X = graph[graph_axes[0]]
            Y = Y[-1]
            graph_line.set_data(X, Y)
            graph_preview.relim()
            graph_title.set_text(id)

            if graph_legend:
                graph_legend.set_text(self.formatExpression(expression, data))

        curve_line = None
        annotations = {} 
        def updateCurve(id: str):
            """Retraces the rightmost graph."""
            nonlocal curve_initialized
            nonlocal curve_line, annotations

            if not curve.hasData:
                curve_plot = curve.populate(data, {})
            else:
                curve_plot = curve.update(id, data[id], {})

            if not curve_initialized:
                curve_preview.set_xlabel(curve_plot['x_label'])
                curve_preview.set_ylabel(curve_plot['y_label'])
                curve_line = curve_preview.scatter(curve_plot['x'], curve_plot['y'], c=colors, s=sizes)
                curve_preview.autoscale_view()
                for j, id in enumerate(data.keys()):
                    # annotations[id] = curve_preview.annotate(id, (curve_plot['x'][j], curve_plot['y'][j]), textcoords='offset points', xytext=(0, 5), ha='center')
                    annotations[id] = curve_preview.annotate(id, (curve_plot['x'][j], curve_plot['y'][j]))
                curve_initialized = True

            curve_line.set_offsets(list(zip(curve_plot['x'], curve_plot['y'])))
            curve_line.set_color(colors)
            curve_line.set_sizes(sizes)
            curve_preview.relim()

            annotations[id].set_y(curve_plot['y'][self.indexes[id]])

        def updateSliders(id: str):
            """Redraws sliders with current values."""
            for key, slider in sliders.items():
                slider.setSliderValue(data[id][key])

        def updatePage(increment: int):
            """Redraws everything on screen and handles page change."""
            curr, _ = updateCurrent(increment)

            updateGraph(curr)
            updateCurve(curr)
            updateSliders(curr)
            
            fig.canvas.draw_idle()

        def resetCurrent():
            """Resets values of the current entry to previous ones."""
            id = self.ids[self.curr_index]
            data[id] = self.data[id].copy()
            updatePage(0)

        def apply():
            """Saves the changes and updates the file."""
            self.data = Helper.safeDictCopy(data)
            self.writeToFile()

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