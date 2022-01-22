from typing import Literal, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import SubplotSpec
import numpy as np
from model.types import ScaleFunction
from scipy.interpolate import make_interp_spline

ScopeData = dict[str, list[float]]


class Scope:
    """A wrapper class for holding scope data."""
    # Labels
    x_label: str
    y_label: str

    # Data
    x_data: list[float]
    y_data: list[float]

    # Plot Axes
    ax_wrapper: plt.Axes    # Wrapper axis
    ax_og: plt.Axes         # Original axis
    ax_sc: plt.Axes         # Scaled axis

    # Plotted Lines
    ln_og: Line2D          # Original plot
    ln_sc: Line2D          # Scaled plot

    # Parameters
    on_same_graph: bool
    plot_original: bool
    plot_scaled: bool

    def __init__(self, scope_data: ScopeData) -> None:
        keys = list(scope_data.keys())

        self.x_label = keys[0]
        self.x_data = scope_data[self.x_label]
        self.y_label = keys[1]
        self.y_data = scope_data[self.y_label]

        self.ax_og = None
        self.ax_sc = None

        self.ln_og = None
        self.ln_sc = None

    def setXdata(self, new_data: list[float]):
        """Sets a new value for the X axis."""
        assert len(new_data) == len(self.y_data)
        self.x_data = new_data

    def setYdata(self, new_data: list[float]):
        """Sets a new value for the Y axis."""
        assert len(new_data) == len(self.x_data)
        self.y_data = new_data

    # pylint: disable = unsubscriptable-object, inconsistent-return-statements
    def getAxis(self, axis_type: Literal['wrapper', 'og', 'sc']) -> plt.Axes:
        """Returns the given axis if it exists."""
        if axis_type == 'wrapper':
            return self.ax_wrapper
        if self.on_same_graph:
            return self.ax_og
        if axis_type == 'og':
            assert self.plot_original, 'Original graph is not drawn.'
            return self.ax_og
        if axis_type == 'sc':
            assert self.plot_scaled, 'Scaled graph is not drawn.'
            return self.ax_sc

    # pylint: disable = too-many-arguments
    def traceGraph(self, spec: SubplotSpec,
                    scaling_function: ScaleFunction = None,
                    scaling_arguments: list[float] = None,
                    on_same_graph: bool = False,
                    should_plot_original: bool = True,
                    should_plot_scaled: bool = True) -> list[Line2D]:
        """Draws the scope graph using the provided `spec`."""

        self.on_same_graph = on_same_graph
        self.plot_original = should_plot_original
        self.plot_scaled = should_plot_scaled

        def plotOriginal():
            self.ln_og, = self.getAxis('og').plot(self.x_data, self.y_data)

        def plotScaled():
            if scaling_function:
                y_data = [scaling_function(x, *(scaling_arguments or [])) for x in self.y_data]
            else:
                y_data = self.y_data
            self.ln_sc, = self.getAxis('sc').plot(self.x_data, y_data)

        def createAxes():
            self.ax_wrapper = plt.subplot(spec)
            if on_same_graph:
                self.ax_og = self.ax_wrapper
                self.ax_sc = None
            elif should_plot_original and plotScaled:
                inner = spec.subgridspec(2, 1, wspace=0.05, hspace=0.1)
                curr_fig = plt.gcf()
                self.ax_og = curr_fig.add_subplot(inner[0, 0])

                # pylint: disable = expression-not-assigned
                [t.set_color('white') for t in self.ax_wrapper.yaxis.get_ticklabels() + self.ax_wrapper.xaxis.get_ticklabels()]
                [self.ax_wrapper.tick_params(axis=_ax, colors='white') for _ax in ['x', 'y']]
                [spine.set_color('white') for spine in self.ax_wrapper.spines.values()]

                self.ax_wrapper.set_xlabel(self.x_label)
                self.ax_wrapper.set_ylabel(self.y_label)
                self.ax_sc = curr_fig.add_subplot(
                    inner[1, 0], sharex=self.ax_og)
            else:
                if should_plot_original:
                    self.ax_og = self.ax_wrapper
                elif should_plot_scaled:
                    self.ax_sc = self.ax_wrapper

        createAxes()

        if should_plot_original:
            plotOriginal()

        if should_plot_scaled:
            plotScaled()

        return [self.ln_og, self.ln_sc]

    def updateGraph(self, scaling_function: ScaleFunction = None,
                     scaling_arguments: list[float] = None):
        """Updates the scaled graph."""
        assert self.ln_sc, 'Nothing to update.'

        if scaling_function:
            y_data = [scaling_function(x, *(scaling_arguments or [])) for x in self.y_data]
        else:
            y_data = self.y_data

        self.ln_sc.set_ydata(y_data)

    def copy(self):
        """Returns a copy of the scope. Without the plot data."""
        return Scope(dict([(self.x_label, self.x_data), (self.y_label, self.y_data)]))

    def plot(self):
        """Plots the scope in a new window."""
        plt.plot(self.x_data, self.y_data)

        spl = make_interp_spline(self.x_data, self.y_data, k=3)
        xnew = np.linspace(min(self.x_data), max(self.x_data), 300)
        Y = spl(xnew)

        plt.plot(xnew, Y)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()


# pylint: disable = unsubscriptable-object
Data = dict[str, dict[str, Union[float, Scope]]]
