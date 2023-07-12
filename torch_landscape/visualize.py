from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go

from torch_landscape.landscape import LossSurfaceData
from torch_landscape.names import component_one_name, component_two_name, surface_landscape_name
from torch_landscape.trajectory import TrajectoryData
from torch_landscape.visualize_options import VisualizationOptions

DEFAULT_PLOTLY_TEMPLATE = "plotly_white"


class VisualizationData:
    def __init__(self, loss_surface_data: LossSurfaceData, trajectory_data: Optional[TrajectoryData] = None):
        self.loss_surface_data = loss_surface_data
        self.trajectory_data = trajectory_data


class VisualizationBase(ABC):
    """
    Base class for visualizations, which can plot a loss surface and trajectories.
    """

    def __init__(self, options: VisualizationOptions):
        """
        Initializes the visualization class.
        :param options: The options to create any plots.
        """
        self.options = options

    @abstractmethod
    def plot(self, loss_surface: VisualizationData):
        raise NotImplementedError()


class PlotlyVisualization(VisualizationBase, ABC):
    def __init__(self, options: VisualizationOptions, plotly_template: str = DEFAULT_PLOTLY_TEMPLATE):
        """
        Initializes the Plotly 3d visualization class.
        :param options: The visualization options to use.
        :param plotly_template: The template to use for plotly.
        """
        super().__init__(options)
        self.plotly_template = plotly_template

    def get_z_axis_type(self):
        """
        Determines the z axis type depending on the options.
        :return: A string describing the axis type.
        """
        z_axis_type = "linear"
        if self.options.use_log_z:
            z_axis_type = "log"
        return z_axis_type

    def set_labels(self, fig: go.Figure, title: Optional[str] = None):
        """
        Sets the labels of the x/y/z axis, the template, the type of the z axis and the title of the plot.
        :param fig: The figure to modify.
        :param title: (Optional) The title to set.
        """
        fig.update_layout(
            template=self.plotly_template,
            title=title,
            scene={
                "xaxis": {"title": "", "showticklabels": False},
                "yaxis": {"title": "", "showticklabels": False},
                "zaxis": {"title": "", "showticklabels": False, "type": self.get_z_axis_type(),
                          },
            },
            showlegend=False,
        )

    @staticmethod
    def save_file(fig: go.Figure, file_path: Union[str, Path], file_extension: str = "html"):
        """
        Saves the figure to a file.
        :param fig: The figure to save.
        :param file_path: The path to the file.
        :param file_extension: The file extension to use.
        """
        if file_extension == "html":
            fig.write_html(file_path + ".html")
        else:
            fig.write_image(file_path + "." + file_extension)


class Plotly3dVisualization(PlotlyVisualization):
    """
    Visualization class to create 3d plots using the plotly library.
    """

    def __init__(
        self,
        options: VisualizationOptions,
        z_axis_lines: int = 50,
        opacity: float = 0.8,
        plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
        show_color_scale: bool = False,
    ):
        """
        Initializes the Plotly 3d visualization class.
        :param options: The visualization options to use.
        :param z_axis_lines: The number of contour lines to use on the z-axis. Default is 50.
        :param opacity: The opacity of the surface. Default is 0.8.
        :param plotly_template: The template to use for plotly.
        :param show_color_scale: Set to true to show a scale for the color axis.
        """
        super().__init__(options, plotly_template)
        self.z_axis_lines = z_axis_lines
        self.opacity = opacity
        self.plotly_template = plotly_template
        self.show_color_scale = show_color_scale

    def plot(
        self,
        data: VisualizationData,
        file_path: Optional[Union[Path, str]] = None,
        title: Optional[str] = None,
        file_extension: str = "html",
    ):
        """
        Creates a 3d plot of the specified data and stores it to the specified path.
        :param data: The data to plot
        :param file_path: The path to the file, where the plot is saved.
        :param title: The title of the plot.
        :param file_extension: The file extension of the exported file (html, png, pdf, ...).
        """
        fig = self.create_figure(data, title)
        self.save_file(fig, file_path, file_extension)

    def create_figure(self, data: VisualizationData, title: Optional[str]) -> go.Figure:
        """
        Creates a plotly 3d figure with the specified data.
        :param data: The data to plot.
        :param title: The title of the plot.
        :return: A figure with the specified data.
        """
        x = data.loss_surface_data.x_coordinates
        y = data.loss_surface_data.y_coordinates
        z = data.loss_surface_data.z_coordinates
        # limit the z values to the 20th percentile to avoid outliers
        z_threshold = np.percentile(z, 20)
        z = np.where(z > z_threshold, z_threshold, z)
        # create a grid for the x/y coordinates
        x_grid, y_grid = np.meshgrid(x, y)
        lighting_effects = dict(diffuse=0.9)
        fig = go.Figure(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z,
                opacity=self.opacity,
                contours={
                    "z": {
                        "show": True,
                        "start": np.nanmin(z),
                        "end": np.nanmax(z),
                        "size": (np.nanmax(z) - np.nanmin(z)) / self.z_axis_lines,
                        "width": 5,
                        "color": "black",
                    },
                },
                lighting=lighting_effects,
                showscale=self.show_color_scale,
                colorscale="viridis",
            ),
        )
        self.set_labels(fig, title)
        fig.update_layout(
            title_y=0.87,
            title_yanchor="bottom",
            margin=dict(b=35, t=40, l=0, r=0),
            scene={
                "xaxis": {"titlefont": {"size": 8}},
                "yaxis": {"titlefont": {"size": 8}},
            },
        )
        if data.trajectory_data is not None:
            fig.add_scatter3d(
                x=[x for x, _, _ in data.trajectory_data.points],
                y=[y for _, y, _ in data.trajectory_data.points],
                z=[z for _, _, z in data.trajectory_data.points],
                mode="markers+lines",
                marker=dict(
                    size=5, colorscale="viridis", color=[float(i) for i in range(len(data.trajectory_data.points))]
                ),
            )

        fig.update_xaxes(automargin=True)
        fig.update_scenes(aspectmode="cube")
        return fig


class Plotly2dVisualization(PlotlyVisualization):
    def __init__(
        self, options: VisualizationOptions, plotly_template: str = DEFAULT_PLOTLY_TEMPLATE, mark_zero: bool = True
    ):
        """
        Initializes the Plotly 2d visualization class.
        :param options: The options to use for creating the plots.
        :param plotly_template: The plotly template to use.
        :param mark_zero: Set to true to set a marker at the point corresponding to (0,0).
        """
        super().__init__(options, plotly_template)
        self.mark_zero = mark_zero

    def plot(
        self,
        data: VisualizationData,
        file_path: Optional[Union[Path, str]] = None,
        title: Optional[str] = None,
        file_extension: str = "html",
    ):
        """
        Creates a 2d plot of the specified data.
        :param data: The data to plot.
        :param file_path: The path to the file, where the plot is saved.
        :param title: (Optional) The title of the plot.
        :param file_extension: The file extension of the exported file (html, pdf, png, ...).
        """
        fig = self.create_figure(data, title)
        self.save_file(fig, file_path, file_extension)

    def create_figure(self, data: VisualizationData, title: Optional[str]) -> go.Figure:
        """
        Creates a 2d figure with the specified data.
        :param data: The data to plot.
        :param title: The title of the plot.
        :return: A plot with the specified data.
        """
        x = np.array(data.loss_surface_data.x_coordinates)
        y = np.array(data.loss_surface_data.y_coordinates)
        z = np.array(data.loss_surface_data.z_coordinates)
        fig = go.Figure(data=go.Contour(x=x, y=y, z=z, contours=dict(), colorscale="viridis", showscale=False))
        if self.mark_zero:
            fig.add_scatter(x=[0], y=[0], mode="markers", name="Zero", marker=dict(size=10, symbol="x", color="red"))
        self.set_labels(fig, title)
        if data.trajectory_data is not None:
            fig.add_scatter(
                x=[x for x, _, _ in data.trajectory_data.points],
                y=[y for _, y, _ in data.trajectory_data.points],
                mode="markers+lines",
                marker=dict(
                    size=5, colorscale="viridis", color=[float(i) for i in range(len(data.trajectory_data.points))]
                ),
            )
        return fig
