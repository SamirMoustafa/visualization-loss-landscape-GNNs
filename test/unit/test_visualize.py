from unittest import TestCase, main
from numpy import array
from torch import tensor

from torch_landscape.landscape import LossSurfaceData
from torch_landscape.trajectory import TrajectoryData
from torch_landscape.visualize import (Plotly2dVisualization, Plotly3dVisualization, VisualizationData,
                                       VisualizationOptions)


class VisualizeTest(TestCase):
    def setUp(self) -> None:
        x_coordinates = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_coordinates = array([-9, -8, -7, -6, -5, -4, -3, -2, 1])
        z_coordinates = array([[x + y for x in x_coordinates] for y in y_coordinates])
        loss_surface_data = LossSurfaceData(x_coordinates, y_coordinates, z_coordinates)
        trajectory = TrajectoryData([(12, 13, 14), (11, 12, 13)])
        self.data = VisualizationData(loss_surface_data, trajectory)
        b1 = [tensor([1.0, 0.0])]
        b2 = [tensor([0.0, 1.0])]
        self.directions = [b1, b2]
        self.intermediate_results = [(-0.5, -0.5, 0.25 + 0.1), (-0.25, 0, -0.25 + 0.1), (0.1, 0.1, 0.01 + 0.1)]

    def test_create_2d_plot(self):
        """
        Smoke test for 2d plot creation.
        """
        options = VisualizationOptions()
        Plotly2dVisualization(options).plot(self.data, "visualize2d_test", "My test plot")

    def test_create_3d_plot(self):
        """
        Smoke test for 3d plot creation.
        """
        options = VisualizationOptions()
        Plotly3dVisualization(options).plot(self.data, "visualize3d_test", "My test plot")


if __name__ == "__main__":
    main()
