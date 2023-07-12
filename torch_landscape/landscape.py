from typing import Iterable, List, Tuple

import numpy as np
from numpy import ndarray
from torch import Tensor


class LossSurfaceData:
    """
    Class which stores the loss surface data.
    """

    # The x values of the grid.
    x_coordinates: ndarray
    # The y values of the grid.
    y_coordinates: ndarray
    # A matrix which holds the z-values for each grid point.
    z_coordinates: ndarray

    def __init__(self, x_coordinates: ndarray, y_coordinates: ndarray, z_coordinates: ndarray):
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinates = z_coordinates


def setup_surface_data_linear(
    min_x_value: float,
    max_x_value: float,
    num_data_point: int,
    min_y_value: float,
    max_y_value: float,
) -> LossSurfaceData:
    """
    Creates a dictionary containing the x-coordinates, the y-coordinates and the z-coordinates, which can be used to
    plot the loss surface.
    The z-coordinates are set to -1 by default.
    The values for the x and y coordinates are linearly distributed between min_data_point and max_data_point.

    The x/y values contain the values of the grid.
    The z_coordinate_name contains a matrix which holds the z-values for each grid point.

    :param min_x_value: Minimum value for the x coordinate.
    :param max_x_value: Maximum value for the x coordinate.
    :param min_y_value: Minimum value for the y coordinate.
    :param max_y_value: Maximum value for the y coordinate.
    :param num_data_point: The count of data points - 1.
    :return: An instance of LossSurfaceData.

    """
    x_min, x_max, x_num = min_x_value, max_x_value, num_data_point
    y_min, y_max, y_num = min_y_value, max_y_value, num_data_point

    x_coordinates = np.linspace(x_min, x_max, x_num)
    y_coordinates = np.linspace(y_min, y_max, y_num)

    shape = (len(x_coordinates), len(y_coordinates))
    default_landscape_values = -np.ones(shape=shape)

    surface_dictionary = LossSurfaceData(x_coordinates, y_coordinates, default_landscape_values)
    return surface_dictionary


def overwrite_parameters(
    model_parameters: Iterable[Tensor],
    optimized_parameters: Iterable[Tensor],
    directions: Tuple[List[Tensor], List[Tensor]],
    step: List[float],
):
    """
    Changes the weights of the model. The new weights are the
    initial weights plus
    directions[0]*step[0] + directions[1]*step[1].

    :param model_parameters: The parameters to change.
    :param optimized_parameters: The initial (optimal) parameters (Theta*).
    :param directions: A list containing two vectors in the parameter/weights space.
    :param step: The coordinates with respect to the directions.
    """
    b1 = directions[0]
    b2 = directions[1]
    x, y = step[0], step[1]

    for param, optimized_param, b1_i, b2_i in zip(model_parameters, optimized_parameters, b1, b2):
        param.data = optimized_param + x * b1_i + y * b2_i
