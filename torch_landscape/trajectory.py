from typing import List, Tuple, TypeVar

from numpy import hstack
from numpy.linalg import lstsq
from torch import Tensor, no_grad
from torch.nn.utils import parameters_to_vector

from torch_landscape.visualize_options import VisualizationOptions

_TrajectoryData = TypeVar("_TrajectoryData", bound="TrajectoryData")


class TrajectoryData:
    """
    Class which holds the data of a trajectory, which are the points of the form (x,y,z).
    """

    def __init__(self, points: List[Tuple[float, float, float]] = ()):
        """
        Initializes the TrajectoryData class.
        :param points: The points of the trajectory.
        """
        self.points = points

    def filter_points_by_range(self, options: VisualizationOptions) -> _TrajectoryData:
        """
        Filters the points such that they fit within the range of the specified visualization options.
        :param options: The options to use.
        :return: An instance which only holds the points which fit within the range of the specified options.
        """
        x_min, y_min, x_max, y_max = options.min_x_value, options.min_y_value, options.max_x_value, options.max_y_value
        filtered_points = [(x, y, z) for x, y, z in self.points if x_min <= x <= x_max and y_min <= y <= y_max]
        return TrajectoryData(filtered_points)

    def set_range_to_fit_trajectory(self, options: VisualizationOptions, extension_factor: float = 0.1):
        """
        Sets the ranges x_min,x_max,y_min,y_max of the provided options
        such that the points fit within this range. Extends the range a bit using the provided factor.
        :param options: The visualization options to modify.
        :param extension_factor: The factor to use to extend the minimum/maximum values (between 0 and 1).
        """
        x_max, x_min, y_max, y_min = TrajectoryData.calculate_ranges(self.points, extension_factor=extension_factor)
        options.max_x_value, options.min_x_value, options.max_y_value, options.min_y_value = x_max, x_min, y_max, y_min

    @staticmethod
    def calculate_ranges(
        intermediate_results: List[Tuple[float, float, float]], extension_factor: float = 0.1
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the ranges x_min,x_max,y_min,y_max such that the provided intermediate results fit within this range.
        Extends the range a bit using the provided factor.
        :param intermediate_results: The intermediate results of the optimizer in the visualization space.
        :param extension_factor: The factor to extend the range.
        :return: The visualization range x_max, x_min, y_max, y_min
        """
        x_values = [x for x, _, _ in intermediate_results]
        y_values = [y for _, y, _ in intermediate_results]
        # Use minimum values as boundaries for the range of the loss landscape which will be visualized.
        x_min = min(x_values)
        x_max = max(x_values)
        x_span = x_max - x_min
        y_min = min(y_values)
        y_max = max(y_values)
        y_span = y_max - y_min
        # increase boundary a little bit.
        x_min -= x_span * extension_factor
        x_max += x_span * extension_factor
        y_min -= y_span * extension_factor
        y_max += y_span * extension_factor
        return x_max, x_min, y_max, y_min


class TrajectoryCalculator:
    """
    Calculates the trajectory points (x,y,z) of an optimizer during training by projecting the intermediate
    parameters onto the visualization space using linear least squares.
    """

    def __init__(self, optimized_parameters: List[Tensor], directions: Tuple[List[Tensor], List[Tensor]]):
        """
        Initializes the Trajectory calculator.
        :param optimized_parameters: The optimized parameters.
        :param directions: The directions (b1, b2) to use.
        """
        self._optimized_parameters = optimized_parameters
        self._directions = directions
        b1_params, b2_params = self._directions
        self._b1 = parameters_to_vector(b1_params)
        self._b2 = parameters_to_vector(b2_params)
        self._zero_point = parameters_to_vector(self._optimized_parameters)

    def project_with_loss(self, intermediate_parameters_with_loss: List[Tuple[List[Tensor], float]]) -> TrajectoryData:
        """
        Projects the intermediate parameters to the visualization space and uses the provided loss value.
        :param intermediate_parameters_with_loss: A list of intermediate parameters as tuple: the list of parameters and
        the loss.
        :return: The list of the coordinates for the projected intermediate parameters.
        """
        intermediate_results_coordinates = []
        with no_grad():
            for parameters, loss in intermediate_parameters_with_loss:
                parameters_vector = parameters_to_vector(parameters)
                x, y = TrajectoryCalculator.to_visualization_space(
                    parameters_vector, self._zero_point, self._b1, self._b2
                )
                intermediate_results_coordinates.append((x, y, loss))
        return TrajectoryData(intermediate_results_coordinates)

    def project_disregard_z(self, intermediate_parameters: List[List[Tensor]], z_value: float = -1) -> TrajectoryData:
        """
        Projects the intermediate parameters to the visualization space and disregards the calculation of the z-value.
        This is useful for 2d visualizations which do not show the z value of the projected points.
        :param intermediate_parameters: A list of intermediate parameters. the loss.
        :param z_value: The z value to use for the projected coordinates.
        :return: The list of coordinates for the projected intermediate parameters.
        """
        intermediate_results_coordinates = []
        with no_grad():
            for parameters in intermediate_parameters:
                parameters_vector = parameters_to_vector(parameters)
                x, y = TrajectoryCalculator.to_visualization_space(
                    parameters_vector, self._zero_point, self._b1, self._b2
                )
                intermediate_results_coordinates.append((x, y, z_value))
        return TrajectoryData(intermediate_results_coordinates)

    @staticmethod
    def to_visualization_space(input_vector: Tensor, zero_point: Tensor, b1: Tensor, b2: Tensor) -> Tuple[float, float]:
        """
        Calculates the best representation for the input_vector in the visualization space using the
        least squares method:
        input_vector = zero_point + b1 * x + b2*y + r
        such that r is minimal.
        :param input_vector: The vector from the parameter space.
        :param zero_point: The zero point of the visualization space in coordinates of parameter space.
        :param b1: First basis vector in visualization space in coordinates of parameter space.
        :param b2: Second basis vector in visualization space in coordinates of parameter space.
        :return: Tuple [x,y].
        """
        # vector = b1*x + b2*y = input_vector - zero_point
        # zero_point + b1*x + b2*y = input_vector.
        vector = input_vector - zero_point.to(input_vector.device)
        # solve equation (input_vector - zero_point) = b1 * x + b2 * y.
        A = hstack((b1.cpu().detach().numpy().reshape(-1, 1), b2.cpu().detach().numpy().reshape(-1, 1)))
        coordinates, _, _, _ = lstsq(A, vector.cpu().detach().numpy(), rcond=None)
        return coordinates

    @staticmethod
    def calculate_reconstruction_error(
        b1: List[Tensor], b2: List[Tensor], zero_point: List[Tensor], input_params: List[Tensor]
    ) -> float:
        """
        Calculate the reconstruction error r of a list of parameters:
        input_params = zero_point + x*b1 + y*b2 + r.
        :param b1: First basis vector in visualization space in coordinates of parameter space.
        :param b2: Second basis vector in visualization space in coordinates of parameter space.
        :param zero_point: The zero point of the visualization space in coordinates of parameter space.
        :param input_params: The input parameters for which the reconstruction error is calculated.
        :return: the 1-norm of r.
        """
        zero_point_vec = parameters_to_vector(zero_point)
        b1_params = parameters_to_vector(b1)
        b2_params = parameters_to_vector(b2)
        input_vec = parameters_to_vector(input_params)

        x, y = TrajectoryCalculator.to_visualization_space(input_vec, zero_point_vec, b1_params, b2_params)

        reconstructed = zero_point_vec + x * b1_params + y * b2_params
        error = (reconstructed - input_vec).norm(p=1).item()
        return error

    @staticmethod
    def reconstruction_error_mean(
        b1: List[Tensor], b2: List[Tensor], zero_point: List[Tensor], samples: List[List[Tensor]]
    ) -> float:
        """
        Calculate the reconstruction error r of a set of parameter lists. For each parameter:
        parameter = zero_point + x*b1 + y*b2 + r.
        The 1-norm of r is calculated and the sum is returned.
        :param b1: First basis vector in visualization space in coordinates of parameter space.
        :param b2: Second basis vector in visualization space in coordinates of parameter space.
        :param zero_point: The zero point of the visualization space in coordinates of parameter space.
        :param samples: The list of parameters for which the reconstruction error is calculated.
        :return: The sum of the 1-norms of r.
        """
        errors_sum = sum(
            TrajectoryCalculator.calculate_reconstruction_error(b1, b2, zero_point, sample) for sample in samples
        )
        return errors_sum / len(samples)
