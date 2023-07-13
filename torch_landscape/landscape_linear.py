from typing import Callable, Iterable, List, Optional, Tuple

from joblib import delayed
from torch import Tensor, no_grad
from torch.nn import Module

from torch_landscape.landscape import LossSurfaceData, overwrite_parameters, setup_surface_data_linear
from torch_landscape.parallel_progress_bar import ProgressParallel
from torch_landscape.utils import clone_parameters, reset_params
from torch_landscape.visualize_options import VisualizationOptions


def increase_parameters(parameters: Iterable[Tensor], addend: Iterable[Tensor]):
    """
    Increases the parameters by the given addend.
    Basically adds to each tensor in the variable "parameters" the corresponding tensor in "addends"
    Uses in-place operations.
    """
    for parameter, addend_part in zip(parameters, addend):
        parameter.add_(addend_part)


def _evaluate_row_parallel(
    evaluation_function: Callable[[Iterable[Tensor]], float],
    is_positive_direction: bool,
    model_parameters: List[Tensor],
    x_i: int,
    min_x_value: int,
    max_x_value: int,
    min_y_value: int,
    max_y_value: int,
    num_data_point: int,
    directions: Tuple[List[Tensor], List[Tensor]],
    optimized_parameters: Iterable[Tensor],
):
    """
    Evaluates a row of the loss surface (all points in the row have the same "x" value).

    :param evaluation_function: Function to evaluate the model.
    :param is_positive_direction: True if the direction is positive, False otherwise.
    :param model_parameters: The parameters of the model.
    :param x_i: The index of the row.
    :param num_data_point: The count of data points - 1.
    :param min_x_value: Minimum value for the x coordinate.
    :param max_x_value: Maximum value for the x coordinate.
    :param min_y_value: Minimum value for the y coordinate.
    :param max_y_value: Maximum value for the y coordinate.
    :param directions: The directions for the x and y axis.
    :param optimized_parameters: The optimized parameters.
    :return: A list of the evaluated values.
    """
    with no_grad():
        y_points = range(num_data_point)
        b1, b2 = directions
        x_step = (max_x_value - min_x_value) / (num_data_point - 1)
        y_step = (max_y_value - min_y_value) / (num_data_point - 1)
        b2_step = [b2_part * y_step for b2_part in b2]
        b2_step_negative = [-b2_part for b2_part in b2_step]
        b2_step_direction = b2_step if is_positive_direction else b2_step_negative

        # calculate x coordinate
        x = min_x_value + x_i * x_step

        results = {}
        for y_j in y_points if is_positive_direction else y_points[::-1]:
            # calculate y coordinate
            y = min_y_value + y_j * y_step

            overwrite_parameters(model_parameters, optimized_parameters, directions, [x, y])

            # save results to a dictionary
            results[(x_i, y_j)] = evaluation_function(model_parameters)

            if (y_j != num_data_point - 1 and is_positive_direction) or (y_j != 0 and not is_positive_direction):
                # Step in y-direction (b2).
                increase_parameters(model_parameters, b2_step_direction)
        return results


class LinearLandscapeCalculator:
    """
    Calculate the loss landscape L(T + x*b1 + y*b2)
    for the optimized parameters T with directions b1,b2 in a specified range.
    The values for x and y will be distributed linearly.
    """

    def __init__(
        self,
        optimized_parameters: Iterable[Tensor],
        directions: Tuple[List[Tensor], List[Tensor]],
        min_x_value: int = -1,
        max_x_value: int = 1,
        min_y_value: int = -1,
        max_y_value: int = 1,
        num_data_point: int = 50,
        options: Optional[VisualizationOptions] = None,
        n_jobs: int = 1,
        parallel_backend: str = "loky",
    ):
        """
        Initializes the linear landscape calculator.

        :param optimized_parameters: The optimized parameters, which will correspond to the point at (0,0).
        :param directions: the directions [b1, b2] to use for the x and y-axis.
        :param min_x_value: The minimum value for the x range.
        :param max_x_value: The maximum value for the x range.
        :param num_data_point: The count of points for the x and y range.
        :param min_y_value: The minimum value for the y range.
        :param max_y_value: The maximum value for the y range.
        :param options: (Optional) If provided, will take the minimum/maximum values as well the count of data points
        from the options.
        :param n_jobs: The number of jobs to use for parallelization.
        """
        if options is not None:
            min_x_value = options.min_x_value
            min_y_value = options.min_y_value
            max_x_value = options.max_x_value
            max_y_value = options.max_y_value
            num_data_point = options.num_points

        self._min_x_value = min_x_value
        self._max_x_value = max_x_value
        self._min_y_value = min_y_value
        self._max_y_value = max_y_value
        self._num_data_point = num_data_point
        self._directions = directions
        self._optimized_parameters = clone_parameters(optimized_parameters, True)
        # b1 is basis vector for x coordinate, b2 is basis vector
        # for y coordinate.
        self._b1, self._b2 = self._directions

        # calculate steps to make in x/y direction.
        self._x_step = (self._max_x_value - self._min_x_value) / (self._num_data_point - 1)
        self._y_step = (self._max_y_value - self._min_y_value) / (self._num_data_point - 1)

        self._b1_step = [b1_part * self._x_step for b1_part in self._b1]
        self._b2_step = [b2_part * self._y_step for b2_part in self._b2]
        # For b2, we also need the negative direction. For example, start at
        # x=x_min, y=y_min; inner loop iterates until y=y_max. Then x is
        # increased by one step, and y should decrease until it reaches y_min.
        self._b2_step_negative = [-b2_part for b2_part in self._b2_step]
        # set the number of jobs to use for parallelization
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

    def calculate_loss_surface_data_model(
        self,
        model: Module,
        evaluation_function: Callable[[], float],
    ) -> LossSurfaceData:
        """
        Calculates the loss landscape for the specified model using the specified evaluation function.

        :param model: The torch model to use to calculate the loss landscape dictionary.
        :param evaluation_function: The evaluation function which takes one parameter (the model) and returns the loss.
        :return: The loss landscape in a dictionary.
        """
        surface_dict = self.calculate_loss_surface_data([*model.parameters()], lambda p: evaluation_function())
        return surface_dict

    def calculate_loss_surface_data(
        self,
        model_parameters: List[Tensor],
        evaluation_function: Callable[[Iterable[Tensor]], float],
    ) -> LossSurfaceData:
        """
        Calculates the loss surface for the specified model.
        :param model_parameters: the parameters of the model.
        :param evaluation_function: a function which should evaluate the loss of the model with the specified
        parameters, provided as first argument.
        :return: the loss surface data.
        """
        original_parameters = clone_parameters(model_parameters)
        surface_data = setup_surface_data_linear(
            self._min_x_value, self._max_x_value, self._num_data_point, self._min_y_value, self._max_y_value
        )

        with no_grad():
            # Iterate through the x/y grid in a snake-like ordering, such
            # that we can modify the
            # parameters of the model by using in-place additions only.
            # Example:
            # 1st row: (x_min, y_min), ..., (x_min, y_max),
            # 2nd row: (x_min+x_step,y_max), ..., (x_min+x_step, y_min),
            # 3rd row: (x_min+2*x_step, y_min), ...,(x_min+2*x_step, y_max),..

            results = ProgressParallel(n_jobs=self.n_jobs, batch_size=1, backend=self.parallel_backend)(
                delayed(_evaluate_row_parallel)(
                    evaluation_function=evaluation_function,
                    is_positive_direction=bool(x_i % 2 == 0),
                    model_parameters=model_parameters,
                    x_i=x_i,
                    min_x_value=self._min_x_value,
                    max_x_value=self._max_x_value,
                    min_y_value=self._min_y_value,
                    max_y_value=self._max_y_value,
                    num_data_point=self._num_data_point,
                    directions=self._directions,
                    optimized_parameters=self._optimized_parameters,
                )
                for x_i in range(len(surface_data.x_coordinates))
            )
            # Parallel processes which are associated to x-direction (b1 direction).
            # Evaluate the current row (go through all values for y-direction (b2 direction)).

            # merge results into surface_data
            for result in results:
                for (x_i, y_j), z in result.items():
                    surface_data.z_coordinates[y_j, x_i] = z

            # reset parameters of the model to the original parameters.
            reset_params(model_parameters, original_parameters)

        return surface_data
