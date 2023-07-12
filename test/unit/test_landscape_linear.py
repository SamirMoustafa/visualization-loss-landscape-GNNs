from unittest import TestCase, main

from numpy import allclose, meshgrid
from torch import Tensor, matmul, sum, tensor
from torch.nn import Linear
from torch.nn.utils import parameters_to_vector

from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.utils import clone_parameters
from torch_landscape.visualize_options import VisualizationOptions


class LinearLandscapeCalculatorTest(TestCase):
    def eval_model(self) -> float:
        """
        Evaluates the test using the current parameters and the test input.
        :return: The evaluated value.
        """
        return self.model.forward(self.test_input).item()

    def eval_model_xy(self, x: float, y: float) -> float:
        """
        Evaluates the test model using the parameters
        (init_params + x*b1 + y*b2).
        and the test input.
        :param x: The x coordinate
        :param y: The y coordinate
        :return: The value of the model.
        """
        return (
            matmul(self.init_params[0] + x * self.b1[0] + y * self.b2[0], self.test_input)
            + (x * self.b1[1] + y * self.b2[1] + self.init_params[1])
        ).item()

    def setUp(self):
        self.model = Linear(4, 1)

        self.test_input = Tensor([1, 2, 1, 3])

        self.b1 = [Tensor([[1, 2, 3, 4]]), Tensor([0])]
        self.b2 = [Tensor([[4, 7, 1, 9]]), Tensor([1])]
        self.directions = (self.b1, self.b2)
        # the initial_params are the "optimized" parameters.
        self.init_params = [Tensor([[2, 3, 6, 7]]), Tensor([8])]

    def test_calculate_surface_landscape_generic(self):
        """
        Tests the calculation of a loss landscape with a custom loss function.
        """

        def parameter_summation_func(p):
            return sum(next(iter(p))).item()

        calculator = LinearLandscapeCalculator(self.init_params, self.directions, -1, 1, 2, -3, -4)
        surface_data = calculator.calculate_loss_surface_data(self.init_params, parameter_summation_func)

        self.assertSequenceEqual([-1, 1], surface_data.x_coordinates.tolist())
        self.assertSequenceEqual([-3, -4], surface_data.y_coordinates.tolist())

        x_c, y_c = meshgrid(surface_data.x_coordinates, surface_data.y_coordinates)
        expected_grid = [
            [
                parameter_summation_func([self.init_params[0] + x * self.b1[0] + y * self.b2[0]])
                for x, y in zip(x_row, y_row)
            ]
            for x_row, y_row in zip(x_c, y_c)
        ]
        actual_grid = surface_data.z_coordinates
        self.assertTrue(allclose(expected_grid, actual_grid))

    def test_calculate_surface_landscape_linear(self):
        calculator = LinearLandscapeCalculator(self.init_params, self.directions, -2, -1, 3, 1, 2)
        surface_data = calculator.calculate_loss_surface_data_model(self.model, self.eval_model)

        self.assertSequenceEqual([-2.0, -1.5, -1.0], surface_data.x_coordinates.tolist())
        self.assertSequenceEqual([1.0, 1.5, 2.0], surface_data.y_coordinates.tolist())

        grid_x, grid_y = meshgrid(tensor(surface_data.x_coordinates), tensor(surface_data.y_coordinates))
        expected_grid = [
            [self.eval_model_xy(x, y) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(grid_x, grid_y)
        ]

        self.assertTrue(allclose(expected_grid, surface_data.z_coordinates))

    def test_calculate_surface_landscape_linear_should_reset_parameters(self):
        """
        Checks if the calculate_surface_landscape_linear resets the parameters
        to the initial parameters of the model after calculating the loss landscape.
        """
        init_parameters = clone_parameters(self.model.parameters())

        calculator = LinearLandscapeCalculator(self.init_params, self.directions, -2, -1, 3, 1, 2)
        calculator.calculate_loss_surface_data_model(self.model, self.eval_model)
        actual_parameters = parameters_to_vector(self.model.parameters()).detach().numpy()
        expected_parameters = parameters_to_vector(init_parameters).detach().numpy()

        self.assertTrue(allclose(expected_parameters, actual_parameters))

    def test_constructor_should_take_numbers_from_options(self):
        """
        Tests if the constructor uses the min/max values from the options, if provided.
        """
        options = VisualizationOptions(-4, 9, -5, 7, 100)
        calculator = LinearLandscapeCalculator(self.init_params, self.directions, options=options)
        self.assertEqual(-4, calculator._min_x_value)
        self.assertEqual(-5, calculator._min_y_value)
        self.assertEqual(9, calculator._max_x_value)
        self.assertEqual(7, calculator._max_y_value)
        self.assertEqual(100, calculator._num_data_point)


if __name__ == "__main__":
    main()
