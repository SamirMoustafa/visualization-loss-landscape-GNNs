from unittest import TestCase, main

from numpy import meshgrid
from numpy.ma.testutils import assert_allclose
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

        # Compute the affine transformations using the parameter tensors
        affine_1 = self.init_params[0] + x * self.b1[0] + y * self.b2[0]
        affine_2 = x * self.b1[1] + y * self.b2[1] + self.init_params[1]
        # Perform matrix multiplication and addition
        output = matmul(affine_1, self.test_input) + affine_2
        return output.item()

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

        calculator = LinearLandscapeCalculator(optimized_parameters=self.init_params,
                                               directions=self.directions,
                                               min_x_value=-1,
                                               max_x_value=1,
                                               min_y_value=-3,
                                               max_y_value=-4,
                                               num_data_point=2,
                                               )
        surface_data = calculator.calculate_loss_surface_data(model_parameters=self.init_params,
                                                              evaluation_function=parameter_summation_func)

        expected_x_coordinates = [-1.0, 1.0]
        assert_allclose(expected_x_coordinates, surface_data.x_coordinates)
        expected_y_coordinates = [-3.0, -4.0]
        assert_allclose(expected_y_coordinates, surface_data.y_coordinates)

        x_c, y_c = meshgrid(surface_data.x_coordinates, surface_data.y_coordinates)
        expected_grid = [
            [
                parameter_summation_func([self.init_params[0] + x * self.b1[0] + y * self.b2[0]])
                for x, y in zip(x_row, y_row)
            ]
            for x_row, y_row in zip(x_c, y_c)
        ]
        actual_grid = surface_data.z_coordinates
        assert_allclose(expected_grid, actual_grid)

    def test_calculate_surface_landscape_linear(self):
        calculator = LinearLandscapeCalculator(optimized_parameters=self.init_params,
                                               directions=self.directions,
                                               min_x_value=-2,
                                               max_x_value=-1,
                                               min_y_value=1,
                                               max_y_value=2,
                                               num_data_point=3,
                                               )
        surface_data = calculator.calculate_loss_surface_data_model(model=self.model,
                                                                    evaluation_function=self.eval_model)

        expected_x_coordinates = [-2.0, -1.5, -1.0]
        assert_allclose(expected_x_coordinates, surface_data.x_coordinates)
        expected_y_coordinates = [1.0, 1.5, 2.0]
        assert_allclose(expected_y_coordinates, surface_data.y_coordinates)

        grid_x, grid_y = meshgrid(tensor(surface_data.x_coordinates), tensor(surface_data.y_coordinates))
        expected_grid = [
            [self.eval_model_xy(x, y) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(grid_x, grid_y)
        ]

        assert_allclose(expected_grid, surface_data.z_coordinates)

    # @skip("Test does not work with pytest")
    def test_calculate_surface_landscape_linear_parallel(self):
        """
        Checks if the calculations run in parallel give the correct results.
        This test does not run with pytest.
        """
        for workers in range(1, 8):
            calculator = LinearLandscapeCalculator(optimized_parameters=self.init_params,
                                                   directions=self.directions,
                                                   min_x_value=-1,
                                                   max_x_value=-1,
                                                   min_y_value=1,
                                                   max_y_value=1,
                                                   num_data_point=32,
                                                   options=None,
                                                   n_jobs=workers,
                                                   parallel_backend="threading")
            surface_data = calculator.calculate_loss_surface_data_model(model=self.model,
                                                                        evaluation_function=self.eval_model)

            grid_x, grid_y = meshgrid(tensor(surface_data.x_coordinates), tensor(surface_data.y_coordinates))
            expected_grid = [
                [self.eval_model_xy(x, y) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(grid_x, grid_y)
            ]
            # rtol is set to 1 because there a single elements that shift by 1. in the parallel run.
            assert_allclose(expected_grid, surface_data.z_coordinates, rtol=1, atol=0)

    def test_calculate_surface_landscape_linear_should_reset_parameters(self):
        """
        Checks if the calculate_surface_landscape_linear resets the parameters
        to the initial parameters of the model after calculating the loss landscape.
        """
        init_parameters = clone_parameters(self.model.parameters())

        calculator = LinearLandscapeCalculator(optimized_parameters=self.init_params,
                                               directions=self.directions,
                                               min_x_value=-2,
                                               max_x_value=-1,
                                               min_y_value=1,
                                               max_y_value=2,
                                               num_data_point=3)
        calculator.calculate_loss_surface_data_model(model=self.model,
                                                     evaluation_function=self.eval_model)
        actual_parameters = parameters_to_vector(self.model.parameters()).detach().numpy()
        expected_parameters = parameters_to_vector(init_parameters).detach().numpy()

        assert_allclose(expected_parameters, actual_parameters)

    def test_constructor_should_take_numbers_from_options(self):
        """
        Tests if the constructor uses the min/max values from the options, if provided.
        """
        options = VisualizationOptions(-4, 9, -5, 7, 100)
        calculator = LinearLandscapeCalculator(optimized_parameters=self.init_params,
                                               directions=self.directions,
                                               options=options)
        self.assertEqual(-4.0, calculator._min_x_value)
        self.assertEqual(-5.0, calculator._min_y_value)
        self.assertEqual(9.0, calculator._max_x_value)
        self.assertEqual(7.0, calculator._max_y_value)
        self.assertEqual(100.0, calculator._num_data_point)


if __name__ == "__main__":
    main()
