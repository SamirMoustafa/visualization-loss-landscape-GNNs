from unittest import TestCase, main

from torch import tensor
from torch.nn.utils import parameters_to_vector

from torch_landscape.trajectory import TrajectoryCalculator, TrajectoryData
from torch_landscape.visualize import VisualizationOptions


class TrajectoryTest(TestCase):
    def test_to_visualization_space(self):
        """
        Tests if the conversion of a vector in parameter space to visualization space is correct for
        an example, which can be represented exactly in the visualization space.
        """
        zero_vector = tensor([8, 9, 10, 11, 12, 13])
        b1 = tensor([1, 2, 3, 4, 5, 6])
        b2 = tensor([1, 9, 7, 1, 5, 3])
        expected_x, expected_y = 1, -1
        vector = zero_vector + expected_x * b1 + expected_y * b2

        result = TrajectoryCalculator.to_visualization_space(vector, zero_vector, b1, b2)

        actual_x, actual_y = result
        self.assertAlmostEqual(expected_x, actual_x, 12)
        self.assertAlmostEqual(expected_y, actual_y, 12)

    def test_to_visualization_space_projection(self):
        """
        Tests if the conversion of a vector in parameter space to visualization space is correct for
        an example, which cannot be represented exactly in the visualization space.
        """
        zero_vector = tensor([8, 9, 10, 11, 12, 13])
        b1 = tensor([1, 2, 3, 4, 5, 6])
        b2 = tensor([1, 9, 7, 1, 5, 3])
        expected_x, expected_y = 1, -1
        vector = zero_vector + expected_x * b1 + expected_y * b2 + tensor([0.005, 0.02, 0.007, 0.0011, 0.009, 0.008])

        result = TrajectoryCalculator.to_visualization_space(vector, zero_vector, b1, b2)

        actual_x, actual_y = result
        self.assertAlmostEqual(expected_x, actual_x, 2)
        self.assertAlmostEqual(expected_y, actual_y, 2)

    def test_calculate_reconstruction_error(self):
        """
        Tests if the reconstruction error is calculated correctly.
        """
        zero_vector = [tensor([8, 9]), tensor([10, 11, 12, 13])]
        b1 = [tensor([1, 2]), tensor([3, 4, 5, 6])]
        b2 = [tensor([1, 9]), tensor([7, 1, 5, 3])]

        vector = [tensor([9.0, 2.0]), tensor([9.4, 1.2, 3.4, 7.4])]
        # calculate coordinates of vector.
        actual_x, actual_y = TrajectoryCalculator.to_visualization_space(
            parameters_to_vector(vector),
            parameters_to_vector(zero_vector),
            parameters_to_vector(b1),
            parameters_to_vector(b2),
        )
        # reconstruct the vector.
        actual_vector = [
            zero_vector[0] + actual_x * b1[0] + actual_y * b2[0],
            zero_vector[1] + actual_x * b1[1] + actual_y * b2[1],
        ]
        difference = [vector[0] - actual_vector[0], vector[1] - actual_vector[1]]
        expected_error = difference[0].norm(p=1).item() + difference[1].norm(p=1).item()
        actual_error = TrajectoryCalculator.calculate_reconstruction_error(b1, b2, zero_vector, vector)
        self.assertAlmostEqual(expected_error, actual_error, 4)

    def test_transform_intermediate_parameters_with_loss(self):
        """
        Tests if the intermediate parameters are transformed correctly to the visualization space.
        """
        intermediate_results = [([tensor([1, 2]), tensor([3])], 5), ([tensor([4, 5]), tensor([3])], 6)]
        zero_point = [tensor([0, 0]), tensor([3])]
        b1 = [tensor([1, 0]), tensor([0])]
        b2 = [tensor([0, 1]), tensor([0])]
        calculator = TrajectoryCalculator(zero_point, (b1, b2))
        actual_coordinates = calculator.project_with_loss(intermediate_results).points
        expected_coordinates = [(1, 2, 5), (4, 5, 6)]
        self.assertSequenceEqual(expected_coordinates, actual_coordinates)

    def test_transform_intermediate_parameters_disregard_z(self):
        """
        Tests if the intermediate parameters without loss values are transformed correctly to the visualization space.
        """
        intermediate_results = [[tensor([1, 2]), tensor([3])], [tensor([4, 5]), tensor([3])]]
        zero_point = [tensor([0, 0]), tensor([3])]
        b1 = [tensor([1, 0]), tensor([0])]
        b2 = [tensor([0, 1]), tensor([0])]
        calculator = TrajectoryCalculator(zero_point, (b1, b2))
        actual_coordinates = calculator.project_disregard_z(intermediate_results, -4).points
        expected_coordinates = [(1, 2, -4), (4, 5, -4)]
        self.assertSequenceEqual(expected_coordinates, actual_coordinates)

    def test_calculate_ranges(self):
        """
        Tests if the function calculate_ranges works correctly.
        """
        intermediate_results = [(1, 3, 3), (4, 5, 6), (7, 8, 9)]
        x_max, x_min, y_max, y_min = TrajectoryData.calculate_ranges(intermediate_results, 0.5)
        self.assertAlmostEqual(1 - 6 * 0.5, x_min)
        self.assertAlmostEqual(7 + 6 * 0.5, x_max)
        self.assertAlmostEqual(3 - 5 * 0.5, y_min)
        self.assertAlmostEqual(8 + 5 * 0.5, y_max)

    def test_set_range_to_fit_trajectory(self):
        """
        Tests if the set_range_to_fit_trajectory correctly modifies the VisualizationOptions.
        """
        points = [(1, 3, 3), (4, 5, 6), (7, 8, 9)]
        options = VisualizationOptions()
        data = TrajectoryData(points)
        data.set_range_to_fit_trajectory(options, 0.5)
        self.assertAlmostEqual(1 - 6 * 0.5, options.min_x_value)
        self.assertAlmostEqual(7 + 6 * 0.5, options.max_x_value)
        self.assertAlmostEqual(3 - 5 * 0.5, options.min_y_value)
        self.assertAlmostEqual(8 + 5 * 0.5, options.max_y_value)

    def test_filter_points_by_range(self):
        """
        Tests if filter_points_by_range correctly filters out points, which are not within the visualization range.
        """
        points = [(1, -1, 3), (-1, 1, 6), (1, 1, 9), (-1, -1, 5), (9, 3, 2), (10, -1, 4), (1, -10, 3)]
        options = VisualizationOptions()
        data = TrajectoryData(points).filter_points_by_range(options)
        self.assertSequenceEqual(points[:4], data.points)


if __name__ == "__main__":
    main()
