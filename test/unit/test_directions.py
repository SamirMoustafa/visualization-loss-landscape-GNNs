from math import sqrt
from unittest import TestCase, main
from torch import allclose, norm, rand, sign, sort, tensor
from torch.linalg import eigh
from torch.nn import Linear
from torch.nn.utils import parameters_to_vector

from torch_landscape.directions import PcaDirections, RandomDirections, normalize_direction, LearnableDirections
from torch_landscape.trajectory import TrajectoryCalculator


class DirectionsTest(TestCase):
    def test_calculate_covariance_matrix(self):
        """
        Tests the calculation of a covariance matrix.
        Example is from https://jamesmccaffrey.wordpress.com/2017/11/03/example-of-calculating-a-covariance-matrix/.
        """
        samples = [
            tensor([64, 580, 29]),
            tensor([66, 570, 33]),
            tensor([68, 590, 37]),
            tensor([69, 660, 46]),
            tensor([73, 600, 55]),
        ]
        expected_matrix = tensor([[11.5, 50, 34.75], [50, 1250, 205], [34.75, 205, 110]])
        actual_matrix = PcaDirections.calculate_covariance_matrix(samples)
        self.assertTrue(allclose(actual_matrix, expected_matrix))

    def test_create_pca_directions_intermediate_parameters_with_loss(self):
        """
        Tests if the intermediate parameters with loss are correctly converted in the constructor.
        """
        s1 = ([tensor([4]), tensor([11])], 100)
        s2 = ([tensor([8]), tensor([4])], 94)

        samples = [s1, s2]
        directions = PcaDirections(s1[0], samples)
        expected_parameters = [s1[0], s2[0]]
        self.assertSequenceEqual(expected_parameters, directions._intermediate_parameters)

    def test_create_pca_directions_from_example(self):
        """
        Tests if the PCA eigenvectors are calculated correctly
        (from https://www.vtupulse.com/machine-learning/principal-component-analysis-solved-example/).
        """
        s1 = [tensor([4]), tensor([11])]
        s2 = [tensor([8]), tensor([4])]
        s3 = [tensor([13]), tensor([5])]
        s4 = [tensor([7]), tensor([14])]

        samples = [s1, s2, s3, s4]
        b1, b2 = PcaDirections.create_pca_directions(samples, s1)

        expected_b1 = tensor([-0.55739, 0.8303])
        expected_b2 = tensor([-0.8303, -0.5574])
        self.assertTrue(allclose(expected_b1, b1, atol=1e-4))
        self.assertTrue(allclose(expected_b2, b2, atol=1e-4))

    def test_create_pca_directions_vectors_lobpcg(self):
        """
        Tests if the PCA directions are calculated correctly using LOBPCG algorithm by comparing
        results with calculations of "eigh".
        """
        samples = [[rand(8)] for _ in range(10)]

        results = [parameters_to_vector(result) for result in samples]
        covariance_matrix = PcaDirections.calculate_covariance_matrix(results)
        # calculate directions using "eigh" in test - should be equal for this small example to LOBPCG result.
        eigen_values, eigen_vectors = eigh(covariance_matrix)
        _, indices = sort(eigen_values, descending=True)
        expected_b1 = eigen_vectors[:, indices[0]]
        expected_b2 = eigen_vectors[:, indices[1]]

        b1, b2 = PcaDirections.create_pca_directions(samples, samples[0])

        # correct signs of eigenvectors if necessary.
        expected_b1 *= sign(b1[0] * expected_b1[0]).item()
        expected_b2 *= sign(b2[0] * expected_b2[0]).item()

        self.assertTrue(allclose(expected_b1, b1, atol=1e-2))
        self.assertTrue(allclose(expected_b2, b2, atol=1e-2))

    def test_create_pca_directions_check_dimensions(self):
        """
        Tests if the PCA directions have the correct shape.
        """
        s1 = [tensor([100, 100, 9]), tensor([0, 0.01])]
        s2 = [tensor([99, 101, 7]), tensor([0.01, 0])]
        s3 = [tensor([101, 99, 3]), tensor([0.01, 0.01])]
        samples = [s1, s2, s3]
        b1, b2 = PcaDirections(s1, samples).calculate_directions()

        self.assertEqual(2, len(b1))
        self.assertEqual(s1[0].shape, b1[0].shape)
        self.assertEqual(s1[1].shape, b1[1].shape)
        self.assertEqual(2, len(b2))
        self.assertEqual(s1[0].shape, b2[0].shape)
        self.assertEqual(s1[1].shape, b2[1].shape)

    def test_create_pca_directions_vectors_check_dimensions(self):
        """
        Tests if the PCA directions have the correct dimensions.
        """
        s1 = [tensor([100, 100]), tensor([0, 0.01])]
        s2 = [tensor([99, 101]), tensor([0.01, 0])]
        s3 = [tensor([101, 99]), tensor([0.01, 0.01])]
        samples = [s1, s2, s3]
        b1, b2 = PcaDirections.create_pca_directions(samples, s1)
        expected_direction_length = 4
        actual_b1_length = len(b1)
        actual_b2_length = len(b2)

        self.assertEqual(expected_direction_length, actual_b1_length)
        self.assertEqual(expected_direction_length, actual_b2_length)

    def test_create_random_direction_from_parameters_check_tensors(self):
        parameters = [tensor([1.0, 2.0, 3.0]), tensor([4.0, 5.0])]
        random_direction, _ = RandomDirections(parameters).calculate_directions()
        expected_tensor1_length = len(parameters[0])
        expected_tensor2_length = len(parameters[1])
        actual_tensor1_length = len(random_direction[0])
        actual_tensor2_length = len(random_direction[1])
        self.assertEqual(expected_tensor1_length, actual_tensor1_length)
        self.assertEqual(expected_tensor2_length, actual_tensor2_length)

    def test_create_random_directions_check_shape(self):
        """
        Tests if create_random_directions outputs random directions of the correct shape.
        """
        model = Linear(4, 2)
        rand_directions = RandomDirections(model=model).calculate_directions()
        parameters = list(model.parameters())
        b1, b2 = rand_directions
        self.assertEqual(len(parameters), len(b1))
        self.assertEqual(len(parameters), len(b2))
        self.assertEqual(parameters[0].shape, b1[0].shape)
        self.assertEqual(parameters[0].shape, b2[0].shape)
        self.assertEqual(parameters[1].shape, b1[1].shape)
        self.assertEqual(parameters[1].shape, b2[1].shape)

    def test_create_random_directions_check_normalization(self):
        """
        Tests if create_random_directions normalizes the directions with filter normalization.
        """
        model = Linear(4, 2)
        rand_directions = RandomDirections(model=model).calculate_directions()
        parameters = list(model.parameters())
        parameters_norm_row1 = norm(parameters[0]).item()
        parameters_norm_row2 = norm(parameters[1]).item()
        b1, b2 = rand_directions
        # using assertAlmostEqual, because normalization routine uses a small trick to avoid division by zero,
        # which will change the norm slightly.
        self.assertAlmostEqual(parameters_norm_row1, norm(b1[0]).item(), 6)
        self.assertAlmostEqual(parameters_norm_row1, norm(b2[0]).item(), 6)
        self.assertAlmostEqual(parameters_norm_row2, norm(b1[1]).item(), 6)
        self.assertAlmostEqual(parameters_norm_row2, norm(b2[1]).item(), 6)

    def test_create_random_directions_check_normalization_two_dimensions(self):
        """
        Tests if create_random_directions normalizes the directions with filter normalization.
        """
        parameters = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        direction = tensor([[0.4, 0.6, -0.3], [0.4, 0.8, -0.6]])
        normalize_direction(direction, parameters)
        # the norm of the first row of b1 should be equal to the norm of the first row of "parameters".
        norm_parameters_row1 = sqrt(1**2 + 2**2 + 3**2)
        norm_parameters_row2 = sqrt(4**2 + 5**2 + 6**2)
        norm_direction_row1 = norm(direction[0]).item()
        norm_direction_row2 = norm(direction[1]).item()
        self.assertAlmostEqual(norm_parameters_row1, norm_direction_row1, 4)
        self.assertAlmostEqual(norm_parameters_row2, norm_direction_row2, 4)

    def test_create_random_directions_check_normalization_three_dimensions(self):
        """
        Tests if create_random_directions normalizes the directions with filter normalization.
        """
        parameters = tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[4.0, 5.0, 6.0], [2.0, 3.0, 4.0]]])
        direction = tensor([[[2.0, -5.0, -2.0], [-2.0, -6.0, 7.0]], [[3.0, -5.0, -6.0], [5.0, -2.0, -9.0]]])
        normalize_direction(direction, parameters)
        # the norm of the first row of b1 should be equal to the norm of the first row of "parameters".
        norm_parameters_row1 = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).norm(p=2).item()
        norm_parameters_row2 = tensor([[4.0, 5.0, 6.0], [2.0, 3.0, 4.0]]).norm(p=2).item()
        norm_direction_row1 = norm(direction[0], p=2).item()
        norm_direction_row2 = norm(direction[1], p=2).item()
        self.assertAlmostEqual(norm_parameters_row1, norm_direction_row1, 4)
        self.assertAlmostEqual(norm_parameters_row2, norm_direction_row2, 4)

    def test_create_learnable_directions(self):
        """
        Tests if create_learnable_directions calculates directions which have approximately lower or equal
        reconstruction error than PCA directions.
        """
        samples = [
            [tensor([4.0]), tensor([11.0])],
            [tensor([8.0]), tensor([4.0])],
            [tensor([13.0]), tensor([5.0])],
            [tensor([7.0]), tensor([14.0])],
        ]
        zero_point = samples[0]

        pca_b1 = [tensor([-0.55739]), tensor([0.8303])]
        pca_b2 = [tensor([-0.8303]), tensor([-0.5574])]
        expected_error = TrajectoryCalculator.reconstruction_error_mean(pca_b1, pca_b2, zero_point, samples)

        b1, b2 = LearnableDirections.create_learnable_directions(samples, zero_point)
        error = TrajectoryCalculator.reconstruction_error_mean([b1], [b2], zero_point, samples)

        self.assertLess(abs(error - expected_error), 1e-5)


if __name__ == "__main__":
    main()
