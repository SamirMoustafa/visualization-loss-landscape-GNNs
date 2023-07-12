from abc import ABC, abstractmethod
from copy import deepcopy
from logging import getLogger
from typing import Iterable, List, Optional, Tuple, Union

from math import sqrt
from torch import Tensor, cov, cuda, device, lobpcg, randn, sort, stack, sum
from torch.linalg import eigh
from torch.nn import Module, MSELoss, Parameter, init
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Adam
from tqdm import tqdm

from torch_landscape.utils import clone_parameters


def normalize_direction_using_params(direction: List[Tensor], parameters: List[Tensor]):
    """
    Normalize each direction in the input list with respect to the corresponding parameters.

    :param direction: List containing PyTorch tensors representing directions.
    :param parameters: List containing PyTorch tensors representing the optimized parameters.
    :return: None
    """
    assert len(direction) == len(parameters)
    for direction_i, parameter_i in zip(direction, parameters):
        normalize_direction(direction_i, parameter_i)


def normalize_direction(direction: Tensor, parameters: Tensor):
    """
    Normalize the direction vector with respect to the weights.

    :param direction: PyTorch tensor representing a direction.
    :param parameters: List containing PyTorch tensors representing the optimized parameters.
    :return: None
    """
    # take the norm along the first dimension, which means the norm is calculated against all dimensions
    # but the first one.
    norm_dimensions = 1 if parameters.dim() > 1 else 0
    norm_direction = direction.norm(dim=norm_dimensions, keepdim=True)
    norm_parameters = parameters.norm(dim=norm_dimensions, keepdim=True)

    scaling_factor = norm_parameters / (norm_direction + 1e-10)
    direction *= scaling_factor


class Directions(ABC):
    """
    Abstract class which describes the methods to calculate directions.
    """

    def __init__(self, optimized_parameters: List[Tensor]):
        """
        Initializes the directions calculations class.
        :param optimized_parameters: The optimized parameters of the model.
        """
        self._optimized_parameters = optimized_parameters

    @abstractmethod
    def calculate_directions(self) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Calculates two directions using the optimized parameters.
        :return: two directions of the same shape as the optimized parameters.
        """
        raise NotImplementedError()


class RandomDirections(Directions):
    """
    Calculates directions for visualization using random numbers and filter normalization.
    """

    def __init__(self, optimized_parameters: Optional[List[Tensor]] = None, model: Optional[Module] = None):
        """
        Initializes the Random directions class. Provide either the optimized parameters or the pytorch model.
        :param optimized_parameters: (Optional) The parameters of the optimized model.
        :param model: (Optional) The pytorch model for which to create the random directions.
        """

        if model is not None and optimized_parameters is None:
            optimized_parameters = [*model.parameters()]

        if optimized_parameters is None:
            raise ValueError("Either optimized parameters or model must be provided.")

        super().__init__(optimized_parameters)

    def calculate_directions(self, apply_normalization: bool = True) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Gets random directions.
        :param apply_normalization: Set to true to apply filter normalization to the created directions.
        :return: [b1, b2] where b1 and b2 are random directions.
        """
        return self.create_random_directions_from_parameters(self._optimized_parameters, apply_normalization)

    @staticmethod
    def create_random_directions_from_parameters(
        model_parameters: List[Tensor], apply_normalization: bool = True
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Create a random direction for the model.

        :param model_parameters: The parameters of the model.
        :param apply_normalization: Set to true, to apply filter normalization.
        :return: PyTorch tensor representing random direction.
        """
        x_direction = RandomDirections.create_random_direction_from_parameters(model_parameters, apply_normalization)
        y_direction = RandomDirections.create_random_direction_from_parameters(model_parameters, apply_normalization)
        return x_direction, y_direction

    @staticmethod
    def create_random_direction_from_parameters(
        parameters: Iterable[Tensor], apply_filter_normalization: bool = True
    ) -> List[Tensor]:
        """
        Create a random direction for the model.

        :param parameters: the parameters of a model.
        :param apply_filter_normalization: Set to true, to apply filter normalization.
        :return: PyTorch tensor representing random direction.
        """
        parameters = [p.data for p in parameters]
        direction = RandomDirections.get_random_parameters(parameters)
        if apply_filter_normalization:
            normalize_direction_using_params(direction, parameters)
        return direction

    @staticmethod
    def get_random_parameters(parameters: List[Tensor]) -> List[Tensor]:
        """
        Generate random parameters for a given list of parameters.

        :param parameters: List containing PyTorch tensors representing parameters of a model.
        :return: List containing PyTorch tensors with the same shape as the input list.
        """
        return [randn(w.size(), device=w.device) for w in parameters]

    @staticmethod
    def normalize_direction(direction: List[Tensor], parameters: List[Tensor]):
        """
        Normalize the direction vector with respect to the weights.

        :param direction: PyTorch tensor representing a direction.
        :param parameters: List containing PyTorch tensors.
        :return: None
        """
        for direction_i, parameters_i in zip(direction, parameters):
            direction_i = direction_i.to(parameters_i.device)
            direction_i.mul_(parameters_i.norm() / (direction_i.to(parameters_i.device).norm() + 1e-10))


class PcaDirections(Directions):
    """
    Calculates directions using the intermediate parameters calculated during training.
    """

    logger = getLogger("visualizations_directions")

    def __init__(
        self,
        optimized_parameters: List[Tensor],
        intermediate_parameters: Union[List[List[Tensor]], List[Tuple[List[Tensor], float]]],
        covariance_device: device = device("cpu"),
    ):
        """
        Initializes the pca directions calculations class.
        :param optimized_parameters: The optimized parameters from the model.
        :param intermediate_parameters: The parameters calculated during training with or without the loss.
        :param covariance_device: The device to use to calculate the covariance matrix. Should be set to a device
        with lots of memory, because the covariance matrix size is the square of the count of parameters.
        """
        super().__init__(optimized_parameters)
        if len(intermediate_parameters) == 0:
            raise ValueError("Intermediate parameters must not be empty.")
        self._optimized_parameters = optimized_parameters

        # Check if intermediate parameters were provided with loss.
        if isinstance(intermediate_parameters[0], tuple):
            self._intermediate_parameters = [parameters for parameters, loss in intermediate_parameters]
        else:
            self._intermediate_parameters = intermediate_parameters
        self._covariance_device = covariance_device

    def calculate_directions(self) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Calculates the directions using PCA on the covariance matrix of the intermediate parameters.
        :return: Two directions in the parameter space.
        """
        b1, b2 = PcaDirections.create_pca_directions(
            self._intermediate_parameters, self._optimized_parameters, pca_device=self._covariance_device
        )
        b1_param = clone_parameters(self._optimized_parameters)
        b2_param = clone_parameters(self._optimized_parameters)
        vector_to_parameters(b1, b1_param)
        vector_to_parameters(b2, b2_param)

        return b1_param, b2_param

    @staticmethod
    def calculate_covariance_matrix(samples: List[Tensor]) -> Tensor:
        """
        Calculates the covariance matrix of the provided samples with low memory (does not create the matrix
        holding all the samples).
        :param samples: A list of the feature vectors for each sample.
        :return: The covariance matrix.
        """
        if samples[0].numel()**2 * 32 * 0.000000000125 > 210:
            raise ValueError(
                "The covariance matrix is too large to fit in memory. "
                "Try using a smaller batch size or a smaller model."
            )
        covariance_matrix = cov(stack(samples, dim=1))
        return covariance_matrix

    @staticmethod
    def create_pca_directions(
        intermediate_results: List[List[Tensor]], parameters: List[Tensor], pca_device=None
    ) -> Tuple[Tensor, Tensor]:
        """
        Creates directions for visualizing the loss landscape using PCA of the intermediate parameters (which were
        calculated during training).
        :param intermediate_results: List of intermediate parameters.
        :param parameters: The "best" parameters to subtract from each intermediate result.
        :param pca_device: (optional) The device on which the PCA is calculated.
        :return: List containing two basis vectors for the parameter space.
        """
        if len(intermediate_results) == 0:
            raise ValueError("Intermediate results must not be empty.")

        parameters_vector = parameters_to_vector(parameters)
        # subtract the optimized parameters from each intermediate parameter.
        results = [
            (parameters_to_vector(result) - parameters_vector).to(device=pca_device) for result in intermediate_results
        ]

        PcaDirections.logger.debug("Calculating covariance matrix")
        covariance_matrix = PcaDirections.calculate_covariance_matrix(results)
        eigen_values, eigen_vectors = PcaDirections._calculate_eigenpairs(covariance_matrix)

        _, indices = sort(eigen_values, descending=True)
        eigen_values_sum = sum(eigen_values)
        PcaDirections.logger.info("1st PC explains: {}%".format(100 * eigen_values[indices[0]] / eigen_values_sum))
        PcaDirections.logger.info("2nd PC explains: {}%".format(100 * eigen_values[indices[1]] / eigen_values_sum))

        # store resulting directions on the same device as the optimized parameters.
        target_device = parameters[0].device
        b1 = eigen_vectors[:, indices[0]].clone().detach().to(device=target_device)
        b2 = eigen_vectors[:, indices[1]].clone().detach().to(device=target_device)

        return b1, b2

    @staticmethod
    def _calculate_eigenpairs(covariance_matrix: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculates the eigenpairs of the covariance matrix.
        :param covariance_matrix: The covariance matrix.
        :return: The eigenvalues and eigen vectors.
        """
        # Check if LOBPCG algorithm can be used: rows of matrix must be larger than 3x(number of eigenpairs=2).
        if covariance_matrix.size(dim=0) > 3 * 2:
            PcaDirections.logger.debug("Using LOBPCG algorithm to calculate eigenpairs.")
            eigen_values, eigen_vectors = lobpcg(covariance_matrix, largest=True, k=2)
        else:
            PcaDirections.logger.debug("Using torch.linalg.eigh to calculate eigenpairs.")
            eigen_values, eigen_vectors = eigh(covariance_matrix)
        return eigen_values, eigen_vectors


class LearnableDirections(Directions):
    """
    Calculates directions using the intermediate parameters calculated during training.
    """

    logger = getLogger("visualizations_directions")

    def __init__(
        self,
        optimized_parameters: List[Tensor],
        intermediate_parameters: Union[List[List[Tensor]], List[Tuple[List[Tensor], float]]],
        learnable_model_device: device = device("cpu"),
        autoencoder_lr: float = 0.01,
        training_epochs: int = 1000,
        early_stopping_epochs: int = 100,
    ):
        """
        Initializes the pca directions calculations class.
        :param optimized_parameters: The optimized parameters from the model.
        :param intermediate_parameters: The parameters calculated during training with or without the loss.
        :param learnable_model_device: The device to use to calculate the covariance matrix. Should be set to a device
        :param autoencoder_lr: The learning rate used for training the autoencoder.
        :param training_epochs: The count of epochs to train the autoencoder.
        :param early_stopping_epochs: The count of epochs in which no progress is made until training is stopped.
        with lots of memory, because the covariance matrix size is the square of the count of parameters.
        """
        super().__init__(optimized_parameters)
        if len(intermediate_parameters) == 0:
            raise ValueError("Intermediate parameters must not be empty.")
        self._optimized_parameters = optimized_parameters

        # Check if intermediate parameters were provided with loss.
        if isinstance(intermediate_parameters[0], tuple):
            self._intermediate_parameters = [parameters for parameters, loss in intermediate_parameters]
        else:
            self._intermediate_parameters = intermediate_parameters
        self.learnable_model_device = learnable_model_device
        self.autoencoder_lr = autoencoder_lr
        self.training_epochs = training_epochs
        self.early_stopping_epochs = early_stopping_epochs

    def calculate_directions(self) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Calculates the directions using PCA on the covariance matrix of the intermediate parameters.
        :return: Two directions in the parameter space.
        """
        b1, b2 = LearnableDirections.create_learnable_directions(
            self._intermediate_parameters,
            self._optimized_parameters,
            learnable_model_device=self.learnable_model_device,
            autoencoder_lr=self.autoencoder_lr,
            training_epochs=self.training_epochs,
            early_stopping_epochs=self.early_stopping_epochs,
        )
        b1_param = clone_parameters(self._optimized_parameters)
        b2_param = clone_parameters(self._optimized_parameters)
        vector_to_parameters(b1, b1_param)
        vector_to_parameters(b2, b2_param)

        return b1_param, b2_param

    @staticmethod
    def create_learnable_directions(
        intermediate_results: List[List[Tensor]],
        parameters: List[Tensor],
        learnable_model_device=None,
        early_stopping_epochs: int = 100,
        training_epochs: int = 1000,
        autoencoder_lr: float = 0.01,
    ) -> Tuple[Tensor, Tensor]:
        """
        Creates directions for visualizing the loss landscape using PCA of the intermediate parameters (which were
        calculated during training).
        :param intermediate_results: List of intermediate parameters.
        :param parameters: The "best" parameters to subtract from each intermediate result.
        :param learnable_model_device: (optional) The device on which the PCA is calculated.
        :param autoencoder_lr: The learning rate used for training the autoencoder.
        :param training_epochs: The count of epochs to train the autoencoder.
        :param early_stopping_epochs: The count of epochs in which no progress is made until training is stopped.
        :return: List containing two basis vectors for the parameter space.
        """
        if len(intermediate_results) == 0:
            raise ValueError("Intermediate results must not be empty.")

        class LearnableAutoEncoder(Module):
            def __init__(self, feature_dim: int):
                super(LearnableAutoEncoder, self).__init__()
                self.weight = Parameter(Tensor(2, feature_dim), requires_grad=True)
                self.reset_parameters()

            def reset_parameters(self):
                init.kaiming_uniform_(self.weight, a=sqrt(5))

            def forward(self, x):
                x = x @ self.weight.t()
                x = x @ self.weight
                return x

        optimized_parameters_vector = parameters_to_vector(parameters)
        # subtract the optimized parameters from each intermediate parameter.
        x = [
            (parameters_to_vector(result) - optimized_parameters_vector).to(device=learnable_model_device)
            for result in intermediate_results
        ]
        dataset = stack(x)  # matrix of size NxF
        feature_dim = len(x[0])
        criterion = MSELoss(reduction="sum")
        model = LearnableAutoEncoder(feature_dim=feature_dim)
        model.to(device=learnable_model_device)
        optimizer = Adam(model.parameters(), lr=autoencoder_lr, weight_decay=2e-2)

        model.train()
        best_epoch, best_loss, best_model_state_dict = 0, float("inf"), None
        pbar = tqdm(range(training_epochs))
        for epoch in pbar:
            epoch_loss = 0
            optimizer.zero_grad()
            y_pred = model(dataset)
            loss = criterion(y_pred, dataset)
            # If L1 loss is used, the following statement calculates the mean reconstruction error when calculating
            # the coordinates using scalar product.
            # loss = loss.norm(p=1) / len(x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_state_dict = deepcopy(model.state_dict())

            # stop if no improvement for some epochs is detected.
            if epoch - best_epoch >= early_stopping_epochs:
                break
            pbar.set_description(f"Best Epoch: {best_epoch}, Best Loss: {best_loss:.4f}")

        model.load_state_dict(best_model_state_dict)
        b1, b2 = model.weight[0, :], model.weight[1, :]
        return b1, b2
