import os
from typing import Iterable, List, Optional

from torch import Tensor, no_grad, device
from torch.nn import Module


def seed_everything(seed: Optional[int] = None):
    """
    Sets the seed of torch, torch cuda, torch geometric and native random library.

    :param seed: (Optional) the seed to use.
    """
    seed = 42 if not seed else seed

    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from torch_geometric import seed_everything

    seed_everything(seed)


def clone_parameters(parameters: Iterable[Tensor], detach=False) -> List[Tensor]:
    """
    Clones the parameters of a model.

    :param parameters: The parameters to clone.
    :param detach: Set to true to detach the tensors before cloning.
    :return: The cloned parameters.
    """
    if detach:
        return [param.detach().clone() for param in parameters]
    else:
        return [param.clone() for param in parameters]


def move_parameters(parameters: Iterable[Tensor], to_device: device) -> List[Tensor]:
    """
    Moves the parameters in the list to another device.
    :param parameters: The parameters to move.
    :param to_device: The device to move the parameters to.
    :return: The parameters on the specified device.
    """
    return [param.to(to_device) for param in parameters]


def reset_parameters(model: Module, parameters: List[Tensor]):
    """
    Resets the parameters of a model to the supplied parameters.

    :param model: The model which parameters should be reset.
    :param parameters: The parameters which should be copied to the model.
    """
    reset_params(model.parameters(), parameters)


def reset_params(model_parameters: Iterable[Tensor], parameters: List[Tensor]):
    """
    Resets the parameters of a model to the supplied parameters.

    :param model_parameters: The parameters of the model which should be reset.
    :param parameters: The parameters which should be copied to the model.
    """
    with no_grad():
        for model_parameter, parameter in zip(model_parameters, parameters):
            model_parameter.data = parameter.clone()
