from typing import Optional

from torch import cuda, device
from torch.nn import Module
from torch_geometric.nn.models import GAT, GCN, GIN

hidden_layers_default = 4
channels = 16
heads = 8
dropout = 0.5

torch_device = device("cuda" if cuda.is_available() else "cpu")


def create_model(
    dataset, model_type: str, jk: Optional[str] = None, hidden_layers: int = hidden_layers_default
) -> Module:
    """
    Initializes a GNN model for the specified dataset.
    :param dataset: the dataset for which this model will be used.
    :param model_type: the name of the GNN: GCN, GIN or GAT.
    :param jk: either None, "max", "cat". Passed to GCN/GIN/GAT constructor.
    :param hidden_layers: the count of hidden layers.
    """
    if model_type == "GCN":
        # old model:
        # model = gcn.GCNWithJK(dataset, hidden_layers, channels).to(device)
        return GCN(
            dataset.num_features,
            hidden_channels=channels,
            dropout=dropout,
            num_layers=hidden_layers,
            out_channels=dataset.num_classes,
            jk=jk,
        ).to(torch_device)
    elif model_type == "GIN":
        return GIN(
            in_channels=dataset.num_features,
            hidden_channels=channels,
            dropout=dropout,
            num_layers=hidden_layers,
            out_channels=dataset.num_classes,
            heads=heads,
            jk=jk,
        ).to(torch_device)
    elif model_type == "GAT":
        return GAT(
            in_channels=dataset.num_features,
            hidden_channels=channels * heads,
            dropout=dropout,
            num_layers=hidden_layers,
            out_channels=dataset.num_classes,
            heads=heads,
            jk=jk,
        ).to(torch_device)

    raise ValueError(f"model type {model_type} is invalid")
