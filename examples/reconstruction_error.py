from logging import DEBUG, basicConfig
from time import perf_counter
from typing import List

from numpy import mean, round, std
from pandas import DataFrame
from torch import Tensor, cuda, device, no_grad
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAT, GCN, GIN
from tqdm.auto import tqdm

from torch_landscape.directions import LearnableDirections, PcaDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.trajectory import TrajectoryCalculator
from torch_landscape.utils import clone_parameters
from torch_landscape.visualize import VisualizationData, Plotly2dVisualization
from torch_landscape.visualize_options import VisualizationOptions

torch_device = device("cpu")

channels = 8
dropout = 0.5
jk = "max"
heads = 8


def create_model(dataset, model_type: str, num_layers: int) -> Module:
    """
    Initializes a GNN model for the specified dataset.
    :param dataset: the dataset for which this model will be used.
    :param model_type: the name of the GNN: GCN, GIN or GAT.
    :param num_layers: the count of hidden layers.
    """
    if model_type == "GCN":
        return GCN(
            dataset.num_features,
            hidden_channels=channels,
            dropout=dropout,
            num_layers=num_layers,
            out_channels=dataset.num_classes,
            jk=jk,
        ).to(torch_device)
    elif model_type == "GIN":
        return GIN(
            in_channels=dataset.num_features,
            hidden_channels=channels,
            dropout=dropout,
            num_layers=num_layers,
            out_channels=dataset.num_classes,
            heads=heads,
            jk=jk,
        ).to(torch_device)
    elif model_type == "GAT":
        return GAT(
            in_channels=dataset.num_features,
            hidden_channels=channels * heads,
            dropout=dropout,
            num_layers=num_layers,
            out_channels=dataset.num_classes,
            heads=heads,
            jk=jk,
        ).to(torch_device)

    raise ValueError(f"model type {model_type} is invalid")


def train(model: Module, optimizer: Optimizer, epochs: int) -> List[List[Tensor]]:
    intermediate_results = []

    model.train()

    # Insert starting point
    with no_grad():
        intermediate_results.append(clone_parameters(model.parameters()))

    # Train model
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        progress_bar.set_description("Loss: {}".format(loss.item()))
        if epoch % 10 == 0:
            intermediate_results.append(clone_parameters(model.parameters()))
        loss.backward()
        optimizer.step()

    # Add final result.
    intermediate_results.append(clone_parameters(model.parameters()))

    return intermediate_results


def evaluate(model_) -> float:
    with no_grad():
        out = model_(x, edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        return loss.item()


if __name__ == "__main__":
    basicConfig()
    PcaDirections.logger.setLevel(DEBUG)
    LearnableDirections.logger.setLevel(DEBUG)
    torch_device = device("cpu")
    cpu_device = device("cpu")

    results = []
    for dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        # Load dataset
        dataset = Planetoid(root=f"/tmp/{dataset_name}", name=dataset_name)
        criterion = CrossEntropyLoss()

        epochs = 200
        repetitions = 10

        for model_type in ["GCN", "GIN", "GAT"]:
            for layer_count in [2, 3, 4, 5]:
                pca_errors = []
                ae_errors = []
                pca_runtimes = []
                ae_runtimes = []
                data = dataset[0].to(torch_device)
                x, edge_index, batch = data.x, data.edge_index, data.batch

                model = create_model(dataset, model_type, layer_count)
                optimizer = Adam(model.parameters(), lr=0.01)
                intermediate_parameters = train(model, optimizer, epochs)
                optimized_parameters = [*model.parameters()]

                model = model.to(cpu_device)
                data = data.to(cpu_device)
                x, edge_index = data.x.to(cpu_device), data.edge_index.to(cpu_device)
                intermediate_parameters = [[p.to(cpu_device) for p in p_lst] for p_lst in intermediate_parameters]
                optimized_parameters = [p.to(cpu_device) for p in optimized_parameters]

                for i in range(repetitions):
                    try:
                        pca_start = perf_counter()
                        b1, b2 = PcaDirections(optimized_parameters, intermediate_parameters, cpu_device).calculate_directions()
                        pca_end = perf_counter()
                        pca_runtime = pca_end - pca_start
                        pca_runtimes.append(pca_runtime)

                        pca_error = TrajectoryCalculator.reconstruction_error_mean(b1, b2, optimized_parameters, intermediate_parameters)
                        pca_errors.append(pca_error)
                    except ValueError as e:
                        pca_error = -10
                        pca_errors.append(-10)
                        pca_runtimes.append(-10)

                    ae_start = perf_counter()
                    b1, b2 = LearnableDirections(optimized_parameters, intermediate_parameters, training_epochs=1000, learnable_model_device=cpu_device).calculate_directions()
                    ae_end = perf_counter()
                    ae_runtime = ae_end - ae_start
                    ae_runtimes.append(ae_runtime)
                    ae_error = TrajectoryCalculator.reconstruction_error_mean(b1, b2, optimized_parameters, intermediate_parameters)
                    ae_errors.append(ae_error)
                    print(f"Dataset: {dataset_name}, Model: {model_type}, Layers: {layer_count}, PCA error: {pca_error}, AE error: {ae_error}")

                results.append(
                    [
                        dataset_name,
                        model_type,
                        layer_count,
                        round(mean(pca_errors), 2),
                        round(std(pca_errors), 1),
                        round(mean(pca_runtimes), 8),
                        round(std(pca_runtimes), 4),
                        round(mean(ae_errors), 2),
                        round(std(ae_errors), 1),
                        round(mean(ae_runtimes), 8),
                        round(std(ae_runtimes), 4),
                    ]
                )

    df = DataFrame(
        results,
        columns=[
            "dataset_name",
            "model_type",
            "layer_count",
            "pca_error_mean",
            "pca_error_std",
            "pca_runtime_mean",
            "pca_runtime_std",
            "ae_error_mean",
            "ae_error_std",
            "ae_runtime_mean",
            "ae_runtime_std",
        ],
    )
    df.to_csv("reconstruction_error.csv", index=False)
