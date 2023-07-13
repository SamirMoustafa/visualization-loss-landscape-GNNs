# This file visualizes the loss landscape of Cora data in Planetoid dataset for different count of layers.
# One variant is with jumping knowledge and one without (also called skip connection).
# Code based on: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
import os.path
from argparse import ArgumentParser

import numpy
import pandas
from common import create_model
from torch import cuda, device, no_grad
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

from torch_landscape.directions import RandomDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.utils import clone_parameters, reset_parameters, seed_everything
from torch_landscape.visualize import Plotly3dVisualization, VisualizationData
from torch_landscape.visualize_options import VisualizationOptions

torch_device = device("cuda" if cuda.is_available() else "cpu")
criterion = CrossEntropyLoss()


def train(model: Module, optimizer: Optimizer):
    model.train()
    intermediate_parameters = []
    for epoch in tqdm(range(200)):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        intermediate_parameters.append((clone_parameters(model.parameters()), loss.item()))
        loss.backward()
        optimizer.step()

    return intermediate_parameters


def evaluate(model):
    with no_grad():
        out = model(x, edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
    return loss


def visualize_model(model: Module, file_path: str, title: str):
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train(model, optimizer)

    model.eval()
    acc = calculate_accuracy(model)
    model.train()

    directions = RandomDirections([*model.parameters()]).calculate_directions()

    options = VisualizationOptions(num_points=20, show_title=False, show_axes_labels=False)
    landscape_calculator = LinearLandscapeCalculator(model.parameters(), directions, options=options)
    landscape = landscape_calculator.calculate_loss_surface_data_model(model, lambda: evaluate(model))
    plot_title = title + f" - Accuracy: {acc:.3f}, Loss: {evaluate(model).item():.3f}"
    file_path += f"_accuracy{acc:.3f}_loss{evaluate(model).item():.3f}"
    Plotly3dVisualization(options).plot(VisualizationData(landscape), file_path, plot_title, "pdf")

    return acc


def calculate_accuracy(model):
    predictions = model(x, edge_index).argmax(dim=1)
    correct = (predictions[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f"Accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Jumping knowledge comparison visualization",
        description="Creates visualizations of models with and without jumping knowledge connections.",
    )
    parser.add_argument("-d", "--dataset", default="Cora")
    parser.add_argument("-m", "--model_type", default="GCN")
    output_folder = "results"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    args = parser.parse_args()
    dataset_name = args.dataset
    model_type = args.model_type

    print(f"Model type: {model_type}, Dataset: {dataset_name}")

    dataset = Planetoid(root="data", name=dataset_name)
    data = dataset[0].to(torch_device)

    # x is the feature matrix with shape [node count, feature count] - one row per node.
    # edge_index: edges in COO format, shape [2, num_edges].
    x, edge_index, batch = data.x, data.edge_index, data.batch

    results = []
    seed = 12345
    for layer_count in [2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 20, 24]:
        print(f"Layer count: {layer_count}")
        seed_everything(seed)
        model_jk = create_model(dataset, model_type, "max", layer_count).to(torch_device)

        parameters = clone_parameters(model_jk.parameters(), False)
        seed_everything(seed)
        model = create_model(dataset, model_type, None, layer_count)

        # Check if parameters of non-jk model and jk model are equal (should be equal because of seed).
        assert numpy.all((p1 - p2).norm() == 0.0 for p1, p2 in zip(model_jk.parameters(), model.parameters()))

        acc_withjk = visualize_model(
            model_jk,
            f"./{output_folder}/{dataset_name}_{model_type}_withjk_{layer_count}_layers",
            f"{model_type}-JK - {layer_count} layers",
        )

        acc_withoutjk = visualize_model(
            model,
            f"./{output_folder}/{dataset_name}_{model_type}_withoutjk_{layer_count}_layers",
            f"{model_type} - {layer_count} layers",
        )

        results.append([layer_count, acc_withjk, acc_withoutjk])

    df = pandas.DataFrame(results, columns=["layers", "accuracy_withjk", "accuracy_withoutjk"])
    df.to_csv(f"./{output_folder}/{dataset_name}-{model_type}-jk-comparison.csv", index=False)
