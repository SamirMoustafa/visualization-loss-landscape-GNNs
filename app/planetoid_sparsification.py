#!/usr/bin/env python
import os
from argparse import ArgumentParser

from pandas import DataFrame
from torch import device, no_grad, where
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import (GraphSAINTEdgeSampler, GraphSAINTNodeSampler, GraphSAINTRandomWalkSampler,
                                    NeighborLoader)
from tqdm import tqdm

from common import channels, create_model, hidden_layers_default
from torch_landscape.directions import PcaDirections, LearnableDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.trajectory import TrajectoryCalculator
from torch_landscape.utils import clone_parameters, reset_parameters, seed_everything, move_parameters
from torch_landscape.visualize import Plotly2dVisualization, VisualizationData
from torch_landscape.visualize_options import VisualizationOptions


def train(model, optimizer, train_loader=None):
    model.train()
    intermediate_parameters = []
    progress_bar = tqdm(range(200))
    for epoch in progress_bar:
        optimizer.zero_grad()
        if train_loader is not None:
            loss = 0
            for batch in train_loader:
                out = model(batch.x.to(torch_device), batch.edge_index.to(torch_device))
                loss += criterion(out, batch.y)
        else:
            out = model(data.x.to(torch_device), data.edge_index.to(torch_device))
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Loss: {loss.item():.3f}")
        intermediate_parameters.append((clone_parameters(model.parameters()), loss.item()))

    return intermediate_parameters


def evaluate(model, train_loader=None):
    with no_grad():
        loss = 0
        if train_loader is not None:
            for batch in train_loader:
                out = model(batch.x.to(torch_device), batch.edge_index.to(torch_device))
                loss += criterion(out, batch.y)
        else:
            out = model(data.x.to(torch_device), data.edge_index.to(torch_device))
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        return loss.item()


def get_accuracy(model, mask):
    model.eval()
    predictions = model(data.x.to(torch_device), data.edge_index.to(torch_device)).argmax(dim=1)
    correct = (predictions[mask] == data.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc


def create_optimizer_result(parameters_with_loss, name, model, data_loader, use_pca=True):
    model.train()
    loss = evaluate(model, data_loader)
    # evaluate the loss without batches, such that it is comparable across different
    # experiments.
    loss_no_batches = evaluate(model, None)
    model.eval()
    loss_no_batches_eval = evaluate(model, None)

    accuracy = get_accuracy(model, data.test_mask)
    accuracy_train = get_accuracy(model, data.train_mask)
    accuracy_val = get_accuracy(model, data.val_mask)
    model.train()

    title = name + f", Loss: {loss_no_batches:.3f}, Accuracy: {accuracy:.3f}"
    file_path = os.path.join(output_folder, filename_prefix + name)

    #directions = PcaDirections([*model.parameters()], parameters_with_loss).calculate_directions()
    directions = LearnableDirections([*model.parameters()], parameters_with_loss[::10]).calculate_directions()
    directions = move_parameters(directions[0], "cpu"), move_parameters(directions[1], "cpu")

    options = VisualizationOptions(num_points=30, use_log_z=True)
    trajectory = TrajectoryCalculator([*model.parameters()], directions).project_with_loss(parameters_with_loss)
    trajectory.set_range_to_fit_trajectory(options)
    landscape_calculator = LinearLandscapeCalculator(model.parameters(), directions, options=options, n_jobs=4)
    landscape = landscape_calculator.calculate_loss_surface_data_model(model, lambda: evaluate(model, data_loader))
    Plotly2dVisualization(options).plot(VisualizationData(landscape, trajectory), file_path, title, "pdf")

    results_for_csv.append([name, loss, loss_no_batches, loss_no_batches_eval, accuracy, accuracy_train, accuracy_val])


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Sparsification visualizations",
        description="Creates visualizations of models with and without jumping knowledge connections.",
    )
    parser.add_argument("-d", "--dataset", default="Cora")
    parser.add_argument("-m", "--model_type", default="GCN")

    # sampling implementations do not support cuda, need to use cpu
    torch_device = device("cpu")

    args = parser.parse_args()
    dataset_name = args.dataset
    model_type = args.model_type

    print(f"Model type: {model_type}, Dataset: {dataset_name}")

    dataset = Planetoid(root="data", name=dataset_name)
    data = dataset[0].to(torch_device)
    criterion = CrossEntropyLoss()

    # x is the feature matrix with shape [node count, feature count] - one row per node.
    # edge_index: edges in COO format, shape [2, num_edges].
    x, edge_index, batch = data.x, data.edge_index, data.batch
    train_node_idx = where(data.train_mask)[0]

    seed_everything(12345)
    model = create_model(dataset, model_type, "max").to(torch_device)
    initial_parameters = clone_parameters(model.parameters())
    results_for_csv = []

    output_folder = "results"
    filename_prefix = f"{model_type}_{dataset_name}_{hidden_layers_default}hidden_{channels}channels_"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    subgraph = data.subgraph(data.train_mask)

    # Visualize without sparsification.
    seed_everything(12345)
    reset_parameters(model, initial_parameters)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    parameters_with_loss = train(model, optimizer)
    create_optimizer_result(parameters_with_loss, f"{model_type} without sparsification", model, None)

    # Sample 30 neighbors for each node for 2 iterations: [30] * 2
    # sample 5 neighbors for each node for 5 iterations: [5, 5]
    subgraph = data.subgraph(data.train_mask)
    seed_everything(12345)
    train_loader = NeighborLoader(subgraph, num_neighbors=[1, 2], shuffle=True, batch_size=128)

    reset_parameters(model, initial_parameters)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    parameters_with_loss = train(model, optimizer, train_loader)
    create_optimizer_result(parameters_with_loss, f"{model_type} with sparsification 1-2", model, train_loader)

    seed_everything(12345)
    train_loader = NeighborLoader(subgraph, num_neighbors=[2, 2], shuffle=True, batch_size=128)
    reset_parameters(model, initial_parameters)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    parameters_with_loss = train(model, optimizer, train_loader)
    create_optimizer_result(parameters_with_loss, f"{model_type} with sparsification 2-2", model, train_loader)

    seed_everything(12345)
    train_loader = NeighborLoader(subgraph, num_neighbors=[3, 2], shuffle=True, batch_size=128)
    reset_parameters(model, initial_parameters)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    parameters_with_loss = train(model, optimizer, train_loader)
    create_optimizer_result(parameters_with_loss, f"{model_type} with sparsification 3-2", model, train_loader)

    seed_everything(12345)
    train_loader = NeighborLoader(subgraph, num_neighbors=[1, 4], shuffle=True, batch_size=128)
    reset_parameters(model, initial_parameters)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    parameters_with_loss = train(model, optimizer, train_loader)
    create_optimizer_result(parameters_with_loss, f"{model_type} with sparsification 1-4", model, train_loader)

    if dataset_name != "PubMed":
        # Experiments with different node samples:
        seed_everything(12345)
        train_loader = GraphSAINTNodeSampler(subgraph, batch_size=512, num_steps=2)
        reset_parameters(model, initial_parameters)
        optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        parameters_with_loss = train(model, optimizer, train_loader)
        create_optimizer_result(parameters_with_loss, f"{model_type} with SAINTNodeSampler", model, train_loader)

        #
        seed_everything(12345)
        train_loader = GraphSAINTEdgeSampler(subgraph, batch_size=128)
        reset_parameters(model, initial_parameters)
        optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        parameters_with_loss = train(model, optimizer, train_loader)
        create_optimizer_result(parameters_with_loss, f"{model_type} with SAINTEdgeSampler", model, train_loader)

        #
        seed_everything(12345)
        train_loader = GraphSAINTRandomWalkSampler(subgraph, walk_length=15, batch_size=128)
        reset_parameters(model, initial_parameters)
        optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        parameters_with_loss = train(model, optimizer, train_loader)
        create_optimizer_result(parameters_with_loss, f"{model_type} with RandomWalkSampler, walk length 15", model, train_loader)

    df = DataFrame(
        results_for_csv,
        columns=[
            "name",
            "loss",
            "loss_no_batches",
            "loss_no_batches_eval",
            "accuracy",
            "accuracy_train",
            "accuracy_val",
        ],
    )
    df.to_csv(f"./{output_folder}/sparsification-{model_type}-{dataset_name}.csv", index=False)
