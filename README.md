<div align="center">

<h1> Visualizing Torch Landscape

![python-3.10](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch-1.13.1](https://img.shields.io/badge/torch-1.13.1%2B-orange)
![plotly-1.13.1](https://img.shields.io/badge/plotly-5.13.1%2B-9cf)
![release-version](https://img.shields.io/badge/release-0.1-green)
![license](https://img.shields.io/badge/license-GPL%202-red)
_________________________
</div>


# Installation
```bash
# 1. Create new environment for the the project:
conda create -n pyq python=3.10
# 2. Activate the new environment:
conda activate py310
# 3. Install cudatoolkit 11.3 and PyTorch dependencies:
conda install pytorch cudatoolkit=11.3 -c pytorch
# 4. Clone Visualizing Torch Landscape repository:
git clone https://gitlab.cs.univie.ac.at/samirm97cs/visualizing_torch_landscape.git && cd visualizing_torch_landscape
# 5. Install visualizing_torch_landscape:
pip install -e .
```

# Get Started
```bash
from torch_landscape.directions import create_random_directions
from torch_landscape.landscape import calculate_surface_landscape
from torch_landscape.visualize import visualize


# Define or load your trained model
...

# Define evaluation function that has arguments ONLY two model, and iterative data.
# Example
def evaluate_func(model, test_loader):
    device = [*model.parameters()][0].device
    model.eval()
    test_loss = 0
    with no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nll_loss(output, target, reduction="sum").item()
    test_loss /= len(test_loader.dataset)
    return test_loss

file_path = "./my_landscape_surface"
rand_directions = create_random_directions(model=model)
landscape = calculate_surface_landscape(model=model,
                                        directions=rand_directions,
                                        data_loader=data,
                                        compute_fun=evaluate,
                                        num_data_point=10, )
visualize(landscape_dictionary=landscape, file_path=file_path)
```

![mnist_loss_landscape](./assets/mnist_landscape_surface.png)


# Development

## Formatting
- Use `black -l 120` to reformat a file.
- Use `flake8` to check if a source file is well formatted.
- To reorder the import statements, use `isort`.

## Unit tests
Run in directory test:
- Run code coverage data collection: `python -m coverage run -m unittest`.
- Create html report with code coverage: `python -m coverage html`.