from __future__ import division

import time
import os
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
from torch_landscape.directions import LearnableDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.trajectory import TrajectoryCalculator
from torch_landscape.utils import clone_parameters
from torch_landscape.visualize import VisualizationOptions, VisualizationData, Plotly2dVisualization

import utils as ut
from kfac_torch import psgd


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path_runs = "runs"

def run(
    dataset, 
    model, 
    str_optimizer, 
    str_preconditioner, 
    runs, 
    epochs, 
    lr, 
    weight_decay, 
    early_stopping,  
    logger, 
    momentum,
    eps,
    update_freq,
    gamma,
    alpha,
    hyperparam,
    precondition_at
    ):
    logger_string = logger
    if logger is not None:
        if hyperparam:
            logger += f"-{hyperparam}{eval(hyperparam)}"
        if precondition_at:
            logger += f"-precondition-at{precondition_at}"
        path_logger = os.path.join(path_runs, logger)
        print(f"path logger: {path_logger}")

        ut.empty_dir(path_logger)
        logger = SummaryWriter(log_dir=os.path.join(path_runs, logger)) if logger is not None else None

    val_losses, accs, durations = [], [], []
    torch.manual_seed(42)
    for i_run in range(runs):
        data = dataset[0]
        data = data.to(device)

        model.to(device).reset_parameters()
        if str_preconditioner == 'KFAC':

            preconditioner = psgd.KFAC(
                model, 
                eps, 
                sua=False, 
                pi=False, 
                update_freq=update_freq,
                alpha=alpha if alpha is not None else 1.,
                constraint_norm=False
            )
        else: 
            preconditioner = None

        if str_optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif str_optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=momentum,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []
        intermediate_parameters = []
        last_loss = None

        for epoch in range(1, epochs + 1):
            lam = (float(epoch)/float(epochs))**gamma if gamma is not None else 0.

            if epoch >= precondition_at:
                train(model, optimizer, data, preconditioner, lam)
            else:
                train(model, optimizer, data, None, lam)

            eval_info = evaluate(model, data)
            eval_info['epoch'] = int(epoch)
            eval_info['run'] = int(i_run+1)
            eval_info['time'] = time.perf_counter() - t_start
            eval_info['eps'] = eps
            eval_info['update-freq'] = update_freq

            intermediate_parameters.append(clone_parameters(model.parameters()))

            if gamma is not None:
                eval_info['gamma'] = gamma
            
            if alpha is not None:
                eval_info['alpha'] = alpha

            if logger is not None:
                for k, v in eval_info.items():
                    logger.add_scalar(k, v, global_step=epoch)
                    
            if eval_info['val loss'] < best_val_loss:
                best_val_loss = eval_info['val loss']
                test_acc = eval_info['test acc']

            val_loss_history.append(eval_info['val loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val loss'] > tmp.mean().item():
                    break
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        file_path = path_logger + "-run{0}".format(i_run)

        eval_result_train = evaluate(model, data, False)
        eval_result = evaluate(model, data, True)
        plot_title = logger_string + f" Loss: {eval_result_train['train loss']:.3f}, Accuracy: {eval_result['test acc']:.3f}"

        options = VisualizationOptions(num_points=50)
        directions = LearnableDirections([*model.parameters()], intermediate_parameters).calculate_directions()
        trajectory = TrajectoryCalculator([*model.parameters()], directions).project_disregard_z(intermediate_parameters)
        trajectory.set_range_to_fit_trajectory(options)

        landscape_calculator = LinearLandscapeCalculator(model.parameters(), directions, options=options)
        landscape = landscape_calculator.calculate_loss_surface_data_model(model, lambda: evaluate(model, data, False)["train loss"])
        Plotly2dVisualization(options).plot(VisualizationData(landscape, trajectory), file_path, plot_title, "pdf")

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
    
    if logger is not None:
        logger.close()
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    print('Val Loss: {:.4f}, Test Accuracy: {:.2f} ± {:.2f}, Duration: {:.3f} \n'.
          format(loss.mean().item(),
                 100*acc.mean().item(),
                 100*acc.std().item(),
                 duration.mean().item()))

def train(model, optimizer, data, preconditioner=None, lam=0.):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False
    
    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])
    
    loss.backward(retain_graph=True)
    if preconditioner:
        preconditioner.step(lam=lam)
    optimizer.step()

def evaluate(model, data, eval_mode=True):
    if eval_mode:
        model.eval()
    else:
        model.train()
        
    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc

    return outs
