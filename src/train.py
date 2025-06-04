import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
import numpy as np
from tqdm import tqdm

import os
import itertools
import json
import time
import random

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MR_Loss(weights, y):
    weights = weights.unsqueeze(1)
    return (-weights * y).mean()

def train_step(model, device, loader, optimizer, loss_fn, epoch, params={}, mode="train"):
    model.train()
    gradient_clipping = params.get("gradient_clipping", None)

    is_train = mode == "train"
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()    

            loss = loss_fn(x, y, model, epoch)      
            
            if is_train:
                loss.backward()

                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

                
                optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


# def default_dataset_fn(params):
#     return (train_dataset, test_dataset) 

def grid_to_sets(grids):
    if not isinstance(grids, list):
        grids = [grids]  # Wrap single grid in a list

    all_combinations = []
    for grid in grids:
        keys = grid.keys()
        combinations = [dict(zip(keys, values)) for values in itertools.product(*grid.values())]
        all_combinations.extend(combinations)

    return all_combinations

def model_file_name(params):
    model_name = ",".join(
            f"{key}={val}" for key, val in params.items() if val is not None
        ) + ".pth"
    return model_name


def label_from_params(params, experiments):
    nested = [list(v["params"].items()) for k, v in experiments.items()]

    ps = {}
    for k,v in [item for sublist in nested for item in sublist]:
        if k in ps:
            ps[k].add(v)
        else:
            ps[k] = {v}

    params_to_use = []

    for k,v in ps.items():
        if len(v) > 1:
            params_to_use.append(k)

    if "label" in params:
        label = params["label"] + " "
    else:
        label = ""

    return label + " ".join(
                f"{key}={val}" for key, val in params.items() if key in params_to_use and key != "label"
            )


def name_experiments(experiments):
    return {label_from_params(v["params"], experiments): v for k, v in experiments.items()}

def get_experiment_model(model_fn, experiment_path, params_grid, loss_fn_fn, dataset_fn, device="cuda"):
    experiments = {}

    train_result_path = os.path.join(experiment_path, "train_results.csv")
    os.makedirs(os.path.dirname(train_result_path), exist_ok=True)
    if os.path.exists(train_result_path):
        with open(train_result_path, 'r') as file:
            train_results = json.load(file)
    else:
        train_results = {}

    param_sets = grid_to_sets(params_grid)

    for params in param_sets:
        # name = f"{}"
        epochs = params.get("epochs", 10)
        checkpoint_freq = params.get("checkpoint_freq", epochs)
        
        paths = {}
        for epoch in range(1, epochs + 1):
            if epoch % checkpoint_freq == 0 or epochs==epoch: 
                name = model_file_name(params | {"checkpoint": epoch})
                model_path = os.path.join(experiment_path, name)
                paths[epoch] = model_path

        load_checkpoint = params.get("checkpoint", None)
        print(load_checkpoint)
        if load_checkpoint is not None:
            paths = {load_checkpoint: paths[load_checkpoint]}
            del params["checkpoint"]
            print(paths)
        print([os.path.exists(p) for p in paths.values()])
            
      
       
        if all([os.path.exists(p) for p in paths.values()]):
            for checkpoint, path in paths.items():
                print("Loading", path)
                checkpoint_params = params | {"checkpoint": checkpoint}
                name = model_file_name(checkpoint_params)
                model_path = os.path.join(experiment_path, name)
                model = model_fn(params)
                model.load_state_dict(torch.load(model_path, map_location=device))
                experiments[name] = {"model": model, "params": checkpoint_params, "results": {}, "train_results": train_results[model_path]}
        else:
            print("Training", model_file_name(params))
            if "seed" in params:
                set_seed(params["seed"])
            model = model_fn(params)
            loss_fn = loss_fn_fn(params)
            train_losses = []
            val_losses = []
            lr = params.get("learning_rate", 0.001)
            weight_decay = params.get("weight_decay", 1e-5)
            
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            train_time = 0
            pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="ep")

            train_loader, val_loader = dataset_fn(params)

            for epoch in pbar:
                start_time = time.time()
                train_loss  = train_step(model, device, train_loader, optimizer, loss_fn, epoch, params,"train")
                train_time += time.time() - start_time

                val_loss = train_step(model, device, val_loader, optimizer, loss_fn, epoch, params, "eval")
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                pbar.set_postfix(train_loss=f"{train_loss:.5f}",val_loss=f"{val_loss:.10f}")

                if epoch % checkpoint_freq == 0 or epoch == epochs:
                    checkpoint_params = params | {"checkpoint": epoch}
                    name = model_file_name(checkpoint_params)
                   
                    model_path = os.path.join(experiment_path, name)
                    print("Saving", name)
                    train_results[model_path] = {"train_time": train_time, "train_losses": train_losses.copy(), "val_losses": val_losses.copy()}
                    torch.save(model.state_dict(), model_path)

                    checkpoint_model = model_fn(params)
                    checkpoint_model.load_state_dict(torch.load(model_path, map_location=device))
                    experiments[name] = {"model": checkpoint_model, "params": checkpoint_params, "results": {}, "train_results": train_results[model_path]}
                    with open(train_result_path, 'w') as file: 
                        json.dump(train_results, file, indent=4)
    experiments = name_experiments(experiments)
    return experiments