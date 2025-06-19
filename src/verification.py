from tqdm import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np
import torch


import pickle
from IPython import get_ipython
import os 
import json


def run_abcrown_from_dict_ipython(args_dict, debug_file):
    cmd = "python alpha-beta-CROWN/complete_verifier/abcrown.py"

    output_path = args_dict["output_file"]
    results_path = args_dict["results_file"]
    cex_path = args_dict["cex_path"]


    if os.path.isfile(output_path):
        os.remove(output_path)
    if os.path.isfile(results_path):
        os.remove(results_path)

    with open(cex_path, "w") as f:
        json.dump( {"x": [], "adv_output": []}, f, indent=2)

    for key, value in args_dict.items():
        cmd += f" --{key} {value}"

    cmd = f"{cmd} > {debug_file}"



    get_ipython().system(cmd)


def pgd_attack(model, x_init, y, eps, loss_fn, n_steps=20, step_size=0.1, device='cuda:0'):
    # 0/0
    model.eval()
    x0 = x_init.to(device)
    y0 = y.to(device)
    eps = eps.to(device) if torch.is_tensor(eps) else eps

    # start from the clean example, one tensor that always requires grad
    adv = x0.clone().detach().requires_grad_(True)
    best_loss = torch.full((x0.size(0),), -float('inf'), device=device)
    with torch.enable_grad():
        for _ in range(n_steps):
            # zero the gradient from the previous step
            if adv.grad is not None:
                adv.grad.zero_()
            # forward + objective (we do -loss because we want the *worst* point)
            logits = model(adv)
            obj = -loss_fn(logits, y0)            # this must be a Tensor, not a float

            # keep the best‐so‐far point per example
            mask = (obj >= best_loss).view(-1, *([1] * (adv.dim()-1)))
            best_loss = torch.where(obj >= best_loss, obj.detach(), best_loss)
            # you can store best‐so‐far inputs if needed, same pattern

            # compute gradient w.r.t. adv
            obj.sum().backward()                  # builds d(obj)/d(adv)
            # PGD step: modify adv.data in‐place, then clamp
            adv.data = (adv.data + step_size * adv.grad.data.sign())
            adv.data = adv.data.clamp(x0 - eps, x0 + eps)

    # at the end, adv contains your adversarial examples
    return adv.detach()

def pgd_attack_loss(x, y):
    loss = ((1 + y).prod(1) * x).sum(1)
    return loss

def bound_model(model, x, eps, method, batch_size=1, device="cuda"):
    upper_batches = []
    lower_batches = []
    x = x.to(device)
    N = x.shape[0]
    
    if method in ["a-CROWN", "b-CROWN", "ab-CROWN", "CROWN-Optimized"]:
        bound_args = model.bound_opts["optimize_bound_args"]
        bound_args["enable_alpha_crown"] = method in ["a-CROWN", "ab-CROWN"]
        bound_args["enable_beta_crown"] = method in ["b-CROWN", "ab-CROWN"]
        method = "CROWN-Optimized"
    if method == "CROWN-Optimized":
        model.get_split_nodes()


    for i in tqdm(range(0, N, batch_size)):
        x_batch = x[i: i+batch_size]
        eps_vector = torch.tensor(eps, dtype=torch.float32, device=device)  # One ε per asset
        eps_tensor = eps_vector.view(1, 1, -1).expand_as(x_batch) 

        perturbation = PerturbationLpNorm(norm=np.inf, eps=eps_tensor)
        x_perturbed = BoundedTensor(x_batch, perturbation)
        try:
            lb, ub = model.compute_bounds(x=(x_perturbed,), method=method)
            lb = lb.cpu()
            ub = ub.cpu()
        except Exception as e:
            lb, ub = torch.tensor([torch.nan]*4).unsqueeze(0), torch.tensor([torch.nan]*4).unsqueeze(0)
             
        lower_batches.append(lb)
        upper_batches.append(ub)

    lower_all = torch.cat(lower_batches, 0).nan_to_num(nan=0).clamp_min(0)
    upper_all = torch.cat(upper_batches, 0).nan_to_num(nan=1).clamp_max(1)

    return lower_all, upper_all


# def verified_bounds(model, x, eps, method="CROWN-IBP"):
#     perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)
#     x_perturbed = BoundedTensor(x, perturbation)
#     lb, ub = model.compute_bounds(x=(x_perturbed,), method=method)
#     return lb, ub

# def predict(model, scaler, X_test):
#     return scaler.inverse_transform(model(X_test).cpu().detach().numpy())

# def get_bounds(model, X_test, scaler, epsilon, method="backward"):
#     perturbation = PerturbationLpNorm(norm=np.inf, eps=epsilon)
#     x_perturbed = BoundedTensor(X_test, perturbation)
#     lb, ub = model.compute_bounds(x=(x_perturbed,), method=method)

#     lb = scaler.inverse_transform(lb.cpu().detach().numpy())
#     ub = scaler.inverse_transform(ub.cpu().detach().numpy())

#     return lb, ub    

# def calculate_mean_bound_width(model, X_test, scaler, epsilons, method="backward"):
#     mean_bound_widths = []
    
#     for epsilon in tqdm(epsilons):
#         # Get the bounds for the entire dataset
#         lbs, ubs = get_bounds(model, X_test, scaler, epsilon, method)
        
#         # Calculate the width of each bound
#         bound_widths = ubs - lbs
        
#         # Calculate the mean bound width for this epsilon
#         mean_width = np.mean(bound_widths)
#         mean_bound_widths.append(mean_width)
    
#     return mean_bound_widths


# def check_spec_satisfaction(f_clean, f_lower, f_upper, delta):
#     """
#     Check whether the spec is satisfied for a given input.
#     """
#     max_deviation = max(abs(f_upper - f_clean), abs(f_lower - f_clean))
#     return max_deviation <= delta  

# def calculate_spec_satisfaction(model, X_test, y_test, scaler, epsilons, delta, method="backward"):
#     """
#     Calculate the percentage of inputs satisfying the spec for each epsilon.
#     """
#     spec_satisfaction_rates = {epsilon: 0 for epsilon in epsilons}
    
#     for epsilon in tqdm(epsilons):
#         num_spec_satisfied = 0
#         total = len(X_test)
        
#         # Get the CROWN bounds for the entire dataset at once
#         f_clean_batch = predict(model, scaler, X_test)
#         f_lower_batch, f_upper_batch = get_bounds(model, X_test, scaler, epsilon, method)  # CROWN-IBP bounds for the whole batch
        
#         # Loop over each example in the batch to check spec satisfaction
#         for i in range(total):
#             f_clean = f_clean_batch[i]  # Clean prediction for current input
#             f_lower = f_lower_batch[i]  # Lower bound for current input
#             f_upper = f_upper_batch[i]  # Upper bound for current input
#             y_true = y_test[i]  # True label for current input
            
#             if check_spec_satisfaction(f_clean, f_lower, f_upper, delta):
#                 num_spec_satisfied += 1
        
#         # Calculate the percentage of inputs satisfying the spec
#         spec_satisfaction_rate = (num_spec_satisfied / total) * 100
#         spec_satisfaction_rates[epsilon] = spec_satisfaction_rate

#     return spec_satisfaction_rates


# def check_robust_accuracy(f_lower, f_upper, y_true, delta):
#     """
#     Check whether the robust accuracy is satisfied for a given input.
#     """
#     max_error = max(abs(f_upper - y_true), abs(f_lower - y_true))
#     return max_error <= delta  # Returns True if robust accuracy is satisfied

# def calculate_robust_accuracy(model, X_test, y_test, scaler, epsilons, delta, method="backward"):
#     """
#     Calculate the percentage of inputs satisfying the robust accuracy spec for each epsilon.
#     """
#     robust_accuracy_rates = {epsilon: 0 for epsilon in epsilons}

#     # y_test = y_test.detach().numpy()
#     y_test = scaler.inverse_transform(y_test.detach().numpy())


#     for epsilon in tqdm(epsilons):
#         num_robust_accurate = 0
#         total = len(X_test)
        
#         # Get the CROWN bounds for the entire dataset at once
#         f_clean_batch = predict(model, scaler, X_test)  # model's predictions for the whole batch
#         f_lower_batch, f_upper_batch = get_bounds(model, X_test, scaler, epsilon, method)  # CROWN-IBP bounds for the whole batch

#         # Loop over each example in the batch to check robust accuracy
#         for i in range(total):
#             f_clean = f_clean_batch[i]  # Clean prediction for current input
#             f_lower = f_lower_batch[i]  # Lower bound for current input
#             f_upper = f_upper_batch[i]  # Upper bound for current input
#             y_true = y_test[i]  # True label for current input
            
#             if check_robust_accuracy(f_lower, f_upper, y_true, delta):
#                 num_robust_accurate += 1

#         # Calculate the percentage of inputs satisfying the robust accuracy spec
#         robust_accuracy_rate = (num_robust_accurate / total) * 100
#         robust_accuracy_rates[epsilon] = robust_accuracy_rate

#     return robust_accuracy_rates

# def calculate_mean_mse(model, X_test, scaler, targets):
#     prediction = predict(model, scaler, X_test)
#     mse = np.mean((targets - prediction)**2)
#     return mse