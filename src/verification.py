from tqdm import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np


def verified_bounds(model, x, eps, method="CROWN-IBP"):
    perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)
    x_perturbed = BoundedTensor(x, perturbation)
    lb, ub = model.compute_bounds(x=(x_perturbed,), method=method)
    return lb, ub

def predict(model, scaler, X_test):
    return scaler.inverse_transform(model(X_test).cpu().detach().numpy())

def get_bounds(model, X_test, scaler, epsilon, method="backward"):
    perturbation = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    x_perturbed = BoundedTensor(X_test, perturbation)
    lb, ub = model.compute_bounds(x=(x_perturbed,), method=method)

    lb = scaler.inverse_transform(lb.cpu().detach().numpy())
    ub = scaler.inverse_transform(ub.cpu().detach().numpy())

    return lb, ub    

def calculate_mean_bound_width(model, X_test, scaler, epsilons, method="backward"):
    mean_bound_widths = []
    
    for epsilon in tqdm(epsilons):
        # Get the bounds for the entire dataset
        lbs, ubs = get_bounds(model, X_test, scaler, epsilon, method)
        
        # Calculate the width of each bound
        bound_widths = ubs - lbs
        
        # Calculate the mean bound width for this epsilon
        mean_width = np.mean(bound_widths)
        mean_bound_widths.append(mean_width)
    
    return mean_bound_widths


def check_spec_satisfaction(f_clean, f_lower, f_upper, delta):
    """
    Check whether the spec is satisfied for a given input.
    """
    max_deviation = max(abs(f_upper - f_clean), abs(f_lower - f_clean))
    return max_deviation <= delta  

def calculate_spec_satisfaction(model, X_test, y_test, scaler, epsilons, delta, method="backward"):
    """
    Calculate the percentage of inputs satisfying the spec for each epsilon.
    """
    spec_satisfaction_rates = {epsilon: 0 for epsilon in epsilons}
    
    for epsilon in tqdm(epsilons):
        num_spec_satisfied = 0
        total = len(X_test)
        
        # Get the CROWN bounds for the entire dataset at once
        f_clean_batch = predict(model, scaler, X_test)
        f_lower_batch, f_upper_batch = get_bounds(model, X_test, scaler, epsilon, method)  # CROWN-IBP bounds for the whole batch
        
        # Loop over each example in the batch to check spec satisfaction
        for i in range(total):
            f_clean = f_clean_batch[i]  # Clean prediction for current input
            f_lower = f_lower_batch[i]  # Lower bound for current input
            f_upper = f_upper_batch[i]  # Upper bound for current input
            y_true = y_test[i]  # True label for current input
            
            if check_spec_satisfaction(f_clean, f_lower, f_upper, delta):
                num_spec_satisfied += 1
        
        # Calculate the percentage of inputs satisfying the spec
        spec_satisfaction_rate = (num_spec_satisfied / total) * 100
        spec_satisfaction_rates[epsilon] = spec_satisfaction_rate

    return spec_satisfaction_rates


def check_robust_accuracy(f_lower, f_upper, y_true, delta):
    """
    Check whether the robust accuracy is satisfied for a given input.
    """
    max_error = max(abs(f_upper - y_true), abs(f_lower - y_true))
    return max_error <= delta  # Returns True if robust accuracy is satisfied

def calculate_robust_accuracy(model, X_test, y_test, scaler, epsilons, delta, method="backward"):
    """
    Calculate the percentage of inputs satisfying the robust accuracy spec for each epsilon.
    """
    robust_accuracy_rates = {epsilon: 0 for epsilon in epsilons}

    # y_test = y_test.detach().numpy()
    y_test = scaler.inverse_transform(y_test.detach().numpy())


    for epsilon in tqdm(epsilons):
        num_robust_accurate = 0
        total = len(X_test)
        
        # Get the CROWN bounds for the entire dataset at once
        f_clean_batch = predict(model, scaler, X_test)  # model's predictions for the whole batch
        f_lower_batch, f_upper_batch = get_bounds(model, X_test, scaler, epsilon, method)  # CROWN-IBP bounds for the whole batch

        # Loop over each example in the batch to check robust accuracy
        for i in range(total):
            f_clean = f_clean_batch[i]  # Clean prediction for current input
            f_lower = f_lower_batch[i]  # Lower bound for current input
            f_upper = f_upper_batch[i]  # Upper bound for current input
            y_true = y_test[i]  # True label for current input
            
            if check_robust_accuracy(f_lower, f_upper, y_true, delta):
                num_robust_accurate += 1

        # Calculate the percentage of inputs satisfying the robust accuracy spec
        robust_accuracy_rate = (num_robust_accurate / total) * 100
        robust_accuracy_rates[epsilon] = robust_accuracy_rate

    return robust_accuracy_rates

def calculate_mean_mse(model, X_test, scaler, targets):
    prediction = predict(model, scaler, X_test)
    mse = np.mean((targets - prediction)**2)
    return mse