import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
import numpy as np
from tqdm import tqdm


def train_model(model, train_loader, test_loader, device, epochs=6, lr=0.001, patience=5):
    model = model.to(device)  # Move model to the selected device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)  # Move batch to device
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)  # Move test batch to device
                preds = model(xb)
                test_loss += loss_fn(preds, yb).item() * len(xb)

        test_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch+1:2d}, Test Loss: {test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

        scheduler.step(test_loss)
    return model


def lower_bound_verified(model, x, eps, method="CROWN-IBP"):
    perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)
    x_perturbed = BoundedTensor(x, perturbation)
    lb, _ = model.compute_bounds(x=(x_perturbed,), method=method)
    return lb


import os

def load_or_train_model(model, model_path, train_fn, device):
    # model_path = RESULT_PATH + model_path
    if os.path.exists(model_path):
        print(f"🔄 Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"🚀 Training new model")
        model = train_fn(model)
        print(f"🔄 Saving model from {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    return model

def pgd_attack(model, x_init, y, eps, loss_fn= nn.MSELoss(), n_steps=40, step_size=0.01):
    was_training = model.training
    model.eval()

    attack_point = x_init.clone().detach()
    attack_loss = (-float('inf')) * torch.ones(x_init.shape[0], device=x_init.device)

    with torch.enable_grad():
        adv_input = x_init.clone().detach().requires_grad_(True)

        for _ in range(n_steps):
            if adv_input.grad is not None:
                adv_input.grad.zero_()

            adv_outs = model(adv_input)
            obj = loss_fn(adv_outs, y)

            attack_point = torch.where(
                (obj >= attack_loss).view((-1,) + (1,) * (x_init.dim() - 1)),
                adv_input.detach().clone(), attack_point)
            attack_loss = torch.where(obj >= attack_loss, obj.detach().clone(), attack_loss)

            grad = torch.autograd.grad(obj.sum(), adv_input)[0]
            adv_input = adv_input + step_size * grad.sign()

            # Project to L∞ ball
            delta = torch.clamp(adv_input - x_init, min=-eps, max=eps)
            adv_input = (x_init + delta).detach().requires_grad_(True)

        # Final selection
        adv_outs = model(adv_input)
        obj = loss_fn(adv_outs, y)
        attack_point = torch.where(
            (obj >= attack_loss).view((-1,) + (1,) * (x_init.dim() - 1)),
            adv_input.detach().clone(), attack_point)

    # Restore model mode
    if was_training:
        model.train()

    return attack_point

        
def train_step(model, device, loader, optimizer, loss_fn, epoch, params={}, mode="train"):

    model.train()

    robust_eps = params.get("robust_eps", 0.1)
    method = params.get("robust_method", "standard")
    alpha = params.get("robust_alpha", 0.5)

    warmup_epochs = params.get("warmup_epochs", 1)
    alpha = min(epoch / warmup_epochs, 1.0) * alpha

    gradient_clipping = params.get("gradient_clipping", None)


    is_train = mode == "train"
    if is_train:
        model.train()
    else:
        model.eval()

    # mse_loss = 
    total_loss = 0.0
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    show_bar = params.get("progress_bar", False)

    bar = tqdm(loader) if show_bar else loader


    with context:

        for x, y in bar:
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()    

            loss = loss_fn(x, y, model)      
            # if method == "standard":
            #     output = model(x)
            #     loss = loss_fn(output, y)

            # elif method == "pgd":
            #     output_adv = model(pgd_attack(model, x, y, robust_eps, loss_fn))
            #     loss = loss_fn(output_adv, y)

            # elif method == "IBP":
            #     output_lb_cert = lower_bound_verified(model, x, robust_eps, method="IBP")
            #     loss =  loss_fn(output_lb_cert, y)

            # elif method == "CROWN-IBP":
            #     output = model(x)
            #     output_lb_cert = lower_bound_verified(model, x, robust_eps, method="CROWN-IBP")
            #     loss = (1-alpha) * loss_fn(output, y) + loss_fn(output_lb_cert, y) * alpha

            # elif method in ["MTL-IBP", "CC-IBP", "Exp-IBP"]:
            #     output_lb_cert = lower_bound_verified(model, x, robust_eps, method="IBP")
            #     output_adv = model(pgd_attack(model, x, y, robust_eps, loss_fn))

            #     if method == "CC-IBP":
            #         loss = loss_fn((1 - alpha) * output_adv +  alpha * output_lb_cert, y)

            #     elif method == "MTL-IBP":
            #         robust_loss = loss_fn(output_lb_cert, y)
            #         adv_loss = loss_fn(output_adv, y)

            #         loss = (1 - alpha) * adv_loss +  alpha * robust_loss

            #     elif method == "Exp-IBP":
            #         robust_loss = loss_fn(output_lb_cert, y)
            #         adv_loss = loss_fn(output_adv, y)

            #         loss = (adv_loss **(1 - alpha)) * (robust_loss ** alpha)       

            # else:
            #     raise ValueError(f"Unknown method '{method}'")
            
            if is_train:
                loss.backward()

                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

                
                optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)



def train_model_final(model, device, train_loader, test_loader, loss_fn, params={}):

    lr = params.get("learning_rate", 0.001)
    epochs = params.get("epochs", 10)
    weight_decay = params.get("weight_decay", 1e-5)
    patience = params.get("patience", None)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    best_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss  = train_step(model, device, train_loader, optimizer, loss_fn, epoch+1, params, "train")
        val_loss = train_step(model, device, test_loader, optimizer, loss_fn, epoch+1, params, "eval")

        print(f"Epoch {epoch+1:2d}, Train Loss: {train_loss:.6f}: Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience is not None and patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return train_losses, val_losses
