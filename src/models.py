import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from sklearn.covariance import LedoitWolf

class CustomSoftmax(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.exp(x)
        return e / (e.sum(dim=self.dim, keepdim=True) + self.eps)

class SimpleAssetAllocationModel(nn.Module):
    def __init__(self, input_channels=4, output_channels=4):
        super(SimpleAssetAllocationModel, self).__init__()
        
        input_size = input_channels
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_channels)
        self.softmax = CustomSoftmax(dim=-1)

        # self.fc4 = nn.Linear(4*50, 4)

    def forward(self, x):

        x = torch.flatten(x, start_dim=-2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)    
    
class CNNAllocatorCausal(nn.Module):
    def __init__(self, input_channels=4, time_steps=50, hidden_size=100, dropout_rate=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * time_steps, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_channels)
        self.softmax = CustomSoftmax(dim=-1)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))

        x = self.fc2(x)
        
        return self.softmax(x)
    
# import torch.nn as nn

# class CNNAllocatorCausal(nn.Module):
#     def __init__(self, input_channels=4, time_steps=50, hidden_size=100):
#         super().__init__()

#         # First causal convolution: kernel_size=3, dilation=1 → left padding = 2
#         self.pad1 = nn.ConstantPad1d((2, 0), 0)
#         self.conv1 = nn.Conv1d(
#             in_channels=input_channels,
#             out_channels=32,
#             kernel_size=3,
#             dilation=1,
#             padding=0
#         )

#         # Second causal convolution: kernel_size=3, dilation=1 → left padding = 2
#         self.pad2 = nn.ConstantPad1d((2, 0), 0)
#         self.conv2 = nn.Conv1d(
#             in_channels=32,
#             out_channels=64,
#             kernel_size=3,
#             dilation=1,
#             padding=0
#         )

#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * time_steps, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, input_channels)
#         self.softmax = CustomSoftmax(dim=-1)

#     def forward(self, x):
#         # x: (batch_size, time_steps, input_channels)
#         # transpose to (batch_size, input_channels, time_steps)
#         x = x.transpose(1, 2)

#         # First causal conv
#         x = self.pad1(x)
#         x = self.relu(self.conv1(x))

#         # Second causal conv
#         x = self.pad2(x)
#         x = self.relu(self.conv2(x))

#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return self.softmax(x)


class UniformModel(nn.Module):
    def __init__(self, n_assets=4):
        super(UniformModel, self).__init__()
        self.n_assets = n_assets

    def forward(self, x):
        b = x.shape[0]
        return torch.full((b, self.n_assets), 1/self.n_assets, device=x.device, dtype=x.dtype)
    
class FixedWeightModel(nn.Module):
    def __init__(self, default_weights):
        super(FixedWeightModel, self).__init__()
        self.weights = default_weights
    def forward(self, x):
        b = x.shape[0]
        return self.weights.to(x.device, dtype=x.dtype).expand(b, -1)
    


class MinVarianceModel(nn.Module):
    def __init__(self, saved_mean, saved_std):
        super(MinVarianceModel, self).__init__()
        self.shrink = LedoitWolf()
        self.eps = 1e-6
        self.saved_mean = saved_mean
        self.saved_std = saved_std
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        m = self.saved_mean.view(1, 1, -1).to(device)
        s = self.saved_std.view(1, 1, -1).to(device)
        x_raw = x * s + m
        windows = x_raw.detach().cpu().numpy()
        batch, _, n_assets = windows.shape
        mv_list = []
        for i in range(batch):
            data = windows[i]
            Sigma_shrunk = self.shrink.fit(data).covariance_
            inv_S = np.linalg.pinv(Sigma_shrunk)
            ones = np.ones(n_assets)
            w_raw = inv_S.dot(ones)
            w_clipped = np.maximum(w_raw, self.eps)
            w = w_clipped / w_clipped.sum()
            mv_list.append(w)
        mv_arr = np.stack(mv_list, axis=0)
        return torch.tensor(mv_arr, device=x.device, dtype=x.dtype)

class UniformModel(nn.Module):
    def __init__(self, n_assets: int = 4):
        super(UniformModel, self).__init__()
        self.n_assets = n_assets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.full((b, self.n_assets), 1/self.n_assets,
                          device=x.device, dtype=x.dtype)

class InverseVolModel(nn.Module):
    def __init__(self, saved_mean, saved_std):
        super(InverseVolModel, self).__init__()
        self.saved_mean = saved_mean
        self.saved_std = saved_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window, n_assets) standardized returns
        """

        m = self.saved_mean.view(1, 1, -1).to(x.device)
        s = self.saved_std.view(1, 1, -1).to(x.device)
        x_raw = x * s + m

        # compute sample vol per asset
        vol = x_raw.std(dim=1, unbiased=True)         # (batch, n_assets)
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum(dim=1, keepdim=True)
        return weights


class MLP(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x 

class CNN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear((input_size - 2) * 64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = self.fc(x)
        return x

    
# Custom Multihead Attention (No Dropout)
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape  # batch_size, seq_length, num_features
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Perform attention
        q, k, v = [t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in (q, k, v)]

        # dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn = torch.sigmoid(dots) #dots.softmax(dim=-1)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        dots = torch.clamp(dots, min=1e-6, max=1.0)  # Ensure no negative or zero values
       

        #Check that this actually does work
        # attn = dots.softmax(dim=-1)
        dots_exp = torch.exp(dots)
        attn = dots_exp / dots_exp.sum(dim=-1, keepdim=True)
        # attn = dots.softmax(dim=-1).clamp(min=1e-6, max=1.0)


        out = torch.matmul(attn, v)
        # out = out.transpose(1, 2).reshape(b, n, -1)
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1)
        out = self.to_out(out)
        return out

# Custom Transformer Encoder Layer (No Dropout)
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_head=16):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = Attention(d_model, heads=nhead, dim_head=dim_head)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        ##This seems to mess up the bounds??
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()

    def forward(self, src):
        # Multi-head self attention
        attn_output = self.self_attn(src)
        src = self.norm1(src + attn_output)
        
        # Feedforward network
        ff_output = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + ff_output)
        
        return src

# Custom Transformer Encoder (No Dropout)
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
    
class FinalCNN(nn.Module):
    def __init__(self, input_channels=4, time_steps=50, hidden_size=100, dropout_rate=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * time_steps, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_channels)
        self.softmax = CustomSoftmax(dim=-1)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        # print(x,  self.softmax(x), self.softmax(x/10))
        
        return self.softmax(x)

# Transformer2 Model (No Dropout)
class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model=4, nhead=2, num_layers=2, dim_head=16):
        super(Transformer, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(input_size, d_model))
        
        # Define the custom transformer encoder
        encoder_layer = CustomTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_head=dim_head)
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(d_model * input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # x = x.unsqueeze(-1)  # (B, T, 1)

        x = self.input_proj(x) + self.pos_embedding  # (B, T, d_model)
        
        # Pass through custom transformer encoder
        x = self.transformer(x)
        
        # Flatten the output for fully connected layers
        x = x.flatten(start_dim=1)  # (B, T * d_model)
        return self.fc(x)
    


class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model=4, nhead=2, num_layers=2, dim_head=16):
        super(Transformer, self).__init__()
        # input_proj now takes 4 input features per time step
        self.input_proj = nn.Linear(4, d_model)
        
        # Positional embeddings: (1, input_size, d_model)
        # Broadcasted to each batch
        self.pos_embedding = nn.Parameter(torch.randn(1, input_size, d_model))
        
        # Custom transformer encoder (you probably have these defined elsewhere)
        encoder_layer = CustomTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_head=dim_head)
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(d_model * input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        self.softmax = CustomSoftmax(dim=-1)

    def forward(self, x):
        """
        x: (batch_size, seq_len, 4)  # 4 input features (assets)
        """
        # Project 4 input features to d_model
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Add positional encoding (broadcasted across batch dimension)
        x = x + self.pos_embedding  # (B, T, d_model)
        
        # Pass through transformer
        x = self.transformer(x)  # (B, T, d_model)
        
        # Flatten time and d_model dimensions
        x = x.flatten(start_dim=1)  # (B, T * d_model)

        x = self.fc(x)

        x = self.softmax(x/10)
        
        return x
    
        
class LSTM_Unrolled(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=4, num_classes=4):
        super(LSTM_Unrolled, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = CustomSoftmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.lstm_cells):
                h[i], c[i] = cell(input_t, (h[i], c[i]))
                input_t = h[i]

        out = self.fc(h[-1])

        return self.softmax(out)
    

class LSTM_Cell_Unrolled(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_classes=4):
        super(LSTM_Unrolled, self).__init__()
        self.hidden_size = hidden_size

        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell( hidden_size)
        ])
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = CustomSoftmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.lstm_cells):
                h[i], c[i] = cell(input_t, (h[i], c[i]))
                input_t = h[i]

        out = self.fc(h[-1])

        return self.softmax(out)


class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=4, num_classes=4):
        super().__init__()
        # self.temperature = temperature

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            )

        self.fc = nn.Linear(50 * hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)   # (B, 50, hidden_size)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)      # project to (B, 4)
        return torch.softmax(out, dim=1)

