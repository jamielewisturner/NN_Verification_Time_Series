import torch.nn as nn
import torch
from torch.nn import functional as F


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
    


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states for each layer
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.lstm_cells):
                h[i], c[i] = cell(input_t, (h[i], c[i]))
                input_t = h[i]  # output of current layer is input to next

        out = self.fc(h[-1])  # use last layer's final hidden state
        return out
    
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

# Transformer2 Model (No Dropout)
class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model=32, nhead=2, num_layers=2, dim_head=16):
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
        x = x.unsqueeze(-1)  # (B, T, 1)
        x = self.input_proj(x) + self.pos_embedding  # (B, T, d_model)
        
        # Pass through custom transformer encoder
        x = self.transformer(x)
        
        # Flatten the output for fully connected layers
        x = x.flatten(start_dim=1)  # (B, T * d_model)
        return self.fc(x)
    
