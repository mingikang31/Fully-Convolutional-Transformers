import torch 
import torch.nn as nn 
import torch.nn.functional as F

"""

"""

class LinearAttention(nn.Module):
    def __init__(self, d_hidden, num_heads, attention_dropout=0.1):
        super(LinearAttention, self).__init__()
        
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.head_dim = d_hidden // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)
        
        self.dropout = nn.Dropout(attention_dropout)
        
    def feature_map(self, x):
        """Apply kernel feature map (e.g., ReLU or ELU+1)"""
        # Using ELU + 1 as a common choice for linear attention
        return F.elu(x) + 1
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, d_hidden) where N is sequence length
            mask: optional attention mask
        Returns:
            output: (B, N, d_hidden)
        """
        B, N, _ = x.shape
        
        # Linear projections and split into heads
        q = self.W_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = self.W_k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        v = self.W_v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        
        # Apply feature map to make attention linear
        q = self.feature_map(q)  # (B, H, N, D)
        k = self.feature_map(k)  # (B, H, N, D)
        
        # Linear attention: O(N) complexity
        # Compute k^T @ v first: (B, H, D, N) @ (B, H, N, D) -> (B, H, D, D)
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)
        
        # Compute normalization: sum over sequence dimension
        k_sum = k.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        
        # Compute output: q @ (k^T @ v) / (q @ k^T @ 1)
        numerator = torch.einsum('bhnd,bhdm->bhnm', q, kv)  # (B, H, N, D)
        denominator = torch.einsum('bhnd,bvhmd->bhn', q, k_sum).unsqueeze(-1)  # (B, H, N, 1)
        
        output = numerator / (denominator + 1e-6)  # (B, H, N, D)
        
        # Reshape and project
        output = output.transpose(1, 2).reshape(B, N, self.d_hidden)  # (B, N, d_hidden)
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output


"""
FLatten Transformer: Vision Transformer using Focused Linear Attention
- https://github.com/LeapLabTHU/FLatten-Transformer/

Focused Linear Attention Module
"""

class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q @ kv * z

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x





"""
Efficient Attention: Attention with Linear Complexities
- https://github.com/cmsflash/efficient-attention/tree/master

Efficient Attention Module
"""
class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention