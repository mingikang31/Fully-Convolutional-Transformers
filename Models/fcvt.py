# PatchEmbedding2D 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchsummary import summary
import numpy as np 

"""
Fully Convolutional Vision Transformer (FCVT) Model Class
"""

class PatchEmbedding2D(nn.Module):
    def __init__(self, d_hidden, img_size, patch_size, n_channels=3):
        super(PatchEmbedding2D, self).__init__()

        self.d_hidden = d_hidden # Dimensionality of Model 
        self.img_size = img_size # Size of Image
        self.patch_size = patch_size # Patch Size 
        self.n_channels = n_channels # Number of Channels in Image
        
        self.linear_projection = nn.Conv2d(in_channels=n_channels, out_channels=d_hidden, kernel_size=patch_size, stride=patch_size) # Linear Projection Layer
        self.norm = nn.LayerNorm(d_hidden) # Normalization Layer

    def forward(self, x):
        x = self.linear_projection(x) # (B, C, H, W) -> (B, d_hidden, H', W')
        x = x.permute(0, 2, 3, 1) # (B, d_hidden, H', W') -> (B, H', W', d_hidden)
        x = self.norm(x) # (B, H', W', d_hidden) -> (B, H', W', d_hidden)
        x = x.permute(0, 3, 1, 2) # (B, H', W', d_hidden) -> (B, d_hidden, H', W')
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_hidden, height, width):
        super(PositionalEncoding2D, self).__init__() 

        self.d_hidden = d_hidden 
        self.height = height 
        self.width = width 

        # Class token 
        self.class_token = nn.Parameter(torch.randn(1, d_hidden, 1, 1))

        # Learnable 2D positional embeddings for width + 1 (patches + class token)
        self.positional_encoding = nn.Parameter(torch.randn(1, d_hidden, height, width + 1))

    def forward(self, x):
        B, _, H, _ = x.shape 

        class_tokens = self.class_token.expand(B, -1, H, -1) 

        # Concatenate along width dimension 
        x = torch.cat([class_tokens, x], dim=3) # (B, d_hidden, H, W + 1)

        # Add positional encoding 
        x = x + self.positional_encoding # (B, d_hidden, H, W + 1)
        return x

# TODO is not working, need to fix later
class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_hidden, height, width, temperature = 10000):
        super(SinusoidalPositionalEncoding2D, self).__init__() 

        self.d_hidden = d_hidden 
        self.height = height 
        self.width = width 
        self.temperature = temperature

        # Class token 
        self.class_token = nn.Parameter(torch.randn(1, d_hidden, 1, 1))

        # 2D sinusoidal positional encoding 
        # split d_hidden: half for height encoding, half for width encoding 
        pe = torch.zeros(d_hidden, height, width + 1) # +1 for class token column

        d_model_half = d_hidden // 2
        
        # Compute division term
        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * 
                           (-np.log(self.temperature) / d_model_half))
        
        # Height encoding (first half of channels)
        pos_h = torch.arange(0, height).unsqueeze(1).float()  # (H, 1)
        pe[:d_model_half:2, :, :] = torch.sin(pos_h * div_term).unsqueeze(-1).repeat(1, 1, width + 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_h * div_term).unsqueeze(-1).repeat(1, 1, width + 1)
        
        # Width encoding (second half of channels)
        # Note: width + 1 to include class token position (position 0)
        pos_w = torch.arange(0, width + 1).unsqueeze(1).float()  # (W+1, 1)
        pe[d_model_half::2, :, :] = torch.sin(pos_w * div_term).unsqueeze(1).repeat(1, height, 1)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_w * div_term).unsqueeze(1).repeat(1, height, 1)
        
        # Register as buffer (non-trainable)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, d_hidden, H, W+1)

    def forward(self, x):
        # x shape: (B, d_hidden, H, W)
        B, _, H, _ = x.shape 

        # Expand class token for batch and height
        class_tokens = self.class_token.expand(B, -1, H, -1)  # (B, d_hidden, H, 1)

        # Concatenate along width dimension 
        x = torch.cat([class_tokens, x], dim=3)  # (B, d_hidden, H, W + 1)

        # Add sinusoidal positional encoding (non-trainable)
        x = x + self.pe
        
        return x

class ConvolutionalAttention2D_Old(nn.Module):
    def __init__(self, d_hidden, num_heads, attention_dropout):
        super(ConvolutionalAttention2D_Old, self).__init__()

        self.d_hidden = d_hidden
        self.dropout = nn.Dropout(attention_dropout)

        # Pointwise Convolution to keep dimension
        self.W_q = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1, bias = False)
        self.W_k = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1, bias = False)
        self.W_v = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1, bias = False)
        self.W_o = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1)

        # Pointwise Convolution for KQV
        self.W_kqv = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1, bias = False)
        self.W_kqv.weight.requires_grad = False

    def phi(self, x):
        return F.elu(x) + 1
    
    def forward(self, x, mask = None):
        B, _, _, _ = x.shape 
        
        q = self.W_q(x) # (B, d_hidden, H, W)
        k = self.W_k(x) # (B, d_hidden, H, W)
        v = self.W_v(x) # (B, d_hidden, H, W)

        # Need Phi_k and Phi_q 
        ## USING PLACEHOLDERS FOR NOW
        phi_q = self.phi(q) 
        phi_k = self.phi(k)
        phi_v = self.phi(v) 

        qv_matrix = torch.einsum('bchw,bdhw->bcd', phi_q, phi_v)  # (32, n_channel_q, n_channel_v)

        # for each batch index, insert for weight and convolve on phi_k
        attended_batch = []
        for i in range(B):
            self.W_kqv.weight.data = qv_matrix[i].unsqueeze(-1).unsqueeze(-1) 
            k_batch_index = phi_k[i].unsqueeze(0)
            out = self.W_kqv(k_batch_index)
            attended_batch.append(out) 

        combined_batch = torch.cat(attended_batch, dim=0)
        out = self.W_o(combined_batch) 
        out = self.dropout(out)
        return out


class ConvolutionalAttention2D(nn.Module):
    """
    Docstring for ConvolutionalAttention2D
    - Dynamic Conv2d Kernel Linear Attention from original idea
    """
    
    def __init__(self, d_hidden, num_heads, attention_dropout):
        super(ConvolutionalAttention2D, self).__init__()

        self.d_hidden = d_hidden
        self.dropout = nn.Dropout(attention_dropout)

        # Pointwise Convolution to keep dimension
        self.W_q = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1, bias = False)
        self.W_k = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1, bias = False)
        self.W_v = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1, bias = False)
        self.W_o = nn.Conv2d(d_hidden, d_hidden, kernel_size = 1, stride = 1)

    def phi(self, x):
        return F.elu(x) + 1
    
    def forward(self, x, mask = None):
        B, C, H, W = x.shape
        
        q = self.W_q(x) # (B, d_hidden, H, W)
        k = self.W_k(x) # (B, d_hidden, H, W)
        v = self.W_v(x) # (B, d_hidden, H, W)

        phi_q = self.phi(q) 
        phi_k = self.phi(k)
        phi_v = self.phi(v) 

        # Comput Q^T @ V for each batch -> (B, C, C)
        qv_matrix = torch.einsum('bchw,bdhw->bcd', phi_q, phi_v)  # (B, n_channel_q, n_channel_v)

        # Reshape for grouped convolution 
        weight = qv_matrix.reshape(B * C, C, 1, 1)  # (B * n_channel_q, n_channel_v, 1, 1)
        k_grouped = phi_k.reshape(1, B * C, H, W)  # (1, B * n_channel_k, H, W )

        out = F.conv2d(k_grouped, weight, groups=B)  # (1, B * n_channel_q, H, W)
        out = out.reshape(B, C, H, W)  # (B, n_channel_q, H, W)

        out = self.W_o(out) 
        return self.dropout(out) 

class FCVTEncoder(nn.Module):
    def __init__(self, d_hidden, d_mlp, n_heads, dropout, attention_dropout):
        super(FCVTEncoder, self).__init__()
        
        self.d_hidden = d_hidden 
        self.d_mlp = d_mlp 
        self.n_heads = n_heads 
        self.dropout = dropout 
        self.attention_dropout = attention_dropout 

        self.conv_attention = ConvolutionalAttention2D(d_hidden, n_heads, attention_dropout)

        # Norm and Dropout 
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) 

        # MLP 
        self.mlp = nn.Sequential(
            nn.Conv2d(d_hidden, d_mlp, kernel_size=1, stride=1), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Conv2d(d_mlp, d_hidden, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x_norm = x.permute(0, 2, 3, 1) # (B, d_hidden, H', W') -> (B, H', W', d_hidden)
        x_norm = self.norm1(x_norm) # (B, H', W', d_hidden) -> (B, H', W', d_hidden)
        x_norm = x_norm.permute(0, 3, 1, 2) # (B, H', W', d_hidden) -> (B, d_hidden, H', W')

        attn_output = self.conv_attention(x_norm)
        x = x + self.dropout1(attn_output)

        # Post-Norm Feed Forward Network
        x_norm = x.permute(0, 2, 3, 1) # (B, d_hidden, H', W') -> (B, H', W', d_hidden)
        x_norm = self.norm2(x_norm) # (B, H', W', d_hidden) -> (B, H', W', d_hidden)
        x_norm = x_norm.permute(0, 3, 1, 2) # (B, H', W', d_hidden) -> (B, d_hidden, H', W')
        mlp_output = self.mlp(x_norm)
        x = x + self.dropout2(mlp_output)
        return x 

class FCVT(nn.Module):
    def __init__(self, d_hidden, d_mlp, img_size, n_classes, n_heads, patch_size, n_channels, n_layers, dropout, attention_dropout):
        super(FCVT, self).__init__() 
        assert img_size[1] % patch_size == 0 and img_size[2] % patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_hidden % n_heads == 0, "d_hidden must be divisible by n_heads"

        self.model = "FCVT"
        self.d_hidden = d_hidden 
        self.d_mlp = d_mlp
        self.img_size = img_size[1:]
        self.n_classes = n_classes 
        self.n_heads = n_heads
        self.patch_size = (patch_size, patch_size)
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.dropout = dropout 
        self.attention_dropout = attention_dropout

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1 # + 1 for class token 

        # Layers
        self.patch_embedding = PatchEmbedding2D(d_hidden, img_size, patch_size, n_channels)
        self.positional_encoding = PositionalEncoding2D(d_hidden, img_size[1] // patch_size, img_size[2] // patch_size)
        # self.positional_encoding = SinusoidalPositionalEncoding2D(d_hidden, img_size[1] // patch_size, img_size[2] // patch_size)

        self.transformer_encoder = nn.Sequential(*[FCVTEncoder(
            d_hidden, d_mlp, n_heads, dropout, attention_dropout
        ) for _ in range(self.n_layers)])


        # (Batch, d_hidden, height, width) as input for classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(d_hidden, n_classes)
        )
        
    def forward(self, x):
        x = self.patch_embedding(x) 
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x) 

        class_output = x[:, :, :, 0:1]
        x = self.classifier(class_output)
        return x
    
    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

if __name__ == "__main__":
    fct = FCVT(
        d_hidden = 192, 
        d_mlp=768, 
        img_size=(3, 224, 224),
        n_classes=100,
        n_heads=3,
        patch_size=16,
        n_channels=3,
        n_layers=12,
        dropout=0.1,
        attention_dropout=0.1
        )
    ex = torch.randn(32, 3, 224, 224) 
    out = fct.forward(ex)
    print(out.shape)

    print(fct.parameter_count())
    # 5,539,300