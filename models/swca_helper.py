import torch
import torch.nn as nn
import numpy as np 
import math
import torch.nn.functional as F
from einops import rearrange

def positional_encoding(max_len, embed_dim, device, requires_grad=False):
    # initialize a matrix angle_rads of all the angles
    angle_rads = np.arange(max_len)[:, np.newaxis] / np.power(
        10_000, (2 * (np.arange(embed_dim)[np.newaxis, :] // 2)) / np.float32(embed_dim)
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    # Out: 1, max_len, embed_dim
    return torch.tensor(pos_encoding, dtype=torch.float32, device=device, requires_grad=requires_grad)
    
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(2, 3))
    
def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)]), 
        dtype=torch.long)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class MultiHeadCrossWindowAttention(nn.Module):
    """Some Information about MultiHeadCrossAttention"""
    def __init__(self, skip_channels, cyclic_shift, window_size, num_heads, 
                    qkv_bias=False, attn_drop_prob=0.0, lin_drop_prob=0.0, device='cuda'):
        super(MultiHeadCrossWindowAttention, self).__init__()
        self.device = device
        self.skip_channels = skip_channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dims = skip_channels // num_heads 
        self.cyclic_shift = cyclic_shift
        
        if cyclic_shift:
            displacement = window_size // 2
            self.cyclic_propagate = CyclicShift(-displacement)
            self.cyclic_revert = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size=window_size, 
                                displacement=displacement,
                                upper_lower=True, 
                                left_right=False), 
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(window_size=window_size, 
                                displacement=displacement,
                                upper_lower=False, 
                                left_right=True), 
                requires_grad=False
            )

        self.relative_indices = get_relative_distances(window_size) + window_size - 1
        # self.pe = nn.Parameter(torch.zeros(2 * window_size - 1, 2 * window_size - 1))
        self.pe = nn.Parameter(torch.randn(window_size**2, window_size**2), requires_grad=True)
        
        self.q = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.k = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.v = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        
        self.lin = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.lin_drop = nn.Dropout(lin_drop_prob)
        
        self.memory = nn.Sequential(
            nn.Conv2d(in_channels=skip_channels*2, out_channels=skip_channels, kernel_size=1, padding="same", bias=False),
            nn.GroupNorm(num_groups=16, num_channels=skip_channels),
            nn.GELU()
        )

    def forward(self, skip, x):
        """
        Args: 
            skip    : b, c, h, w
            x       : b, c, h, w
        Return:
        """
        b, c, h, w = x.shape
        
        # Feature processing
        target_h = math.ceil(h / self.window_size) * self.window_size
        target_w = math.ceil(w / self.window_size) * self.window_size
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        skip = F.interpolate(skip, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        memory = self.memory( torch.cat([x, skip], dim=1) )
        
        if self.cyclic_shift:
            x = self.cyclic_propagate(x)
            skip = self.cyclic_propagate(skip)
        
        b, c, h, w = x.shape
        n_h, n_w = h//self.window_size, w//self.window_size
        window_squared = self.window_size*self.window_size
        
        # Reshape x and skip to [b, num_head, n_h*n_w, windows*window, head_dim]
        # print(x.shape, skip.shape)s
        x = x.reshape(b, self.num_heads, self.head_dims, n_h, self.window_size, n_w, self.window_size)
        x = x.permute(0, 1, 3, 5, 4, 6, 2) # b, num_head, n_h, n_w, window, window, head_dim
        x = x.reshape(b, self.num_heads, n_h*n_w, window_squared, self.head_dims)
        
        skip = skip.reshape(b, self.num_heads, self.head_dims, n_h, self.window_size, n_w, self.window_size)
        skip = skip.permute(0, 1, 3, 5, 4, 6, 2) # b, num_head, n_h, n_w, window, window, head_dim
        skip = skip.reshape(b, self.num_heads, n_h*n_w, window_squared, self.head_dims)
        
        q = self.q(x)       # b, num_head, n_h*n_w, window_squared, head_dim
        k = self.k(skip)    # b, num_head, n_h*n_w, window_squared, head_dim
        v = self.v(skip)    # b, num_head, n_h*n_w, window_squared, head_dim
        
        # qk = b, num_head, n_h*n_w, window_squared, window_squared
        qk = ( torch.matmul(q, k.transpose(3, 4)) ) / np.sqrt(self.head_dims)
        qk += self.pe[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        
        if self.cyclic_shift:
            qk[:, :, -n_w:] += self.upper_lower_mask
            qk[:, :, n_w-1::n_w] += self.left_right_mask
        
        attn_weight_og = torch.softmax(qk, dim=-1)
        attn_weight = self.attn_drop(attn_weight_og) # b, num_head, n_h*n_w, window_squared, window_squared
        out = torch.matmul(attn_weight, v)  # b, num_head, n_h*n_w, window_squared, head_dim
        out = self.lin_drop(self.lin(out))  # b, num_head, n_h*n_w, window_squared, head_dim
        
        # out ==> [b, num_head, n_h*n_w, window_squared, head_dim] to [b, e, h, w]
        out = out.permute(0, 1, 4, 2, 3).reshape(b, c, n_h, n_w, self.window_size, self.window_size)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        
        if self.cyclic_shift:
            out = self.cyclic_revert(out)
        
        # print(out.shape, memory.shape)
        out = out + memory
        
        return out, attn_weight_og


class MultiheadAttention(nn.Module):
    """Some Information about MultiheadAttention"""
    def __init__(self, embed_dim, num_heads, qkv_bias=False, att_drop_prob=0.0, lin_drop_prob=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.head_dims = embed_dim // num_heads
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim*3, bias=qkv_bias)
        self.lin = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        
        self.att_drop = nn.Dropout(p=att_drop_prob)
        self.lin_drop = nn.Dropout(p=lin_drop_prob)
        
        
    def forward(self, x):
        """
        input: 
            x = (batch_size, n_patches, embed_dim)
        output:
            out = (batch_size, n_patches, embed_dim)
        """
        batch_size, n_tokens, dim = x.shape # n_tokes == (n_patches + 1)
        
        if dim != self.embed_dim: 
            print(f"--> Attention | Dim : {dim.shape} != Embed dim : {self.embed_dim}")
            raise ValueError
        
        # qkv = (batch_size, n_tokes, embed_dim * 3)
        qkv = self.qkv(x)
        
        # reshaped qkv = (batch_size, n_tokes, 3, num_heads, head_dims)
        # permuted qkv = (3, batch_size, num_heads, n_tokes, head_dims)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dims)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, and v = (batch_size, num_heads, n_tokes, head_dims)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        qk_transposed = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dims) # (batch_size, num_heads, n_tokens, n_tokens)
        attention_weights_og = torch.softmax(qk_transposed, dim=-1) # (batch_size, num_heads, n_tokens, n_tokens)
        attention_weights = self.att_drop(attention_weights_og)
        
        weighted_avg = torch.matmul(attention_weights, v) # (batch_size, num_heads, n_tokes, head_dims)
        weighted_avg = weighted_avg.transpose(1, 2).flatten(start_dim=2) # (batch_size, n_tokes, num_heads * head_dims)
        
        out = self.lin(weighted_avg) # (batch_size, n_tokes, embed_dim)
        out = self.lin_drop(out) # (batch_size, n_tokes, embed_dim)
        
        return out, attention_weights_og