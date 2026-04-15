import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, rope_hz, **kwargs):
        super().__init__()

        self.nhead = nhead
        self.d_model = d_model
        self.rope_hz = rope_hz

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # head 합친 후 출력

        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    # RoPE 이론
    # Q, K 회전
    def apply_rope(self, x, T, device):
        if not hasattr(self, '_rope_cache') or self._rope_cache[0] != T or self._rope_cache[1] != device:
            d_k = x.shape[-1]
            theta = 1.0 / (self.rope_hz ** (torch.arange(0, d_k, 2, device=device) / d_k))
            positions = torch.arange(T, device=device)
            angles = positions.unsqueeze(1) * theta.unsqueeze(0)
            self._rope_cache = (T, device, angles.cos(), angles.sin())
        _, _, cos, sin = self._rope_cache

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        B, T, D = x.shape
        N = self.nhead
        d_k = D // N

        residual = x
        x = self.norm1(x)
        
        Q = self.Wq(x).view(B, T, N, d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, N, d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, N, d_k).transpose(1, 2)

        Q = self.apply_rope(Q, T, x.device)
        K = self.apply_rope(K, T, x.device)

        attn_mask = None
        
        if src_mask is not None:
            if src_mask.dtype == torch.bool:
                new_mask = torch.zeros_like(src_mask, dtype=x.dtype)
                new_mask.masked_fill_(src_mask, float("-inf"))
                attn_mask = new_mask.unsqueeze(0).unsqueeze(0) # (1, 1, T, T)
            else:
                attn_mask = src_mask.unsqueeze(0).unsqueeze(0)

        if src_key_padding_mask is not None:
            p_mask = src_key_padding_mask.view(B, 1, 1, T)
            
            if attn_mask is None:

                attn_mask = torch.zeros((B, 1, 1, T), device=x.device, dtype=x.dtype)
            
            attn_mask = attn_mask.masked_fill(p_mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False # 상삼각
        )

        # --- 5. Output Projection & Residual ---
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.dropout(self.out_proj(out))

        # --- 6. Feed Forward Block (Pre-Norm) ---
        residual = x
        x = self.norm2(x)
        x = self.ff1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.ff2(x)
        x = residual + self.dropout(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H*W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(
            q.transpose(-1,-2), k.transpose(-1,-2), v.transpose(-1,-2)
        )
        return x + self.out(attn.transpose(-1,-2).reshape(B, C, H, W))
    
class MacroDetector(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float, rope_hz:int, **kwargs):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
            
        )

        self.pad_id = [-1, -1, -1]
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_size) * 0.02)
        
        dim_feedforward = 4 * d_model
        self.encoder_layers = nn.ModuleList([
            CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout, rope_hz)
            for _ in range(num_layers)
        ])

        self.refiner = nn.GRU(d_model, d_model, batch_first=True)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_size)
        )

    def forward(self, src: torch.Tensor):
        B, T, F = src.shape
        src = src.to(self.mask_token.dtype)
        device = src.device
        dtype = src.dtype
        
        padding_mask = (src == -1).all(dim=-1)

        if self.training:
            mask_ratio = torch.empty(1).uniform_(0.5, 0.8).item()
            rand_mask = (torch.rand((B, T, 1), device=device) > mask_ratio).to(dtype=dtype)
        else:
            rand_mask = (torch.rand((B, T, 1), device=device) > 0.5).to(dtype=dtype)

        input_src = src.clone()

        input_src = input_src * rand_mask + self.mask_token * (1 - rand_mask)
        
        x = self.input_proj(input_src)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        memory = x
        memory, _ = self.refiner(memory) 
        prediction = self.out_proj(memory)

        return prediction, rand_mask