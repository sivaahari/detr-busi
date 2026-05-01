import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAttention(nn.Module):
    """
    Single-scale deformable self-attention (pure PyTorch, no custom CUDA).

    For each of the N input tokens:
      - Predict K sampling offsets (relative to the token's reference point)
      - Sample the value map at those K locations via bilinear interpolation
      - Predict K attention weights (softmax-normalised)
      - Output = weighted sum of K sampled values

    Complexity: O(N * K) vs O(N^2) for standard attention.
    With N=4096 and K=4 this is a 1024x reduction in attention cost.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_points = n_points
        self.d_head   = d_model // n_heads

        self.sampling_offsets  = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj        = nn.Linear(d_model, d_model)
        self.output_proj       = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias,   0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias,   0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        query:         torch.Tensor,   # (B, N, d_model)
        reference_pts: torch.Tensor,   # (B, N, 2)  — (x, y) in [0, 1]
        value_src:     torch.Tensor,   # (B, N, d_model)  — source to sample from
        spatial_shape: tuple,          # (H, W)
    ) -> torch.Tensor:                 # (B, N, d_model)

        B, N, _ = query.shape
        H, W    = spatial_shape

        # ── value map ────────────────────────────────────────────────────────
        value = self.value_proj(value_src)                          # (B, N, d_model)
        value = value.reshape(B, N, self.n_heads, self.d_head)      # (B, N, H_n, d_h)
        # Reshape to spatial grid for grid_sample: (B*H_n, d_h, H, W)
        value_spatial = value.permute(0, 2, 3, 1)                   # (B, H_n, d_h, N)
        value_spatial = value_spatial.reshape(B * self.n_heads, self.d_head, H, W)

        # ── sampling offsets ─────────────────────────────────────────────────
        offsets = self.sampling_offsets(query)                      # (B, N, H_n*K*2)
        offsets = offsets.reshape(B, N, self.n_heads, self.n_points, 2)
        # Scale offsets so they are in normalised [0,1] coordinate space
        scale = torch.tensor([W, H], dtype=torch.float32, device=query.device)
        offsets = offsets / scale                                   # (B, N, H_n, K, 2)

        # ── sampling locations = reference + offset, clamped to [0, 1] ──────
        ref = reference_pts[:, :, None, None, :]                    # (B, N, 1, 1, 2)
        sampling_locs = (ref + offsets).clamp(0.0, 1.0)            # (B, N, H_n, K, 2)

        # ── grid_sample expects coords in [-1, 1] ────────────────────────────
        grid = sampling_locs * 2.0 - 1.0                            # (B, N, H_n, K, 2)
        # Flatten to (B*H_n, N*K, 1, 2) — grid_sample needs (N_b, H_out, W_out, 2)
        grid = grid.permute(0, 2, 1, 3, 4)                         # (B, H_n, N, K, 2)
        grid = grid.reshape(B * self.n_heads, N * self.n_points, 1, 2)

        sampled = F.grid_sample(
            value_spatial, grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        # sampled: (B*H_n, d_h, N*K, 1) → squeeze → (B*H_n, d_h, N*K)
        sampled = sampled.squeeze(-1)
        sampled = sampled.reshape(B, self.n_heads, self.d_head, N, self.n_points)
        sampled = sampled.permute(0, 3, 1, 4, 2)                   # (B, N, H_n, K, d_h)

        # ── attention weights ─────────────────────────────────────────────────
        weights = self.attention_weights(query)                     # (B, N, H_n*K)
        weights = weights.reshape(B, N, self.n_heads, self.n_points)
        weights = weights.softmax(-1).unsqueeze(-1)                 # (B, N, H_n, K, 1)

        # ── weighted sum over K points ────────────────────────────────────────
        output = (weights * sampled).sum(dim=3)                     # (B, N, H_n, d_h)
        output = output.reshape(B, N, self.d_model)
        output = self.output_proj(output)                           # (B, N, d_model)

        return output


class DeformableEncoderLayer(nn.Module):
    """Deformable self-attention + FFN with pre-norm residual connections."""

    def __init__(self, d_model: int, n_heads: int, n_points: int, dim_ffn: int):
        super().__init__()
        self.self_attn = DeformableAttention(d_model, n_heads, n_points)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(),
            nn.Linear(dim_ffn, d_model),
        )

    def forward(self, src, ref_pts, spatial_shape):
        src = self.norm1(src + self.self_attn(src, ref_pts, src, spatial_shape))
        src = self.norm2(src + self.ffn(src))
        return src


class DeformableEncoder(nn.Module):
    """Stack of DeformableEncoderLayers."""

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        n_layers: int,
        n_points: int,
        dim_ffn:  int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableEncoderLayer(d_model, n_heads, n_points, dim_ffn)
            for _ in range(n_layers)
        ])

    def forward(self, src, ref_pts, spatial_shape):
        for layer in self.layers:
            src = layer(src, ref_pts, spatial_shape)
        return src
