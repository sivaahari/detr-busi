import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import configs.config as cfg
from models.deformable_attention import DeformableEncoder
from models.segmentation_head import SegmentationHead


class DETR(nn.Module):
    def __init__(self, num_classes=3, num_queries=100, use_segmentation=True):
        super().__init__()
        
        self.use_segmentation = use_segmentation

        # ---------------- Backbone ----------------
        backbone = models.resnet18(weights="DEFAULT")

        # modify first conv layer (2-channel input)
        backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # split backbone into stages
        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        hidden_dim = 256

        # ---------------- Multi-scale projections ----------------
        self.input_proj1 = nn.Conv2d(64, hidden_dim, kernel_size=1)
        self.input_proj2 = nn.Conv2d(128, hidden_dim, kernel_size=1)
        self.input_proj3 = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.input_proj4 = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # learned fusion: concat 4 scale features → single hidden_dim map
        self.fusion_conv = nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1)
        
        # ---------------- Segmentation Head ----------------
        if self.use_segmentation:
            self.seg_head = SegmentationHead(
                in_channels_list=[64, 128, 256, 512],
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )

        # ---------------- Positional Encoding ----------------
        self.row_embed = nn.Embedding(256, hidden_dim // 2)
        self.col_embed = nn.Embedding(256, hidden_dim // 2)

        # ---------------- Transformer ----------------
        self.encoder = DeformableEncoder(
            d_model=hidden_dim,
            n_heads=cfg.NHEAD,
            n_layers=cfg.ENC_LAYERS,
            n_points=cfg.N_POINTS,
            dim_ffn=cfg.DIM_FFN,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=cfg.NHEAD,
                dim_feedforward=cfg.DIM_FFN,
                batch_first=True,
            ),
            num_layers=cfg.DEC_LAYERS,
        )

        # ---------------- Queries ----------------
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # ---------------- Heads ----------------
        self.class_embed = nn.Linear(hidden_dim, num_classes)

        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # normalized output [0,1]
        )

    def forward(self, x):
        # ---------------- Multi-scale feature extraction ----------------
        x = self.layer1(x)
        f1 = x  # low-level features (64, 64x64)

        x = self.layer2(x)
        f2 = x  # (128, 32x32)

        x = self.layer3(x)
        f3 = x  # (256, 16x16)

        x = self.layer4(x)
        f4 = x  # high-level features (512, 8x8)
        
        # Store backbone features for segmentation
        backbone_features = [f1, f2, f3, f4] if self.use_segmentation else None

        # ---------------- Project to common dimension ----------------
        f1 = self.input_proj1(f1)
        f2 = self.input_proj2(f2)
        f3 = self.input_proj3(f3)
        f4 = self.input_proj4(f4)

        # ---------------- Resize to same spatial size ----------------
        target_size = f1.shape[-2:]

        f2 = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode="bilinear", align_corners=False)
        f4 = F.interpolate(f4, size=target_size, mode="bilinear", align_corners=False)

        # ---------------- Fuse features (learned) ----------------
        features = self.fusion_conv(torch.cat([f1, f2, f3, f4], dim=1))

        B, C, H, W = features.shape

        # ---------------- Positional Encoding ----------------
        i = torch.arange(W, device=features.device)
        j = torch.arange(H, device=features.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(H, 1, 1),
                y_emb.unsqueeze(1).repeat(1, W, 1),
            ],
            dim=-1,
        )

        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)

        # ---------------- Flatten ----------------
        features = features.flatten(2).permute(0, 2, 1)
        pos      = pos.flatten(2).permute(0, 2, 1)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # ---------------- Reference points (normalized grid) ----------------
        col_idx = torch.arange(W, device=features.device).float() / W
        row_idx = torch.arange(H, device=features.device).float() / H
        grid_y, grid_x = torch.meshgrid(row_idx, col_idx, indexing="ij")
        ref_pts = torch.stack([grid_x, grid_y], dim=-1)          # (H, W, 2)
        ref_pts = ref_pts.flatten(0, 1).unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)

        # ---------------- Deformable encoder ----------------
        memory = self.encoder(features + pos, ref_pts, (H, W))   # (B, H*W, 256)

        # ---------------- Standard decoder ----------------
        hs = self.decoder(queries, memory)                        # (B, num_queries, 256)

        # ---------------- Predictions ----------------
        class_logits = self.class_embed(hs)
        bbox         = self.bbox_embed(hs)

        # ---------------- Segmentation ----------------
        seg_mask = None
        if self.use_segmentation and backbone_features is not None:
            seg_mask = self.seg_head(backbone_features)  # (B, num_classes, H, W)
        
        if self.use_segmentation:
            return class_logits, bbox, seg_mask
        return class_logits, bbox
