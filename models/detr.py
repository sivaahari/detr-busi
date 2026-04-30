import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DETR(nn.Module):
    def __init__(self, num_classes=3, num_queries=100):
        super().__init__()

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

        # ---------------- Positional Encoding ----------------
        self.row_embed = nn.Embedding(256, hidden_dim // 2)
        self.col_embed = nn.Embedding(256, hidden_dim // 2)

        # ---------------- Transformer ----------------
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            batch_first=True,
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
        f1 = x  # low-level features

        x = self.layer2(x)
        f2 = x

        x = self.layer3(x)
        f3 = x

        x = self.layer4(x)
        f4 = x  # high-level features

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

        # ---------------- Fuse features ----------------
        features = f1 + f2 + f3 + f4

        B, C, H, W = features.shape

        # ---------------- Positional Encoding ----------------
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)

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
        pos = pos.flatten(2).permute(0, 2, 1)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # ---------------- Transformer ----------------
        hs = self.transformer(features + pos, queries)

        # ---------------- Predictions ----------------
        class_logits = self.class_embed(hs)
        bbox = self.bbox_embed(hs)

        return class_logits, bbox