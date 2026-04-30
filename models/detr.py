import torch
import torch.nn as nn
import torchvision.models as models


class DETR(nn.Module):
    def __init__(self, num_classes=3, num_queries=100):
        super().__init__()

        # Backbone
        self.backbone = models.resnet18(weights="DEFAULT")

        self.backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        hidden_dim = 256

        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # Positional Encoding
        self.row_embed = nn.Embedding(50, hidden_dim // 2)
        self.col_embed = nn.Embedding(50, hidden_dim // 2)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            batch_first=True,
        )

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes)

        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.input_proj(features)

        B, C, H, W = features.shape

        # Create positional encodings
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1)
        ], dim=-1)

        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)

        # Flatten
        features = features.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(2).permute(0, 2, 1)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Add positional encoding
        hs = self.transformer(features + pos, queries)

        class_logits = self.class_embed(hs)
        bbox = self.bbox_embed(hs)

        return class_logits, bbox