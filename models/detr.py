import torch
import torch.nn as nn
import torchvision.models as models


class DETR(nn.Module):
    def __init__(self, num_classes=3, num_queries=100):
        super().__init__()

        # -------- Backbone --------
        self.backbone = models.resnet18(weights="DEFAULT")

        # Modify first conv layer (2 channels instead of 3)
        self.backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        hidden_dim = 256

        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # -------- Transformer --------
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            batch_first=True,
        )

        # -------- Object Queries --------
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # -------- Prediction Heads --------
        self.class_embed = nn.Linear(hidden_dim, num_classes)

        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # normalized [0,1]
        )

    def forward(self, x):
        # Backbone
        features = self.backbone(x)  # (B, 512, H, W)
        features = self.input_proj(features)  # (B, 256, H, W)

        B, C, H, W = features.shape

        # Flatten spatial dimensions
        features = features.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Prepare queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Transformer
        hs = self.transformer(features, queries)

        # Predictions
        class_logits = self.class_embed(hs)  # (B, 100, 3)
        bbox = self.bbox_embed(hs)          # (B, 100, 4)

        return class_logits, bbox