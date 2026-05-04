import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out


class AttentionGate(nn.Module):
    def __init__(self, g_channels, x_channels, int_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, int_channels, kernel_size=1),
            nn.BatchNorm2d(int_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, int_channels, kernel_size=1),
            nn.BatchNorm2d(int_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MKIRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2):
        super().__init__()
        mid_channels = in_channels * expansion
        
        self.pw_conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.depth_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                                     padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.pw_conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.skip = in_channels == out_channels

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.pw_conv1(x)))
        out = self.relu(self.bn2(self.depth_conv(out)))
        out = self.bn3(self.pw_conv2(out))
        
        if self.skip:
            out = out + identity
        return out


class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels_list, hidden_dim=256, num_classes=3):
        super().__init__()
        
        self.in_channels_list = in_channels_list  # [64, 128, 256, 512]
        
        # Project all backbone features to hidden_dim
        self.proj1 = nn.Conv2d(in_channels_list[0], hidden_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels_list[1], hidden_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(in_channels_list[2], hidden_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(in_channels_list[3], hidden_dim, kernel_size=1)
        
        # Bottom (deepest) processing
        self.bottom = nn.Sequential(
            MKIRBlock(hidden_dim, hidden_dim),
            MKIRBlock(hidden_dim, hidden_dim)
        )
        
        # Attention gates for skip connections
        self.gate1 = AttentionGate(hidden_dim, hidden_dim, hidden_dim // 2)
        self.gate2 = AttentionGate(hidden_dim, hidden_dim, hidden_dim // 2)
        self.gate3 = AttentionGate(hidden_dim, hidden_dim, hidden_dim // 2)
        
        # Upsampling blocks
        self.up1 = self._up_block(hidden_dim, hidden_dim)
        self.up2 = self._up_block(hidden_dim, hidden_dim)
        self.up3 = self._up_block(hidden_dim, hidden_dim)
        self.up4 = self._up_block(hidden_dim, hidden_dim)
        
        # Output head
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )
        
    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MKIRBlock(out_ch, out_ch)
        )
    
    def forward(self, features_list):
        """
        features_list: [f1, f2, f3, f4] from backbone (from shallow to deep)
        """
        f1, f2, f3, f4 = features_list
        
        # Project to common dimension
        p1 = self.proj1(f1)  # 64 -> 256
        p2 = self.proj2(f2)  # 128 -> 256
        p3 = self.proj3(f3)  # 256 -> 256
        p4 = self.proj4(f4)  # 512 -> 256
        
        # Process deepest features
        d = self.bottom(p4)
        
        # Upsample and fuse with skip connections (decoder pattern)
        # d: 8x8, p3: 16x16
        d = self.up1(d)
        d = self.gate1(d, p3)
        
        # d: 16x16, p2: 32x32
        d = self.up2(d)
        d = self.gate2(d, p2)
        
        # d: 32x32, p1: 64x64
        d = self.up3(d)
        d = self.gate3(d, p1)
        
        # Final upsampling to 256x256
        d = self.up4(d)
        
        # Output segmentation mask
        out = self.output(d)
        
        return out


class SegmentationHead(nn.Module):
    def __init__(self, in_channels_list, hidden_dim=256, num_classes=3):
        super().__init__()
        self.decoder = SegmentationDecoder(in_channels_list, hidden_dim, num_classes)
        
    def forward(self, backbone_features):
        """
        backbone_features: List of features from backbone stages [f1, f2, f3, f4]
        """
        return self.decoder(backbone_features)