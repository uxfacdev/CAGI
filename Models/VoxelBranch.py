import torch
import torch.nn as nn

# --- Voxel (Structural) Branch Components ---

class SqueezeExcitation3D(nn.Module):
    """3D SE block for channel-wise attention in voxel space."""
    def __init__(self, in_channels, reduction=24):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(self.pool(x))

class MBConv3D(nn.Module):
    """3D Inverted Residual Block (MobileNetV3 style) for efficient structural learning."""
    def __init__(self, in_ch, out_ch, expand_ratio=6, kernel_size=3, stride=1, se_reduction=16):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        self.expand = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv3d(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size//2, groups=mid_ch, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.SiLU()
        )

        self.se = SqueezeExcitation3D(mid_ch, reduction=se_reduction)
        self.project = nn.Sequential(
            nn.Conv3d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch)
        )

    def forward(self, x):
        identity = x
        out = self.project(self.se(self.depthwise(self.expand(x))))

        return out + identity if self.use_res_connect else out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt3D(nn.Module):
    """3D Coordinate Attention for capturing long-range spatial dependencies along D, H, W axes."""
    def __init__(self, inp, oup, reduction=16):
        super().__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))  
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1)) 
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))  

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        B, C, D, H, W = x.size()

        # Pooling along axes
        x_d = self.pool_d(x)
        x_h = self.pool_h(x).permute(0, 1, 3, 2, 4)
        x_w = self.pool_w(x).permute(0, 1, 4, 2, 3)

        # Concatenate and encode spatial info
        y = self.act(self.bn1(self.conv1(torch.cat([x_d, x_h, x_w], dim=2))))
        y_d, y_h, y_w = torch.split(y, [D, H, W], dim=2)

        # Restore dimensions and apply sigmoid
        a_d = self.conv_d(y_d).sigmoid()
        a_h = self.conv_h(y_h.permute(0, 1, 3, 2, 4)).sigmoid()
        a_w = self.conv_w(y_w.permute(0, 1, 3, 4, 2)).sigmoid()
        
        return identity * a_d * a_h * a_w

    
class VoxelBranch(nn.Module):
    """Processes 3D structural data and combines it with mutation identity embeddings."""
    def __init__(self, in_ch=63, emb_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            MBConv3D(in_ch, 32), MBConv3D(32, 32),
            MBConv3D(32, 48), MBConv3D(48, 48),
            MBConv3D(48, 64), MBConv3D(64, 64, stride=2), # Downsample to 4x4x4
            MBConv3D(64, 64), MBConv3D(64, 96), MBConv3D(96, 96),
            MBConv3D(96, 96), MBConv3D(96, 96), MBConv3D(96, 96),
            MBConv3D(96, emb_dim), MBConv3D(emb_dim, emb_dim), MBConv3D(emb_dim, emb_dim)
        )

        self.coordatt = CoordAtt3D(emb_dim, emb_dim)
        self.pool = nn.AdaptiveAvgPool3d(1)  

        # Embeddings for Wild-Type and Mutant Amino Acids
        self.ref_emb = nn.Embedding(21, emb_dim // 2)  # 64
        self.mut_emb = nn.Embedding(21, emb_dim // 2)  # 64
        self.mut_fusion = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.SiLU()
        )
        self.refine = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.SiLU()
        )

    def forward(self, x, ref_idx, mut_idx):
        struct_feat = self.pool(self.coordatt(self.backbone(x))).flatten(1)

        # Mutation-specific feature
        mut_feat = self.mut_fusion(torch.cat([self.ref_emb(ref_idx), self.mut_emb(mut_idx)], dim=1))

        # Combine structural context with identity change
        return self.refine(struct_feat + mut_feat)

