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


class StructuralMutationAttention(nn.Module):
    """
    Option 2: 변이 정보(Mutation)를 Query로 하여 
    구조 피처(Voxel)의 중요한 부분을 선택적으로 강조하는 어텐션
    """
    def __init__(self, feat_dim, mut_dim):
        super().__init__()
        self.mut_to_query = nn.Linear(mut_dim, feat_dim)
        self.kv_conv = nn.Conv3d(feat_dim, feat_dim * 2, kernel_size=1)
        self.scale = feat_dim ** -0.5
        self.final_conv = nn.Conv3d(feat_dim, feat_dim, kernel_size=1)

    def forward(self, x, mut_emb):
        # x: [B, C, D, H, W] (구조 피처 맵)
        # mut_emb: [B, mut_dim] (변이 아미노산 임베딩)
        B, C, D, H, W = x.shape
        
        # 1. Query 생성: 변이 정보를 구조 피처와 같은 차원으로 매핑
        q = self.mut_to_query(mut_emb).view(B, 1, C) # [B, 1, C]
        
        # 2. Key, Value 생성: 3D 공간의 각 위치를 Key/Value로 변환
        kv = self.kv_conv(x).view(B, 2*C, -1).permute(0, 2, 1) # [B, D*H*W, 2*C]
        k, v = torch.chunk(kv, 2, dim=-1) # 각각 [B, D*H*W, C]

        # 3. Attention Score: "변이가 구조의 어느 위치와 관련 깊은가?"
        # [B, 1, C] @ [B, C, D*H*W] -> [B, 1, D*H*W]
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 4. Attention 적용
        # [B, 1, D*H*W] @ [B, D*H*W, C] -> [B, 1, C]
        out = torch.bmm(attn, v).view(B, C, 1, 1, 1)
        
        # 5. Gating: 구조 피처 맵에 어텐션 결과 반영
        return x * out.sigmoid()

class VoxelBranch(nn.Module):
    def __init__(self, in_ch=63, emb_dim=128):
        super().__init__()
        
        # 스테이지 1: 초기 구조 특징 추출
        self.stage1 = nn.Sequential(
            MBConv3D(in_ch, 32), MBConv3D(32, 48),
            MBConv3D(48, 64), MBConv3D(64, 64, stride=2) # 4x4x4 하향 샘플링
        )

        # --- [Option 2: Mid-Attention] ---
        # 64채널 시점에서 변이 정보 주입
        self.mut_mid_attn = StructuralMutationAttention(64, emb_dim // 2)
        
        # 스테이지 2: 심층 특징 추출
        self.stage2 = nn.Sequential(
            MBConv3D(64, 96), MBConv3D(96, 96),
            MBConv3D(96, 96), MBConv3D(96, 128),
            MBConv3D(128, emb_dim), MBConv3D(emb_dim, emb_dim)
        )

        self.coordatt = CoordAtt3D(emb_dim, emb_dim)
        self.pool = nn.AdaptiveAvgPool3d(1)

        # 임베딩 레이어
        self.ref_emb = nn.Embedding(21, emb_dim // 2)
        self.mut_emb = nn.Embedding(21, emb_dim // 2)
        
        # 최종 융합 헤드
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x, ref_idx, mut_idx):
        # 1. 변이 임베딩 생성
        r_emb = self.ref_emb(ref_idx)
        m_emb = self.mut_emb(mut_idx)
        mut_context = torch.cat([r_emb, m_emb], dim=1) # [B, emb_dim]

        # 2. 스테이지 1 통과 (구조 정보)
        x = self.stage1(x)

        # 3. [Option 2 적용] 변이 정보를 기반으로 구조 피처 필터링
        # m_emb (64차원)를 Query로 사용하여 64채널의 x를 Attention
        x = self.mut_mid_attn(x, m_emb)

        # 4. 스테이지 2 통과
        x = self.stage2(x)
        struct_feat = self.pool(self.coordatt(x)).flatten(1)

        # 5. 최종 결합 (Late Fusion)
        combined = torch.cat([struct_feat, mut_context], dim=1)
        return self.fusion(combined)
