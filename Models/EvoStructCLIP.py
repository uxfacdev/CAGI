import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MSABranch import MSABranch
from VoxelBranch import VoxelBranch


# --- Main Multimodal Model ---
    
class EvoStructCLIP(nn.Module):
    """
    EvoStructCLIP: A Multimodal framework for Variant Effect Prediction.
    Integrates evolutionary information (MSA) and 3D structural context (Voxel).
    Uses Contrastive Learning (CLIP-style) and Classification.
    """
    def __init__(self, voxel_ch=63, mb_layers=8, embed_dim=128, use_concat=True):
        super().__init__()
        self.voxel_encoder = VoxelBranch(in_ch=voxel_ch, emb_dim=embed_dim)
        self.msa_encoder = MSABranch(num_layers=mb_layers, dim=embed_dim)
        self.use_concat = use_concat

        # CLIP temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        fused_dim = embed_dim * 2 if use_concat else embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, voxel, ref_idx, mut_idx, msa):
        # 1. Extract features from both modalities
        voxel_feat = self.voxel_encoder(voxel, ref_idx, mut_idx) # (B, 128)
        msa_feat = self.msa_encoder(msa)                       # (B, 128)

        # 2. Compute Contrastive Logits (CLIP-style)
        v_norm = F.normalize(voxel_feat, dim=-1, eps=1e-8)
        m_norm = F.normalize(msa_feat, dim=-1, eps=1e-8)

        scale = self.logit_scale.exp()
        logits_per_voxel = torch.matmul(v_norm, m_norm.t()) * scale
        logits_per_msa = logits_per_voxel.t() 

        # 3. Final Classification Output
        fused = torch.cat([voxel_feat, msa_feat], dim=-1) if self.use_concat else voxel_feat + msa_feat
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "logits_per_msa": logits_per_msa,
            "logits_per_voxel": logits_per_voxel,
            "voxel_feat": voxel_feat,
            "msa_feat": msa_feat
        }