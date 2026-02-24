import torch
import torch.nn as nn
from mamba_ssm import Mamba

# --- MSA (Sequence) Branch Components ---

class MSAInputEmbedding(nn.Module):
    """
    Converts raw MSA indices into continuous embedding space.
    Input: (B, L, D) -> Output: (B, L, D, C)
    """
    def __init__(self, vocab_size=21, dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)

    def forward(self, x): 
        return self.embedding(x)  


class CrossAxialMambaMSA(nn.Module):
    """
    Hybrid block processing two axes:
    1. L-axis: Sequential context via Mamba (processed in parallel for all depths).
    2. D-axis: Evolutionary context via 1D-Conv (at the center mutation position).
    """
    def __init__(self, dim):
        super().__init__()
        self.norm_L = nn.RMSNorm(dim, eps=1e-8)
        self.norm_D = nn.RMSNorm(dim, eps=1e-8)

        self.mamba_L = Mamba(d_model=dim, expand=1)
        self.conv_D = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.SiLU()
        )

    def forward(self, x):  # x: (B, L, D, C)
        B, L, D, C = x.shape
        center_L = L // 2  # Focus on the mutation site
        
        # --- D-axis processing: Focused on the mutation center ---
        # Extracts the column at center_L across all MSA depths# --- D-axis: only at center L position ---
        x_d_center = self.norm_D(x[:, center_L])  # (B, D, C)
        x_d = x_d_center.transpose(1, 2)          # (B, C, D) for Conv1d
        d_out = self.conv_D(x_d).transpose(1, 2).unsqueeze(1)  # (B, 1, D, C)

        # --- L-axis processing: Full sequence scan via Mamba ---
        # Treats (B*D) as batch to process each sequence in the MSA independently
        x_l = self.norm_L(x).permute(0, 2, 1, 3).contiguous().view(B * D, L, C)
        l_out = self.mamba_L(x_l).view(B, D, L, C).permute(0, 2, 1, 3)  # (B, L, D, C)

        # --- Residual Connection with Sparse D-axis Update ---
        d_full = torch.zeros_like(x)
        d_full[:, center_L:center_L+1] = d_out

        return x + d_full + l_out
    
class MSAEncoder(nn.Module):
    """Stack of Cross-Axial Mamba blocks for evolutionary feature extraction."""
    def __init__(self, num_layers=8, dim=256):
        super().__init__()
        self.embeddings = MSAInputEmbedding(dim=dim)
        self.blocks = nn.ModuleList([
            CrossAxialMambaMSA(dim) for _ in range(num_layers)
        ])
        self.norm_f = nn.RMSNorm(dim, eps=1e-8)

    def forward(self, x):  
        x = self.embeddings(x)  
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)  

class CenterAwarePooling(nn.Module):
    """
    Uses Multi-head Attention to aggregate global MSA context 
    using the center mutation site as the Query.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):  # x: (B, L, D, C)
        B, L, D, C = x.shape
        center_L = L // 2

        # Query: center residue profile [B*D, 1, C]
        query = x[:, center_L].reshape(B * D, 1, C)
        # Key/Value: entire window [B*D, L, C]
        keyval = x.permute(0, 2, 1, 3).reshape(B * D, L, C)

        attn_out, _ = self.attn(query, keyval, keyval)   # (B*D, 1, C)
        # Average across depth (D) axis to get a single vector per batch
        pooled = attn_out.view(B, D, C).mean(dim=1)
        return pooled
    
class MSABranch(nn.Module):
    """Complete MSA processing branch."""
    def __init__(self, num_layers=4, dim=128):
        super().__init__()
        self.encoder = MSAEncoder(num_layers=num_layers, dim=dim)
        self.pooling = CenterAwarePooling(dim, num_heads=4)
        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU()
        )

    def forward(self, x):  
        x = self.encoder(x)        
        x = self.pooling(x)        
        return self.refine(x)     