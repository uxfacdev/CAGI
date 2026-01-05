import os
import pickle
import torch
from torch.utils.data import Dataset
import random

# Global mapping for amino acids to indices
# Standard 20 AAs, '-' for gap/padding, 'X' for unknown
AA_TO_INDEX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    '-': 20, 'X': 20
}

def random_voxel_rotate(voxel):
    """
    Applies random 90-degree rotations to the 3D voxel tensor.
    Args:
        voxel: Tensor of shape [C, D, H, W]
    Returns:
        Rotated voxel tensor
    """
    if random.random() < 0.5:
        # Define planes for rotation: (Height, Width), (Depth, Width), (Depth, Height)
        axes = [(2, 3), (1, 3), (1, 2)]
        k = random.choice([1, 2, 3])  # Number of 90-degree rotations
        axis = random.choice(axes)
        voxel = torch.rot90(voxel, k=k, dims=axis)
    return voxel

def random_voxel_flip(voxel):
    """
    Applies random flips along the Depth, Height, and Width axes.
    Args:
        voxel: Tensor of shape [C, D, H, W]
    Returns:
        Flipped voxel tensor
    """
    if random.random() < 0.5:
        voxel = torch.flip(voxel, dims=[1])  # Flip D-axis
    if random.random() < 0.5:
        voxel = torch.flip(voxel, dims=[2])  # Flip H-axis
    if random.random() < 0.5:
        voxel = torch.flip(voxel, dims=[3])  # Flip W-axis
    return voxel

class VoxelDataset(Dataset):
    """
    Dataset for loading 3D Voxel representations of protein mutation sites.
    """
    def __init__(self, df, voxel_cache_dir, aug=False):
        self.df = df.reset_index(drop=True)
        self.voxel_cache_dir = voxel_cache_dir
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["UniProtID"]
        mut_pos = row["MutPos(pdb)"]
        wt = row["WT"]
        mut = row["Mut"]
        label = row["Label"]

        # Load pre-computed voxel data from pickle
        key = f"{uid}_{mut_pos}"
        voxel_path = os.path.join(self.voxel_cache_dir, f"{key}.pkl")

        with open(voxel_path, "rb") as f:
            data = pickle.load(f)
            # Original shape usually: (1, 7, 7, 7, 46)
            feature = data["feature"]

        # Convert to tensor and rearrange to (Channels, Depth, Height, Width)
        feature_tensor = torch.from_numpy(feature).permute(0, 4, 1, 2, 3).float().squeeze(0)

        # Apply spatial augmentations if enabled
        if self.aug:
            feature_tensor = random_voxel_rotate(feature_tensor)
            feature_tensor = random_voxel_flip(feature_tensor)

        # Map amino acids to indices for embedding layers
        ref_idx = torch.tensor(AA_TO_INDEX.get(str(wt), 20), dtype=torch.long)
        mut_idx = torch.tensor(AA_TO_INDEX.get(str(mut), 20), dtype=torch.long)

        return feature_tensor, ref_idx, mut_idx, torch.tensor(label).long()
    
class MSADataset(Dataset):
    """
    Dataset for Multiple Sequence Alignment (MSA) centered around a mutation site.
    """
    def __init__(self, df, msa_dict_path, max_depth=80, win_size=61, aug=False):
        with open(msa_dict_path, "rb") as f:
            self.msa_dict = pickle.load(f)
        self.df = df
        self.max_depth = max_depth
        self.win_size = win_size
        self.half_win = win_size // 2  
        self.aug = aug
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["UniProtID"]
        mut_pos = int(row["MutPos"]) - 1  # Convert 1-based indexing to 0-based
        label = int(row["Label"])
        mut = row["Mut"].upper()

        # Extract MSA sequences for the given protein
        msa_seqs = [seq for _, seq in self.msa_dict[uid][:self.max_depth]]
        query_seq = msa_seqs[0]  # The first sequence is the Wild-Type query

        # Create the mutant sequence by replacing the AA at the mutation position
        mut_seq = list(query_seq)
        if 0 <= mut_pos < len(mut_seq):
            mut_seq[mut_pos] = mut

        # Construct sequence list: [Mutant, Wild-Type, ...MSA sequences]
        seqs_to_use = [mut_seq, list(query_seq)]
        if len(msa_seqs) > 1:
            seqs_to_use += [list(seq) for seq in msa_seqs[1:self.max_depth - 2]]

        # Optional augmentation: shuffle the order of sequences in the MSA (excluding Mut and WT)
        if self.aug:
            msa_part = seqs_to_use[2:]  
            random.shuffle(msa_part)  
            seqs_to_use = seqs_to_use[:2] + msa_part

        # Slice a window centered at the mutation position
        centered_msa = []
        for seq in seqs_to_use:
            window = []
            for i in range(self.win_size):
                seq_idx = mut_pos - self.half_win + i
                if 0 <= seq_idx < len(seq):
                    aa = seq[seq_idx]
                else:
                    aa = '-' # Padding for out-of-bounds
                window.append(AA_TO_INDEX.get(aa, 20))
            centered_msa.append(window)

        # Padding for MSA depth to ensure fixed tensor size [max_depth, win_size]
        while len(centered_msa) < self.max_depth:
            centered_msa.append([20] * self.win_size)  

        # Convert to tensor and transpose to [Length, Depth]
        msa_tensor = torch.tensor(centered_msa[:self.max_depth]).long()  # [D, L]
        msa_tensor = msa_tensor.transpose(0, 1)  # [L, D]

        return {
            "msa": msa_tensor, 
            "label": torch.tensor(label).long()
        }

class MultimodalDataset(Dataset):
    """
    Combined Dataset that yields both Voxel and MSA features for a mutation.
    """
    def __init__(self, df, voxel_cache_dir, msa_dict_path, 
                 voxel_aug=False, msa_aug=False, max_depth=80, win_size=61):
        self.df = df.reset_index(drop=True)
        self.voxel_dataset = VoxelDataset(df, voxel_cache_dir, aug=voxel_aug)
        self.msa_dataset = MSADataset(df, msa_dict_path, max_depth=max_depth, win_size=win_size, aug=msa_aug)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve data from sub-datasets
        voxel_feat, ref_idx, mut_idx, label = self.voxel_dataset[idx]
        msa_data = self.msa_dataset[idx]  
        msa_tensor = msa_data["msa"]
        
        # Safety check to ensure both datasets are synchronized by index
        assert label == msa_data["label"], "Mismatch in label between Voxel and MSA datasets!"

        return {
            "voxel": voxel_feat,      # [46, 7, 7, 7]
            "ref_idx": ref_idx,       # Scalar
            "mut_idx": mut_idx,       # Scalar
            "msa": msa_tensor,        # [MSA Length, MSA Depth]
            "label": label            # Scalar
        }