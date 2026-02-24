import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from tqdm import tqdm

from Models import EvoStructCLIP
from multimodal_dataset import MultimodalDataset

def get_args():
    parser = argparse.ArgumentParser(description="Train EvoStructCLIP model with multimodal data.")
    
    # Data paths
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--voxel_cache", type=str, required=True)
    parser.add_argument("--msa_dict", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    
    # Hyperparameters
    parser.add_argument("--batch_size_train", type=int, default=88)
    parser.add_argument("--batch_size_val", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--t_max", type=int, default=100)
    parser.add_argument("--mix_alpha", type=float, default=0.4)
    parser.add_argument("--mix_loss_weight", type=float, default=0.6)
    
    return parser.parse_args()

# --- Utility Functions ---

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels)

def compute_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    return (contrastive_loss(similarity) + contrastive_loss(similarity.t())) / 2

def fusemix(voxel_feat, msa_feat, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(voxel_feat.size(0), device=voxel_feat.device)
    voxel_mix = lam * voxel_feat + (1 - lam) * voxel_feat[idx]
    msa_mix = lam * msa_feat + (1 - lam) * msa_feat[idx]
    return voxel_mix, msa_mix

# --- Core Training Logic ---

def train_one_epoch(model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=" >> Training", leave=False)
    for batch in pbar:
        voxel = batch["voxel"].to(device)
        ref_idx = batch["ref_idx"].to(device)
        mut_idx = batch["mut_idx"].to(device)
        msa = batch["msa"].to(device)
        label = batch["label"].float().to(device)

        optimizer.zero_grad()
        out = model(voxel, ref_idx, mut_idx, msa)
        
        cls_loss = criterion(out["logits"].squeeze(-1), label)
        clip_loss_val = compute_clip_loss(out["logits_per_voxel"])

        v_mix, m_mix = fusemix(out["voxel_feat"], out["msa_feat"], args.mix_alpha)
        v_mix_norm = F.normalize(v_mix, dim=-1)
        m_mix_norm = F.normalize(m_mix, dim=-1)
        sim_mix = torch.matmul(v_mix_norm, m_mix_norm.T) * model.logit_scale.exp()
        loss_mix = compute_clip_loss(sim_mix)

        batch_loss = cls_loss + clip_loss_val + (args.mix_loss_weight * loss_mix)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * voxel.size(0)
        pbar.set_postfix({"batch_loss": f"{batch_loss.item():.4f}"})

    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=" >> Validating", leave=False):
            voxel = batch["voxel"].to(device)
            ref_idx = batch["ref_idx"].to(device)
            mut_idx = batch["mut_idx"].to(device)
            msa = batch["msa"].to(device)
            label = batch["label"].float().to(device)

            out = model(voxel, ref_idx, mut_idx, msa)
            logits = out["logits"].squeeze(-1)
            
            val_loss += criterion(logits, label).item() * voxel.size(0)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    return {
        "loss": val_loss / len(loader.dataset),
        "pr_auc": average_precision_score(all_labels, all_probs),  
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "acc": accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_probs])
    }

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*50)
    print("  EvoStructCLIP Training Pipeline")
    print("="*50)
    print(f"  - Device: {device}")
    print(f"  - Max Epochs: {args.epochs}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Batch Size: {args.batch_size_train} (Train) / {args.batch_size_val} (Val)")
    print("-"*50)

    # Load and Split Data
    print(f"[*] Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path, sep="\t")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(df)) 

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    print(f"[+] Data loaded. Train: {len(train_df)} samples, Val: {len(val_df)} samples")

    # Datasets & Loaders
    print("[*] Initializing Datasets and Loaders...")
    train_dataset = MultimodalDataset(train_df, args.voxel_cache, args.msa_dict, voxel_aug=True, msa_aug=True)
    val_dataset = MultimodalDataset(val_df, args.voxel_cache, args.msa_dict)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, 
                              shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, 
                            shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    # Model Setup
    print("[*] Building EvoStructCLIP model...")
    model = EvoStructCLIP(voxel_ch=46, mb_layers=6, embed_dim=128, use_concat=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)

    print("\n[!] Training started.\n")
    best_pr_auc = 0.0
    start_time = time.time()

    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args)
            val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()

            epoch_duration = time.time() - epoch_start
            
            # Epoch Summary Print
            print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}] ({epoch_duration:.1f}s)")
            print(f"  > Loss: Train={train_loss:.4f} | Val={val_metrics['loss']:.4f}")
            print(f"  > Metrics: PR-AUC={val_metrics['pr_auc']:.4f} | ROC-AUC={val_metrics['roc_auc']:.4f} | Acc={val_metrics['acc']:.4f}")

            # Best Model Saving
            if val_metrics['pr_auc'] > best_pr_auc:
                best_pr_auc = val_metrics['pr_auc']
                torch.save(model.state_dict(), args.save_path)
                print(f"  >>> [SAVED] New best PR-AUC achieved!")
            
            print("-" * 30)

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    total_duration = time.time() - start_time
    print(f"\n[+] Training Complete!")
    print(f"Total Time: {total_duration/60:.2f} mins")
    print(f"Best PR-AUC: {best_pr_auc:.4f}")
    print(f"Best model saved at: {args.save_path}")
    print("="*50)

if __name__ == "__main__":
    main()