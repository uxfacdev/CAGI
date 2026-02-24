import argparse
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models import EvoStructCLIP
from multimodal_dataset import MultimodalDataset

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for EvoStructCLIP.")
    
    # Paths
    parser.add_argument("--test_data", type=str, required=True, help="Path to the unlabeled test TSV file.")
    parser.add_argument("--model_path", type=str, 
                        required=True, 
                        help="Path to the trained model (.pth).")
    parser.add_argument("--voxel_cache", type=str, required=True)
    parser.add_argument("--msa_dict", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions.tsv", help="Path to save the results.")
    
    # Inference settings
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification.")
    
    return parser.parse_args()

@torch.no_grad()
def run_inference(model, loader, device, threshold):
    model.eval()
    all_probs = []
    
    print(f"[*] Starting inference on {len(loader.dataset)} samples...")
    for batch in tqdm(loader, desc=" >> Predicting"):
        voxel = batch["voxel"].to(device)
        ref_idx = batch["ref_idx"].to(device)
        mut_idx = batch["mut_idx"].to(device)
        msa = batch["msa"].to(device)

        out = model(voxel, ref_idx, mut_idx, msa)
        logits = out["logits"].squeeze(-1)
        probs = torch.sigmoid(logits)
        
        all_probs.extend(probs.cpu().numpy())
    
    return all_probs

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*50)
    print("  EvoStructCLIP Inference Pipeline")
    print("="*50)
    print(f"  - Device: {device}")
    print(f"  - Model: {args.model_path}")
    print(f"  - Output: {args.output_path}")
    print("-"*50)

    # 1. Load Data
    if not os.path.exists(args.test_data):
        print(f"[!] Error: Test data not found at {args.test_data}")
        return

    df = pd.read_csv(args.test_data, sep="\t")
    
    if "Label" not in df.columns:
        df["Label"] = 0
    
    test_dataset = MultimodalDataset(df, args.voxel_cache, args.msa_dict, voxel_aug=False, msa_aug=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Load Model
    print("[*] Loading model architecture and weights...")
    model = EvoStructCLIP(voxel_ch=46, mb_layers=6, embed_dim=128, use_concat=True).to(device)
    
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("[+] Model weights loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        return

    # 3. Run Inference
    probs = run_inference(model, test_loader, device, args.threshold)

    # 4. Save Results
    print("[*] Saving results...")
    df["Prediction_Prob"] = probs
    df["Predicted_Label"] = [1 if p >= args.threshold else 0 for p in probs]
    
    if "Label" in df.columns and (df["Label"] == 0).all():
        df.drop(columns=["Label"], inplace=True)

    df.to_csv(args.output_path, sep="\t", index=False)
    print(f"[+] Done! Results saved to: {args.output_path}")
    print("="*50)

if __name__ == "__main__":
    main()