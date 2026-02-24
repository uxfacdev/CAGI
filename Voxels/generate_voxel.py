import argparse
import os
import numpy as np
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
from Voxels.voxel_utils import compute_voxel_features_structural

def voxelize_and_save(row, pdb_dir, output_dir):
    uid  = row["PDB"]
    wt1  = str(row["WT"]).strip().upper()
    mut1 = str(row.get("MT", "")).strip().upper()
    structure_file = str(row["MAPPED_PDB"]).lower() + ".pdb"
    mut_pos = int(row["MAPPED_PDB_POS"])
    label   = int(row["DDG"])
    target_chain = row["MAPPED_CHAIN"] 

    key = f"{uid}_{mut_pos}"
    pdb_path = os.path.join(pdb_dir, structure_file)
    out_path = os.path.join(output_dir, f"{key}.pkl")

    # Skip if file already exists and is not empty
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return key, "Skipped", None

    try:
        # Extract features (WT passed to check for mismatch internally)
        feat, mapping, qc = compute_voxel_features_structural(
            pdb_path, mut_pos,
            chain_id=target_chain,                    
            add_dpos2=True, add_plddt=False, add_gnm_msf=True,
            wt1=wt1                                   
        )

        with open(out_path, "wb") as f:
            pickle.dump({
                "feature": feat.astype(np.float32),
                "mapping": mapping.astype(np.int32),
                "label": label,
                "uid": uid,
                "mut_pos": mut_pos,
                "structure_file": structure_file,
            }, f)

        # Log QC notes if any
        if qc["notes"]:
            qc_row = {
                "Key": key, "UniProtID": uid, "StructureFile": structure_file,
                "MutPos(pdb)": mut_pos, "WT_expected": qc["wt_expected"],
                "Obs_CA_3": qc["obs_center_aa3"], "Obs_CA_1": qc["obs_center_aa1"],
                "CenterMapOK": qc["center_maps_to_mut_pos"], "Notes": "; ".join(qc["notes"])
            }
            return key, "Success_with_QC", qc_row
        else:
            return key, "Success", None

    except Exception as e:
        return key, f"Failed: {type(e).__name__}: {e}", None

def main():
    parser = argparse.ArgumentParser(description="Generate 3D Voxel representations from PDB/AlphaFold structures.")
    parser.add_argument("--tsv", required=True, help="Path to input TSV file")
    parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output voxel .pkl files")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for parallel processing")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading TSV from: {args.tsv}")
    df = pd.read_csv(args.tsv, sep="\t")
    rows = df.to_dict(orient="records")

    results, qc_rows = [], []
    
    # Use partial to pass extra arguments to the target function
    worker_func = partial(voxelize_and_save, pdb_dir=args.pdb_dir, output_dir=args.output_dir)

    print(f"Starting voxelization with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for key, status, qc_row in tqdm(ex.map(worker_func, rows), total=len(rows)):
            results.append((key, status))
            if qc_row is not None:
                qc_rows.append(qc_row)

    # Save QC logs if there are mismatches or warnings
    if qc_rows:
        qc_path = os.path.join(args.output_dir, "voxel_qc_log.csv")
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
        num_qc = len(qc_rows)
        print(f"[Warning] {num_qc} QC issues found. Details saved to: {qc_path}")

    num_failed  = sum(1 for _, s in results if str(s).startswith("Failed"))
    num_skipped = sum(1 for _, s in results if s == "Skipped")
    print(f"Process Complete. Failed: {num_failed} | Skipped: {num_skipped} | Total processed: {len(results)}")

if __name__ == "__main__":
    main()
