from voxel_utils_4 import compute_voxel_features_structural
import os, numpy as np, pandas as pd, pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

tsv_path   = r"C:\Users\Kunny\Research\Project\BiConVarNet\TSC2\PTEN_TPMT_VAMPseq_combined.tsv"
pdb_dir    = r"E:\CAGI_data\pdb_files"
output_dir = r"E:\CAGI_data\voxel_cache_4_PTEN_TPMT"
os.makedirs(output_dir, exist_ok=True)

def voxelize_and_save(row):
    uid  = row["UniProtID"]
    wt1  = str(row["WT"]).strip().upper()
    mut1 = str(row.get("Mut","")).strip().upper()
    structure_file = row["StructureFile"]
    mut_pos = int(row["MutPos(pdb)"])
    label   = int(row["Label"])

    key = f"{uid}_{mut_pos}"
    pdb_path = os.path.join(pdb_dir, structure_file)
    out_path = os.path.join(output_dir, f"{key}.pkl")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return key, "Skipped", None

    try:
        feat, mapping, qc = compute_voxel_features_structural(
            pdb_path, mut_pos,
            chain_id=None,                    # 필요 시 지정
            add_dpos2=True, add_plddt=True, add_rsa=True, add_resdepth=True, add_gnm_msf=True,
            wt1=wt1                           # ✅ WT 전달 → 함수 내부에서 미스매치 체크
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

        # QC에서 주의사항이 있으면 돌려보내 로그에 남기자
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

if __name__ == "__main__":
    df = pd.read_csv(tsv_path, sep="\t")
    rows = df.to_dict(orient="records")

    results, qc_rows = [], []
    with ProcessPoolExecutor(max_workers=16) as ex:
        
        for key, status, qc_row in tqdm(ex.map(voxelize_and_save, rows), total=len(rows)):
            results.append((key, status))
            if qc_row is not None:
                qc_rows.append(qc_row)

    # 상태 로그
    log_path = os.path.join(output_dir, "voxelization_log.csv")
    # pd.DataFrame(results, columns=["Key","Status"]).to_csv(log_path, index=False)

    # QC 로그 (미스매치/주의사항만)
    if qc_rows:
        qc_path = os.path.join(output_dir, "voxel_qc_log.csv")
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
        num_qc = len(qc_rows)
        print(f"⚠️ QC 이슈 {num_qc}건 → {qc_path}")

    num_failed  = sum(1 for _, s in results if str(s).startswith("Failed"))
    num_skipped = sum(1 for _, s in results if s == "Skipped")
    print(f"완료. 실패 {num_failed} / 스킵 {num_skipped} / 총 {len(results)}")
