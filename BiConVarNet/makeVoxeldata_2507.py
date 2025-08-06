from voxel_utils import compute_atomic_distance_profile
import os
import numpy as np
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 경로 설정
tsv_path = r"C:\Users\Kunny\Research\Dataset\Missense_Variant_dataset\rhapsody2_sav_db_exactmatch_only.tsv"
pdb_dir = r"C:\Users\Kunny\Research\Dataset\Missense_Variant_dataset\alphafold_structures"
output_dir = r"C:\Users\Kunny\Research\Dataset\Missense_Variant_dataset\voxel_cache"

# === 병렬 함수는 반드시 전역으로 선언 ===
def voxelize_and_save(row):
    uid, structure_file, mut_pos, label = row["UniProtID"], row["StructureFile"], int(row["MutPos"]), int(row["Label"])
    key = f"{uid}_{mut_pos}"
    pdb_path = os.path.join(pdb_dir, structure_file)
    output_path = os.path.join(output_dir, f"{key}.pkl")
    
    try:
        feature_grid, mapping_grid = compute_atomic_distance_profile(pdb_path, mut_pos)
        with open(output_path, "wb") as f:
            pickle.dump({
                "feature": feature_grid.astype(np.float32),
                "mapping": mapping_grid.astype(np.int32),
                "label": label,
                "uid": uid,
                "mut_pos": mut_pos,
                "structure_file": structure_file,
            }, f)
        return key, True
    except Exception as e:
        return key, f"{type(e).__name__}: {e}"

# === 반드시 __main__ 아래에서 실행 ===
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 불러오기
    df = pd.read_csv(tsv_path, sep="\t", header=None)
    df.columns = ["UniProtID", "StructureFile", "MutPos", "WT", "Mut", "Label"]
    rows = df.to_dict(orient="records")

    # 병렬 실행
    results = []
    with ProcessPoolExecutor(max_workers=20) as executor:
        for result in tqdm(executor.map(voxelize_and_save, rows), total=len(rows)):
            results.append(result)

    # 실패 로그 저장
    failed = [(key, error) for key, error in results if error is not True]
    if failed:
        fail_log_path = os.path.join(output_dir, "voxelization_failed_log.csv")
        pd.DataFrame(failed, columns=["Key", "Error"]).to_csv(fail_log_path, index=False)
        print(f"❌ 실패한 {len(failed)}개 항목이 {fail_log_path}에 저장되었습니다.")
    else:
        print("✅ 모든 voxelization 성공.")