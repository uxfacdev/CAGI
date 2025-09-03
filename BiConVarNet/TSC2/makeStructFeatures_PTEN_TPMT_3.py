# === struct_S_precompute.py (Jupyter/Windows에서 실행) ===
import os
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
from extract_struct_features_more_2 import extract_struct_S, load_ddg_map  
from itertools import repeat

# ---- 사용자 경로 설정 ----
tsv_path   = r"C:\Users\Kunny\Research\Project\BiConVarNet\TSC2\PTEN_TPMT_VAMPseq_combined_filtering.tsv"
pdb_dir    = r"E:\CAGI_data\pdb_files"
out_dir    = r"C:\Users\Kunny\Research\Dataset\Missense_Variant_dataset\struct_feature_cache_PTEN_TPMT_3"

# (선택) 체인 매핑: 기본은 AlphaFold 체인 'A'
DEFAULT_CHAIN = "A"

# ---- ΔΔG_fold 맵 로드 (없으면 빈 맵) ----
ddg_maps = {
    "P60484": load_ddg_map(r"C:\Users\Kunny\Research\Dataset\Missense_Variant_dataset\evoef2_out_P60484\ddg_fold.tsv"),
    "P51580": load_ddg_map(r"C:\Users\Kunny\Research\Dataset\Missense_Variant_dataset\evoef2_out_P51580\ddg_fold.tsv"),
}

# ---- 유틸: 파일 키 생성 (충돌 방지: WT/Mut 포함) ----
def make_key(uid: str, pos: int, wt: str, mut: str) -> str:
    return f"{uid}_{int(pos)}_{wt.upper()}{mut.upper()}"

# ---- 워커: 한 행 처리해서 캐시에 저장 ----
def build_structS_row(row, msa_dict) -> tuple[str, str]:
    """
    입력 row(dict): TSV 한 줄
    반환: (key, status)
    """
    uid = str(row["UniProtID"]).strip()
    mut_pos = int(row["MutPos"])
    wt = str(row["WT"]).strip().upper()
    mut = str(row["Mut"]).strip().upper()
    label = float(row["Label"]) if "Label" in row and row["Label"] != "" else np.nan
    structure_file = str(row["StructureFile"]).strip()

    key = make_key(uid, mut_pos, wt, mut)
    out_path = os.path.join(out_dir, f"{key}.pkl")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return key, "Skipped"

    pdb_path = os.path.join(pdb_dir, structure_file)
    if not os.path.exists(pdb_path):
        return key, f"Failed: PDB not found: {pdb_path}"

    try:
        # S 추출 (Energetics는 ddg_map으로 채움; 없으면 NaN)
        S, names = extract_struct_S(
            pdb_path=pdb_path,
            mut_pos=mut_pos,
            wt1=wt, mut1=mut,
            chain_id=DEFAULT_CHAIN,
            ddg_map_fold=ddg_maps,
            uid=uid,
            msa_dict=msa_dict
        )

        payload = {
            "S": np.asarray(S, dtype=np.float32)
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        return key, "Success"
    except Exception as e:
        # 에러 상세 기록(디버깅용)
        tb = traceback.format_exc(limit=2)
        return key, f"Failed: {type(e).__name__}: {e} | {tb}"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def precompute_msa_freqs(msa_dict):
    precomp = {}
    for uid, seqs in msa_dict.items():
        if not seqs:
            continue
        L = len(seqs[0][1])  # query sequence length
        freqs_per_pos = []
        for pos in range(L):
            counts = {aa:0 for aa in AA_LIST}
            for _, seq in seqs:
                if pos < len(seq):
                    aa = seq[pos]
                    if aa in counts:
                        counts[aa] += 1
            total = sum(counts.values())
            freqs_per_pos.append((counts, total))
        precomp[uid] = freqs_per_pos
    return precomp

# ---- 실행부 ----
if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(tsv_path, sep="\t")
    rows = df.to_dict(orient="records")

    with open(r"E:/CAGI_data/msa_dict_valid.pkl", "rb") as f:
        msa_dict = pickle.load(f)

    uids = set(df["UniProtID"].astype(str))
    msa_precomp = precompute_msa_freqs({uid: msa_dict[uid] for uid in uids if uid in msa_dict})

    results = []
    # Windows/Jupyter에서도 안전한 수준으로 워커 수 설정 (필요시 늘려도 됨)
    with ProcessPoolExecutor(max_workers=20) as ex:
        for res in tqdm(ex.map(build_structS_row, rows, repeat(msa_precomp)), total=len(rows)):
            results.append(res)

    # 로그 저장
    log_items = [(k, s) for (k, s) in results if s != "Success"]
    if log_items:
        log_path = os.path.join(out_dir, "structS_build_log.csv")
        pd.DataFrame(log_items, columns=["Key", "Status"]).to_csv(log_path, index=False)
        n_fail = sum(1 for _, s in log_items if s.startswith("Failed"))
        n_skip = sum(1 for _, s in log_items if s == "Skipped")
        print(f"❌ 실패: {n_fail}개, ⏭ 스킵: {n_skip}개 → {log_path}")
    else:
        print("✅ 모든 struct-S 캐시 생성 성공.")
