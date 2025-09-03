import pandas as pd
import os
import json
import prody as pr
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 경로 설정
json_path = r"C:\Users\Kunny\Research\Dataset\Missense_Variant_dataset\UniProtID_to_seq.json"
pdb_dir = r"E:\CAGI_data\pdb_files"
df_path = r"C:\Users\Kunny\Research\Project\BiConVarNet\TSC2\TSC2_variants_formatted.tsv"

# UniProt 시퀀스 로드
with open(json_path) as f:
    uniprot_seq_dict = json.load(f)

# AA name map
aa_names = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# 단일 row 처리 함수
def check_row(row):
    uid = row["UniProtID"]
    structure_file = row["StructureFile"]
    mut_pos = int(row["MutPos"])
    mut_pos_pdb = int(row["MutPos"]) 
    wt = row["WT"]

    fasta_residue = uniprot_seq_dict.get(uid, "")[mut_pos - 1] if uid in uniprot_seq_dict and 1 <= mut_pos <= len(uniprot_seq_dict[uid]) else "?"

    fasta_match = (fasta_residue == wt)

    pdb_file = os.path.join(pdb_dir, structure_file)
    try:
        structure = pr.parsePDB(pdb_file, subset='ca')
        resnames = structure.getResnames()
        resname = resnames[mut_pos_pdb - 1] if mut_pos_pdb - 1 < len(resnames) else "???"
        pdb_residue = aa_names.get(resname, "?")
        pdb_match = (pdb_residue == wt)
        pdb_error = False
    except:
        pdb_match = False
        pdb_error = True

    fail_reason = None
    if not fasta_match:
        fail_reason = "FASTA mismatch"
    elif pdb_error:
        fail_reason = "PDB parsing error"
    elif not pdb_match:
        fail_reason = "PDB mismatch"

    return {
        "fasta_match": fasta_match,
        "pdb_match": pdb_match,
        "pdb_error": pdb_error,
        "fail_reason": fail_reason,
        "UniProtID": uid,
        "MutPos": mut_pos,
        "WT": wt,
        "StructureFile": structure_file,
        "MutPos(pdb)": mut_pos_pdb
    }

if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv(df_path, sep="\t")

    # 병렬 실행
    with ProcessPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(check_row, [row for _, row in df.iterrows()]), total=len(df)))

    # 통계 요약
    fasta_match = sum(r["fasta_match"] for r in results)
    pdb_match = sum(r["pdb_match"] for r in results)
    pdb_fail = sum(r["pdb_error"] for r in results)
    total = len(results)
    failures = [r for r in results if r["fail_reason"] is not None]

    # 결과 출력
    print(f"총 샘플 수: {total}")
    print(f"FASTA 기준 일치 수: {fasta_match} ({fasta_match / total:.2%})")
    print(f"PDB 기준 일치 수:   {pdb_match} ({pdb_match / total:.2%})")
    print(f"PDB 파싱 실패 수:    {pdb_fail}")
    print(f"불일치 총 {len(failures)}건")

    # 실패 로그 저장
    # pd.DataFrame(failures).to_csv(fail_log_path, sep="\t", index=False)