import pandas as pd
import os
import prody as pr
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 경로 설정
pdb_dir = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\FireProtDB\pdb_files"
df_path = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\FireProtDB\FireProt_ssym_train.tsv"
fail_log_path = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\FireProtDB\train_failed_fasta_match.tsv"
fasta_path = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\FireProtDB\Train_combined.fasta"

# AA name map
aa_names = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# 전역 변수로 FASTA 데이터를 저장 (병렬 처리 시 참조)
fasta_dict = {}

def load_fasta(path):
    """FASTA 파일을 읽어 {ID: Sequence} 딕셔너리로 반환"""
    seqs = {}
    if not os.path.exists(path):
        return seqs
    with open(path, 'r') as f:
        current_id = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:].split()[0] # 첫 번째 공백 이전까지만 ID로 사용
                seqs[current_id] = ""
            else:
                seqs[current_id] += line
    return seqs

print("FASTA 파일 로딩 중...")
fasta_dict = load_fasta(fasta_path)
print(f"FASTA 로드 완료 ({len(fasta_dict)} 개 서열)")

def check_row(row):
    # 기본 정보 추출
    uid = row["Key"]
    structure_file = row["FILE_PATH"]
    mut_pos = int(row["SeqPos"])-1
    mut_pos_pdb = int(row["MutPos"]) 
    wt = row["WT"]
    chain = str(row["Chain"])

    # 1. FASTA 서열 체크 (추가된 부분)
    fasta_match = False
    fasta_found_aa = "?"
    fasta_error_reason = None

    sequence = fasta_dict.get(uid)
    if sequence:
        try:
            # mut_pos가 서열 인덱스 범위를 벗어나지 않는지 확인
            if mut_pos < len(sequence):
                fasta_found_aa = sequence[mut_pos]
                fasta_match = (fasta_found_aa == wt)
                if not fasta_match:
                    fasta_error_reason = f"FASTA mismatch (Expected {wt}, Found {fasta_found_aa})"
            else:
                fasta_error_reason = "FASTA Index out of range"
        except Exception as e:
            fasta_error_reason = f"FASTA processing error: {str(e)}"
    else:
        fasta_error_reason = "ID not found in FASTA"

    # 2. PDB 파일 로드 및 체크 (기존 로직 유지)
    pdb_file = os.path.join(pdb_dir, structure_file)
    pdb_match = False
    pdb_error = False
    pdb_fail_reason = None
    pdb_residue = "?"

    try:
        structure = pr.parsePDB(pdb_file)
        selection_str = f"chain {chain} and resnum {mut_pos_pdb} and ca"
        selection = structure.select(selection_str)

        if selection is not None:
            resname = selection.getResnames()[0]
            pdb_residue = aa_names.get(resname, "?")
            pdb_match = (pdb_residue == wt)
        else:
            pdb_fail_reason = "Selection empty (Chain/Resnum not found)"
    except Exception as e:
        pdb_error = True
        pdb_fail_reason = f"PDB parsing error: {str(e)}"

    if not pdb_error and pdb_fail_reason is None and not pdb_match:
        pdb_fail_reason = f"PDB mismatch (Expected {wt}, Found {pdb_residue})"

    # 최종 실패 여부 결정
    is_failed = (not fasta_match) or (not pdb_match) or pdb_error

    return {
        "is_failed": is_failed,
        "fasta_match": fasta_match,
        "fasta_error": fasta_error_reason,
        "pdb_match": pdb_match,
        "pdb_error": pdb_error,
        "pdb_fail_reason": pdb_fail_reason,
        "UniProtID": uid,
        "MutPos": mut_pos,
        "WT": wt,
        "FASTA_Found_AA": fasta_found_aa,
        "PDB_Found_AA": pdb_residue,
        "StructureFile": structure_file,
        "Chain": chain,
        "MutPos(pdb)": mut_pos_pdb
    }

if __name__ == "__main__":
    # 1. 데이터 로드 및 FASTA 사전 로드
    df = pd.read_csv(df_path, sep="\t")

    # 2. 병렬 실행
    with ProcessPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(check_row, [row for _, row in df.iterrows()]), total=len(df)))

    # 3. 통계 계산
    total = len(results)
    fasta_matches = sum(r["fasta_match"] for r in results)
    pdb_matches = sum(r["pdb_match"] for r in results)
    total_failures = sum(1 for r in results if r["is_failed"])

    # 4. 결과 출력
    print(f"\n--- 최종 검증 결과 ---")
    print(f"총 샘플 수: {total}")
    print(f"FASTA 일치:  {fasta_matches} ({fasta_matches / total:.2%})")
    print(f"PDB 일치:    {pdb_matches} ({pdb_matches / total:.2%})")
    print(f"최종 불일치(둘 중 하나라도 실패): {total_failures}건")

    # 5. 실패 로그 저장 (FASTA 에러와 PDB 에러 모두 포함)
    failures_df = pd.DataFrame([r for r in results if r["is_failed"]])
    failures_df.to_csv(fail_log_path, sep="\t", index=False)
    print(f"상세 실패 로그 저장 완료: {fail_log_path}")