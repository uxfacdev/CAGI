import pandas as pd
import os
import json
import prody as pr

# 경로
tsv_path = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\rhapsody2_sav_db.tsv"
json_path = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\UniProtID_to_seq.json"
pdb_dir = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\alphafold_structures"

# 데이터 로드
df = pd.read_csv(r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\rhapsody2_sav_db.tsv", sep="\t", header=None)
df.columns = ["UniProtID", "StructureFile", "MutPos", "WT", "Mut", "Label"]

with open(json_path) as f:
    uniprot_seq_dict = json.load(f)

# AA name map
aa_names = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# 통계 변수
fasta_match = 0
pdb_match = 0
total = 0
pdb_fail = 0

# 빠르게 테스트 (100개)
for idx, row in df.sample(12094, random_state=42).iterrows():
    uid = row["UniProtID"]
    pdb_file = os.path.join(pdb_dir, row["StructureFile"])
    mut_pos = int(row["MutPos"])
    wt = row["WT"]

    # FASTA 확인
    seq = uniprot_seq_dict.get(uid, "")
    fasta_residue = seq[mut_pos - 1] if 1 <= mut_pos <= len(seq) else "?"
    if fasta_residue == wt:
        fasta_match += 1

    # PDB 확인
    try:
        structure = pr.parsePDB(pdb_file, subset='ca')
        resnames = structure.getResnames()
        resname = resnames[mut_pos - 1] if mut_pos - 1 < len(resnames) else "???"
        pdb_residue = aa_names.get(resname, "?")
        if pdb_residue == wt:
            pdb_match += 1
    except:
        pdb_fail += 1
        pdb_residue = "ERR"

    total += 1

# 결과 출력
print(f"총 샘플 수: {total}")
print(f"FASTA 기준 일치 수: {fasta_match} ({fasta_match / total:.2%})")
print(f"PDB 기준 일치 수:   {pdb_match} ({pdb_match / total:.2%})")
print(f"PDB 파싱 실패 수:    {pdb_fail}")
