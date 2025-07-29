import pandas as pd
import os
import json
import prody as pr

# 경로
tsv_path = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\rhapsody2_sav_db.tsv"
json_path = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\UniProtID_to_seq.json"
pdb_dir = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\alphafold_structures"
filtered_output_path = os.path.join(r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset", "rhapsody2_sav_db_exactmatch_only.tsv")

# aa mapping
aa_names = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# PDB sequence 추출 함수  
def get_pdb_sequence(pdb_path):
    structure = pr.parsePDB(pdb_path, subset='ca')
    resnames = structure.getResnames()
    return ''.join([aa_names.get(res, 'X') for res in resnames])

# 데이터 불러오기
df = pd.read_csv(tsv_path, sep="\t", header=None)
df.columns = ["UniProtID", "StructureFile", "MutPos", "WT", "Mut", "Label"]

with open(json_path) as f:
    seq_dict = json.load(f)

# 조건 검사
keep_flags = []
for idx, row in df.iterrows():
    uid = row["UniProtID"]
    fasta_seq = seq_dict.get(uid, None)
    pdb_file_path = os.path.join(pdb_dir, row["StructureFile"])

    if fasta_seq is None:
        keep_flags.append(False)
        continue

    try:
        pdb_seq = get_pdb_sequence(pdb_file_path)
    except Exception as e:
        keep_flags.append(False)
        continue

    keep_flags.append(fasta_seq == pdb_seq)

# 필터링 및 저장
filtered_df = df[keep_flags]
filtered_df.to_csv(filtered_output_path, sep="\t", index=False)
print(f"✅ 완전 일치하는 {len(filtered_df)}개 row 저장 완료: {filtered_output_path}")
