import pandas as pd
import json
from tqdm import tqdm

# 경로
json_path = r"C:\Users\Kunny\Research\Dataset\Missense_Variant_dataset\UniProtID_to_seq.json"
df_path = r"E:\CAGI_data\AlphaMissense_variants_passed.tsv"
fail_log_path = r"E:\CAGI_data\failed_fasta_match_2.tsv"

# UniProt 시퀀스 로드
with open(json_path) as f:
    uniprot_seq_dict = json.load(f)

# 데이터 로드
df = pd.read_csv(df_path, sep="\t")

# 각 UniProtID의 시퀀스를 lookup 하는 함수
def get_residue(uid, pos):
    seq = uniprot_seq_dict.get(uid, "")
    if 1 <= pos <= len(seq):
        return seq[pos-1]
    return "?"

# Fasta residue 뽑기 (apply보다 map + zip이 훨씬 빠름)
df["FastaAA"] = [
    get_residue(uid, pos) 
    for uid, pos in tqdm(zip(df["UniProtID"], df["MutPos"]), total=len(df))
]

# 매치 여부
df["fasta_match"] = (df["WT"] == df["FastaAA"])

# 실패한 케이스만 저장
failures = df.loc[~df["fasta_match"], ["UniProtID","MutPos","WT","FastaAA"]]
failures.to_csv(fail_log_path, sep="\t", index=False)

# 통계 출력
total = len(df)
fasta_match = df["fasta_match"].sum()
print(f"총 샘플 수: {total}")
print(f"FASTA 기준 일치 수: {fasta_match} ({fasta_match/total:.2%})")
print(f"불일치 총 {len(failures)}건")
