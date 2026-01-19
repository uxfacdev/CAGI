import pandas as pd
from pathlib import Path

import pandas as pd

def transform_prostata_data(input_path, output_csv_path, output_fasta_path):
    # 원본 데이터 읽기 (pos 컬럼이 포함되어 있음)
    df = pd.read_csv(input_path)
    
    # 1. CSV 저장: 'pos' 컬럼을 반드시 포함
    # 'pos'는 서열의 0-based index입니다.
    new_df = pd.DataFrame({
        'PDB': df['pdb_id'],
        'WT': df['mut_info'].str[0],
        'MAPPED_PDB_POS': df['mut_info'].str[1:-1], # PDB상의 번호 (참고용)
        'MT': df['mut_info'].str[-1],
        'DDG': df['ddg'],
        'POS': df['pos']+1 # 실제 서열상의 인덱스 (검증용/학습용 핵심 데이터)
    })
    new_df.to_csv(output_csv_path, index=False, sep='\t')
    
    # 2. FASTA 저장 (기존과 동일)
    unique_seqs = df.drop_duplicates(subset=['pdb_id'])
    with open(output_fasta_path, 'w') as f:
        for _, row in unique_seqs.iterrows():
            f.write(f">{row['pdb_id']}\n{row['wt_seq']}\n")


# --- 실행 부분 ---
# 실제 파일 경로에 맞춰 수정해서 사용하세요.
input_file = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\Stability\PROSTATA_cleaned.csv"
output_csv = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\Stability\PROSTATA.csv"
output_fasta = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\Stability\PROSTATA.fasta"


transform_prostata_data(input_file, output_csv, output_fasta)
