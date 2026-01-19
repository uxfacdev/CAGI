import pandas as pd

def filter_forward_by_consensus(input_path, output_path):
    # 1. 데이터 로드
    df = pd.read_csv(input_path)
    
    print(f"초기 데이터 개수: {len(df)}")

    # 2. PDB ID별로 가장 많이 등장하는 wt_seq(컨센서스) 찾기
    # 진짜 야생형은 여러 변이의 기준이 되므로 빈도가 가장 높음
    consensus_map = df.groupby('pdb_id')['wt_seq'].agg(lambda x: x.value_counts().index[0])
    
    # 3. 각 행의 wt_seq가 해당 PDB ID의 컨센서스 서열과 일치하는지 확인
    def is_consensus_wt(row):
        pdb_id = row['pdb_id']
        current_wt = row['wt_seq']
        return current_wt == consensus_map[pdb_id]

    # 4. 필터링 수행 (순방향 데이터만 남김)
    df_forward = df[df.apply(is_consensus_wt, axis=1)].copy()
    
    print(f"필터링 후(순방향) 데이터 개수: {len(df_forward)}")
    print(f"제거된 역방향 데이터 개수: {len(df) - len(df_forward)}")

    # 5. 결과 저장
    df_forward.to_csv(output_path, index=False)
    return df_forward

# --- 실행 부분 ---
input_csv = r"C:\Users\Kunny\Downloads\PROSTATA-main\PROSTATA_EXPERIMENTS\train_prostata_test_s669.s669_r.s669_dssp\train.csv"
output_csv = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\Stability\PROSTATA_forward_only_internal.csv"

filter_forward_by_consensus(input_csv, output_csv)