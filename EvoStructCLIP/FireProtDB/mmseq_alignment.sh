# 1. 훈련 데이터 통합
cat Mega.fasta notMega.fasta > Train_combined.fasta

# 2. 유사도 검색 (Identity 25% 기준)
# --min-seq-id 0.25: 유사도 25% 이상만 출력
# -c 0.8: 서열의 80% 이상이 겹치는 경우 (Coverage)
mmseqs easy-search S669_DDGemb.fasta Train_combined.fasta result_25.m8 tmp --min-seq-id 0.25 -c 0.8 --cov-mode 0