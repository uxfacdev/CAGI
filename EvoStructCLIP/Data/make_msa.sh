#!/bin/bash

# 파라미터 설정
INPUT_DIR="fasta_files_all_missense"
MERGED_FASTA="merged_all.fasta"
DB_NAME="merged_all_db"
TARGET_DB="uniref90_mmseqs"
RESULT_NAME="merged_result_all"
TMP_DIR="merged_tmp_all"
MSA_OUTPUT="merged_msa_a3m_all"
THREADS=36

echo "Step 1: Merging FASTA files..."
cat ${INPUT_DIR}/*.fasta > ${MERGED_FASTA}
COUNT=$(grep -c "^>" ${MERGED_FASTA})
echo "Total sequences merged: $COUNT"

echo "Step 2: Creating MMseqs2 Database..."
mmseqs createdb ${MERGED_FASTA} ${DB_NAME}

echo "Step 3: Searching against UniRef90..."
mmseqs search ${DB_NAME} ${TARGET_DB} ${RESULT_NAME} ${TMP_DIR} \
  --threads ${THREADS} -e 0.001 --max-seqs 500

echo "Step 4: Generating MSA (a3m)..."
mmseqs result2msa ${DB_NAME} ${TARGET_DB} ${RESULT_NAME} ${MSA_OUTPUT} \
  --msa-format-mode 5 \
  --threads ${THREADS} \
  --max-seq-id 0.95 \
  --qid 0.3 \
  --cov 0.3 \
  --filter-msa 1 \
  --filter-min-enable 100 \
  --diff 500

echo "Process Complete!"