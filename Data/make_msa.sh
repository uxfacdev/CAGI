#!/bin/bash

# Initialize required variables
INPUT_DIR=""
TARGET_DB=""
OUTPUT_PREFIX=""
THREADS=""

# Print usage information
usage() {
    echo "Usage: $0 --input_dir <path> --target_db <path> --output_prefix <name> [--threads <num>]"
    echo "Example: $0 --input_dir ./fasta_data --target_db /db/uniref90 --output_prefix my_experiment --threads 36"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_dir) INPUT_DIR="$2"; shift ;;
        --target_db) TARGET_DB="$2"; shift ;;
        --output_prefix) OUTPUT_PREFIX="$2"; shift ;;
        --threads) THREADS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check for missing required parameters
if [ -z "$INPUT_DIR" ] || [ -z "$TARGET_DB" ] || [ -z "$OUTPUT_PREFIX" ]; then
    echo "[Error] Missing required parameters."
    usage
fi

# Set default threads if not provided
if [ -z "$THREADS" ]; then
    THREADS=8
fi

# Set internal file names based on output prefix
MERGED_FASTA="${OUTPUT_PREFIX}_merged.fasta"
DB_NAME="${OUTPUT_PREFIX}_db"
RESULT_NAME="${OUTPUT_PREFIX}_result"
TMP_DIR="${OUTPUT_PREFIX}_tmp"
MSA_OUTPUT="${OUTPUT_PREFIX}_msa_a3m"

# --- Main Execution ---
echo "Step 1: Merging FASTA files from ${INPUT_DIR}..."
cat ${INPUT_DIR}/*.fasta > ${MERGED_FASTA}
COUNT=$(grep -c "^>" ${MERGED_FASTA})
echo "Total sequences merged: $COUNT"

echo "Step 2: Creating MMseqs2 Database..."
mmseqs createdb ${MERGED_FASTA} ${DB_NAME}

echo "Step 3: Searching against ${TARGET_DB}..."
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

echo "Process Complete! MSA saved to ${MSA_OUTPUT}"
