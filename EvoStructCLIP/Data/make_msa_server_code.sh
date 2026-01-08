/data/gg/CAGI/

cat fasta_files_all_missense/*.fasta > merged_all.fasta

grep -c "^>" merged_all.fasta

mmseqs createdb merged_all.fasta merged_all

mmseqs search merged_all uniref90_mmseqs merged_result_all merged_tmp_all --threads 36 -e 0.001 --max-seqs 500

mmseqs result2msa S2450_all uniref90_mmseqs merged_result_all merged_msa_a3m_all \
  --msa-format-mode 5 \
  --threads 36 \
  --max-seq-id 0.95 \
  --qid 0.3 \
  --cov 0.3 \
  --filter-msa 1 \
  --filter-min-enable 100 \
  --diff 500

mmseqs search S2450_all uniref90_mmseqs S2450_result S2450_tmp --threads 36 -e 0.001 --max-seqs 500
