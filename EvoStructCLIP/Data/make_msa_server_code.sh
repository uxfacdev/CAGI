mmseqs createdb merged_all.fasta merged_all

mmseqs search merged_all uniref90_mmseqs merged_result_all merged_tmp_all --threads 36 -e 0.001 --max-seqs 500

mmseqs result2msa merged_all uniref90_mmseqs merged_result_all merged_msa_a3m_all \
  --msa-format-mode 5 \
  --threads 36 \
  --max-seq-id 0.95 \
  --qid 0.3 \
  --cov 0.3 \
  --filter-msa 1 \
  --filter-min-enable 100 \
  --diff 500

