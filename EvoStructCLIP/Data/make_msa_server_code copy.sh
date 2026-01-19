mmseqs createdb PROSTATA.fasta PROSTATA

mmseqs search PROSTATA uniref90_mmseqs PROSTATA_result PROSTATA_tmp --threads 36 -e 0.001 --max-seqs 500

mmseqs result2msa PROSTATA uniref90_mmseqs PROSTATA_result PROSTATA_a3m \
  --msa-format-mode 5 \
  --threads 36 \
  --max-seq-id 0.95 \
  --qid 0.3 \
  --cov 0.3 \
  --filter-msa 1 \
  --filter-min-enable 100 \
  --diff 500

