mmseqs createdb sequences_train.fasta CLIP_train

mmseqs search CLIP_train uniref90_mmseqs CLIP_train_result CLIP_train_tmp --threads 36 -e 0.001 --max-seqs 500

mmseqs result2msa CLIP_train uniref90_mmseqs CLIP_train_result Clip_train_a3m \
  --msa-format-mode 5 \
  --threads 36 \
  --max-seq-id 0.95 \
  --qid 0.3 \
  --cov 0.3 \
  --filter-msa 1 \
  --filter-min-enable 100 \
  --diff 500



mmseqs createdb sequences_val.fasta CLIP_train

mmseqs search CLIP_val uniref90_mmseqs CLIP_val_result CLIP_val_tmp --threads 36 -e 0.001 --max-seqs 500

mmseqs result2msa CLIP_val uniref90_mmseqs CLIP_val_result Clip_val_a3m \
  --msa-format-mode 5 \
  --threads 36 \
  --max-seq-id 0.95 \
  --qid 0.3 \
  --cov 0.3 \
  --filter-msa 1 \
  --filter-min-enable 100 \
  --diff 500