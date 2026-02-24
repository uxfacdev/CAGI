# EvoStructCLIP: Data Preprocessing Pipeline

This repository contains the data preprocessing pipeline for **EvoStructCLIP**, a multimodal embedding model for variant effect prediction. 

EvoStructCLIP requires three primary inputs to generate embeddings:
1. **TSV file**: Contains variant information (e.g., UniProt ID, sequence, mutation details).
2. **MSA `.pkl` file**: Contains evolutionary constraints derived from Multiple Sequence Alignments (MSA).
3. **Voxel `.pkl` file**: Contains 3D structural features derived from AlphaFold/PDB.

## Dependencies

Install the core Python packages and MMseqs2 before running the pipeline:

```bash
# Python environment
pip install -r requirements.txt

# MMseqs2 (via conda)
conda install -c conda-forge -c bioconda mmseqs2
```
*(Note: A formatted target database like UniRef90 is required for the MMseqs2 search.)*

---

## Pipeline Usage

### Step 1: MSA Pipeline
This step searches for homologous sequences using MMseqs2 and parses the resulting A3M file into a validated pickle format based on the variants in your TSV file.

```bash
# 1. Generate MSA (A3M format) using MMseqs2
bash make_msa.sh \
  --input_dir <path_to_fasta_directory> \
  --target_db <path_to_mmseqs_database> \
  --output_prefix <output_prefix_name> \
  --threads <number_of_threads>

# 2. Parse A3M and validate against TSV to create the final MSA .pkl
python parse_msa.py \
  --tsv <path_to_input_tsv> \
  --a3m <path_to_generated_a3m> \
  --output <path_to_output_pkl>
```
