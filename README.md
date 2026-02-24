## Data Requirements

The pipeline is divided into two stages: **Raw Data** for preprocessing and **Model Inputs** for training/inference.

### 1. Raw Data (Required for Preprocessing)
These files are necessary to run the MSA and Voxel pipelines:
* **TSV file:** The master dataset containing variant-level metadata (e.g., UniProt ID, mutation position, target labels).
* **FASTA files:** Raw sequence files used as queries to generate Multiple Sequence Alignments (MSA).
* **PDB/AlphaFold files:** 3D structural files (.pdb) used to generate localized voxel representations.

### 2. Model Inputs (Generated Outputs)
These are the final processed files directly ingested by the EvoStructCLIP encoders:
* **Master TSV:** Refined variant list used to index and load the corresponding pickle files.
* **MSA `.pkl` file:** Validated dictionary containing processed evolutionary constraints derived from the MSA pipeline.
* **Voxel `.pkl` files:** Multi-channel 3D structural feature tensors (46-channel) generated from the Voxel pipeline.

---

## Dependencies

Ensure you have the core Python packages and MMseqs2 installed before running the pipelines:

```bash
# 1. Install Python environment dependencies
pip install -r requirements.txt

# 2. Install MMseqs2 (via conda) for ultra-fast sequence searching
conda install -c conda-forge -c bioconda mmseqs2
```
*(Note: A formatted target database, such as UniRef90, is required for the MMseqs2 search.)*

---

## Pipeline Usage

The preprocessing pipeline converts Raw Data into Model Inputs.

### Step 1: MSA Pipeline (Evolutionary Features)
This step searches for homologous sequences using MMseqs2 and parses the resulting A3M file into a validated pickle format.

```bash
# 1. Generate MSA (A3M format) from raw FASTA using MMseqs2
bash make_msa.sh \
  --input_dir <path_to_fasta_directory> \
  --target_db <path_to_mmseqs_database> \
  --output_prefix <output_prefix_name> \
  --threads <number_of_threads>

# 2. Parse A3M and validate against TSV to create the final MSA .pkl
python parse_msa.py \
  --tsv <path_to_input_tsv> \
  --a3m <path_to_generated_a3m> \
  --output <path_to_output_msa_pkl>
```

### Step 2: Voxel Pipeline (Structural Features)
This step processes 3D protein structures to extract localized voxel representations centered around the mutated residue.

```bash
# Generate Voxel features (.pkl) from PDB/AlphaFold files using multiprocessing
python generate_voxel.py \
  --tsv <path_to_input_tsv> \
  --pdb_dir <path_to_directory_containing_pdb_files> \
  --output_dir <path_to_save_voxel_pkl_files> \
  --workers <number_of_cpu_threads>
```
