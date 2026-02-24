# EvoStructCLIP



EvoStructCLIP is a small-scale, multimodal mutation-centered embedding model designed to predict the functional consequences of missense variants. Rather than relying exclusively on global protein-wide representations, this model explicitly focuses on the coordinated local contexts of mutations.

The architecture integrates structural and evolutionary evidence through two complementary branches:
* **Voxel Branch:** A 3D convolutional network (based on 3D MBConv and CoordAtt3D) that captures the local, three-dimensional structural environment surrounding a mutated residue.
* **MSA Branch:** A cross-axial Mamba-based encoder that efficiently processes multiple sequence alignments (MSAs) to model deep evolutionary constraints and local consensus signals across homologous sequences.

Embeddings from both modalities are aligned in a shared latent space using a symmetric CLIP-style contrastive loss. This alignment is jointly optimized with supervised variant pathogenicity classification and FuseMix latent-space regularization, providing a compact and transferable representation for diverse downstream variant effect prediction tasks (e.g., thermodynamic stability, transcript-level abundance, and receptor activation).

---

## Required Inputs

To run the data preprocessing pipeline and train/evaluate EvoStructCLIP, the following data formats are utilized:

1. **TSV file:** The master dataset containing variant-level metadata (e.g., UniProt ID, sequence, mutation position, target labels).
2. **FASTA files:** Raw sequence files used as queries to generate multiple sequence alignments.
3. **MSA `.pkl` file:** The processed evolutionary constraints and sequence alignments generated from the FASTA files.
4. **Voxel `.pkl` file:** The processed 46-channel 3D structural features derived from AlphaFold models or experimental PDB structures.

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

The data preprocessing pipeline converts raw TSV, FASTA, and PDB files into the final `.pkl` formats directly ingested by the EvoStructCLIP encoders.

### Step 1: MSA Pipeline (Evolutionary Features)
This step searches for homologous sequences using MMseqs2 and parses the resulting A3M file into a validated pickle format based on the variants in your master TSV file.

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
This step processes 3D protein structures to extract localized, multi-channel voxel representations centered around the mutated residue.

```bash
# Generate Voxel features (.pkl) from PDB/AlphaFold files using multiprocessing
python generate_voxel.py \
  --tsv <path_to_input_tsv> \
  --pdb_dir <path_to_directory_containing_pdb_files> \
  --output_dir <path_to_save_voxel_pkl_files> \
  --workers <number_of_cpu_threads>
```
