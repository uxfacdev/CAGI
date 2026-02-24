# EvoStructCLIP



EvoStructCLIP is a small-scale, multimodal mutation-centered embedding model designed to predict the functional consequences of missense variants. Rather than relying exclusively on global protein-wide representations, this model explicitly focuses on the coordinated local contexts of mutations.

The architecture integrates structural and evolutionary evidence through two complementary branches:
* **Voxel Branch:** A 3D convolutional network (based on 3D MBConv and CoordAtt3D) that captures the local, three-dimensional structural environment surrounding a mutated residue.
* **MSA Branch:** A cross-axial Mamba-based encoder that efficiently processes multiple sequence alignments (MSAs) to model deep evolutionary constraints and local consensus signals across homologous sequences.

Embeddings from both modalities are aligned in a shared latent space using a symmetric CLIP-style contrastive loss. This alignment is jointly optimized with supervised variant pathogenicity classification and FuseMix latent-space regularization.

---

## Dependencies

Install the core Python packages and MMseqs2 before running the pipelines:

```bash
# 1. Install Python environment dependencies
pip install -r requirements.txt

# 2. Install MMseqs2 (via conda)
conda install -c conda-forge -c bioconda mmseqs2
```

---

## Preprocessing

EvoStructCLIP takes a **TSV file**, an **MSA `.pkl` file**, and **Voxel `.pkl` files** as direct model inputs. 이 파일들은 아래의 **Required Files**를 전처리(Preprocessing)하여 생성됩니다.

### Required Files
* **TSV file:** The master dataset containing variant-level metadata.
* **FASTA files:** Raw sequence files used as queries for MSA generation.
* **PDB/AlphaFold files:** 3D structural files (.pdb) for voxel representations.

To ensure successful preprocessing, the unique identifiers (e.g., UniProt ID) must be consistent across all files. For example, if the ID in the TSV is `P01112`, the corresponding structural file must be named `p01112.pdb` and the FASTA entry must start with `>P01112`.

---

## Preprocessing Pipeline

### Step 1: MSA Pipeline
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

### Step 2: Voxel Pipeline
This step processes 3D protein structures to extract localized voxel representations centered around the mutated residue.

```bash
# Generate Voxel features (.pkl) from PDB/AlphaFold files using multiprocessing
python generate_voxel.py \
  --tsv <path_to_input_tsv> \
  --pdb_dir <path_to_directory_containing_pdb_files> \
  --output_dir <path_to_save_voxel_pkl_files> \
  --workers <number_of_cpu_threads>
```