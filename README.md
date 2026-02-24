# EvoStructCLIP: MSA Preprocessing Pipeline

This repository contains the data preprocessing pipeline for **EvoStructCLIP**, a multimodal embedding model for variant effect prediction. 

EvoStructCLIP requires three primary inputs to generate embeddings:
1. **TSV file**: Contains variant information (e.g., UniProt ID, sequence, mutation details).
2. **PDB `.pkl` file**: Contains 3D structural features derived from AlphaFold/PDB.
3. **MSA `.pkl` file**: Contains evolutionary constraints derived from Multiple Sequence Alignments (MSA).

This documentation specifically covers the **MSA `.pkl` generation pipeline**, which consists of two main steps: running MMseqs2 to generate alignments (`make_msa.sh`) and parsing the results into a validated pickle format (`parse_msa.py`).

## ⚙️ Dependencies

Before running the pipeline, ensure you have the required dependencies installed.

### 1. Python Environment
Install the core Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt

2. MMseqs2
This pipeline relies on MMseqs2 for ultra-fast sequence searching and alignment. Please install it following the official MMseqs2 GitHub page or via conda:

Bash

conda install -c conda-forge -c bioconda mmseqs2
Note: You also need a formatted target database (e.g., UniRef90) to run the search.

🚀 Pipeline Usage
Step 1: Generate MSA using MMseqs2 (make_msa.sh)
This bash script merges individual FASTA files, builds an MMseqs2 database, searches against a target database (e.g., UniRef90), and outputs an alignment file in A3M format.

Usage:

Bash

bash make_msa.sh \
  --input_dir <path_to_fasta_directory> \
  --target_db <path_to_uniref90_database> \
  --output_prefix <experiment_name> \
  --threads <number_of_threads>
Example:

Bash

bash make_msa.sh \
  --input_dir ./fasta_files_all_missense \
  --target_db /db/uniref90_mmseqs \
  --output_prefix merged_all \
  --threads 36
Output: This will generate several intermediate files and the final alignment file named merged_all_msa_a3m.

Step 2: Parse and Validate MSA (parse_msa.py)
This Python script parses the generated A3M file, validates it against the unique keys (e.g., UniProt IDs) present in your master TSV file, and saves the parsed MSA data as a Python dictionary in a .pkl file. This .pkl file is the final input required by the EvoStructCLIP MSA branch.

Usage:

Bash

python parse_msa.py \
  --tsv <path_to_input_tsv> \
  --a3m <path_to_a3m_file_from_step1> \
  --output <path_to_save_pickle>
Example:

Bash

python parse_msa.py \
  --tsv ./filtered_variants_cleaned_final.tsv \
  --a3m ./merged_all_msa_a3m \
  --output ./msa_dict_valid.pkl
Validation Process:
The script will check if every Key (e.g., UniProt ID) present in the TSV file exists in the parsed A3M data.

If validation passes, it saves the dictionary to the specified .pkl file.

If validation fails (e.g., missing IDs), it will display an error message with examples and will not save the pickle file, ensuring data integrity.
