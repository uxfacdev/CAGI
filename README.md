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
