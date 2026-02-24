import argparse
from collections import defaultdict
import re
import pandas as pd
import pickle
import sys
import os

def parse_a3m_blocks(a3m_path):
    msa_dict = defaultdict(list)
    print(f"Reading A3M file from: {a3m_path}")
    
    if not os.path.exists(a3m_path):
        print(f"Error: The file {a3m_path} does not exist.")
        sys.exit(1)

    with open(a3m_path, 'r', encoding='utf-8', errors='ignore') as f:
        current_query = None
        current_seq_id = None
        buffer = []

        for line in f:
            line = line.strip()
            # Remove control characters
            line = re.sub(r'[\x00-\x1F\x7F]', '', line)

            if not line:
                continue

            if line.startswith(">") and not line.startswith(">UniRef90_"):
                # Save the previous buffer
                if current_query and current_seq_id and buffer:
                    msa_dict[current_query].append((current_seq_id, ''.join(buffer)))
                    buffer = []
                
                # Parse query ID (extract only the ID before the first space)
                header = line[1:].strip()
                current_query = header.split()[0]
                current_seq_id = "query"
                
            elif line.startswith(">UniRef90_"):
                # Save the previous buffer
                if current_query and current_seq_id and buffer:
                    msa_dict[current_query].append((current_seq_id, ''.join(buffer)))
                    buffer = []
                current_seq_id = line[1:].strip()
            else:
                buffer.append(line)

        # Save the last entry
        if current_query and current_seq_id and buffer:
            msa_dict[current_query].append((current_seq_id, ''.join(buffer)))

    print(f"Total number of queries in A3M: {len(msa_dict)}")
    return msa_dict

def main():
    parser = argparse.ArgumentParser(description="Parse A3M and validate against TSV Keys.")
    
    # Define arguments
    parser.add_argument("--tsv", required=True, help="Path to the input TSV file (must contain 'Key' column)")
    parser.add_argument("--a3m", required=True, help="Path to the merged MSA A3M file")
    parser.add_argument("--output", required=True, help="Path to save the output pickle file")
    
    args = parser.parse_args()

    # 1. Load TSV
    print(f"Loading TSV from: {args.tsv}")
    try:
        df = pd.read_csv(args.tsv, sep="\t")
    except Exception as e:
        print(f"Error loading TSV: {e}")
        sys.exit(1)

    if "Key" not in df.columns:
        print("Error: The column 'Key' is missing in the TSV file.")
        sys.exit(1)

    # 2. Parse A3M file
    msa_dict = parse_a3m_blocks(args.a3m)

    # 3. Validate keys
    print("Validating keys...")
    unique_keys = df["Key"].unique()
    print(f"Number of unique keys in TSV: {unique_keys.shape[0]}")

    invalid = []
    # Validate against unique keys to prevent redundant checks
    for uid in unique_keys:
        if uid not in msa_dict:
            invalid.append((uid, "missing in msa_dict"))

    # 4. Save results
    if not invalid:
        print("Validation passed. Saving pickle...")
        try:
            with open(args.output, "wb") as f:
                pickle.dump(msa_dict, f)
            print(f"Successfully saved msa_dict to {args.output}")
        except Exception as e:
            print(f"Error saving pickle: {e}")
    else:
        print(f"Validation FAILED. {len(invalid)} unique UID(s) are missing in A3M.")
        print("Examples of missing keys:", invalid[:5])
        print("Pickle file was NOT saved.")

if __name__ == "__main__":
    main()
