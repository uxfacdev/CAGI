from collections import defaultdict
import re
import json
import pandas as pd
import pickle

def parse_a3m_blocks(a3m_path):
    msa_dict = defaultdict(list)
    with open(a3m_path, 'r', encoding='utf-8', errors='ignore') as f:
        current_query = None
        current_seq_id = None
        buffer = []

        for line in f:
            line = line.strip()
            line = re.sub(r'[\x00-\x1F\x7F]', '', line)

            if not line:
                continue

            if line.startswith(">") and not line.startswith(">UniRef90_"):
                if current_query and current_seq_id and buffer:
                    msa_dict[current_query].append((current_seq_id, ''.join(buffer)))
                    buffer = []
                current_query = line[1:].strip()
                current_seq_id = "query"
            elif line.startswith(">UniRef90_"):
                if current_query and current_seq_id and buffer:
                    msa_dict[current_query].append((current_seq_id, ''.join(buffer)))
                    buffer = []
                current_seq_id = line[1:].strip()
            else:
                buffer.append(line)

        if current_query and current_seq_id and buffer:
            msa_dict[current_query].append((current_seq_id, ''.join(buffer)))

    print(f"Total number of queries: {len(msa_dict)}")
    return msa_dict


if __name__ == "__main__":
    tsv_path = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\rhapsody2_sav_db_exactmatch_only.tsv"
    json_path = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\UniProtID_to_seq.json"
    msa_dict_path = r"E:\CAGI_data\merged_msa_500_a3m"
    output_pkl_path = r"E:\CAGI_data\msa_dict_valid.pkl"

    df = pd.read_csv(tsv_path, sep="\t", header=None)
    df.columns = ["UniProtID", "StructureFile", "MutPos", "WT", "Mut", "Label"]

    with open(json_path) as f:
        seq_dict = json.load(f)

    msa_dict = parse_a3m_blocks(msa_dict_path)

    invalid = []
    for uid in df["UniProtID"]:
        if uid not in msa_dict:
            invalid.append((uid, "missing in msa_dict"))
        elif msa_dict[uid][0][1] != seq_dict.get(uid, ""):
            invalid.append((uid, "sequence mismatch"))

    if not invalid:
        with open(output_pkl_path, "wb") as f:
            pickle.dump(msa_dict, f)
        print(f"msa_dict saved to {output_pkl_path}")
    else:
        print(f"{len(invalid)} UID(s) failed validation. Not saving.")
        print("Examples:", invalid[:5])