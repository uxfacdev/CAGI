import os
import json
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_msa_file(msa_path, output_dir):
    try:
        with open(msa_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return f"[ERROR] Could not read {msa_path}: {e}"

    if len(sequences) < 3:
        return f"[SKIP] {os.path.basename(msa_path)}: too few sequences (<3)"

    seq_len = len(sequences[0])
    position_freqs = [Counter() for _ in range(seq_len)]
    overall_freqs = Counter()

    for seq in sequences:
        for i, aa in enumerate(seq):
            position_freqs[i][aa] += 1
            overall_freqs[aa] += 1

    total_seqs = len(sequences)
    total_aa_counts = sum(overall_freqs.values())

    overall_freqs = {aa: count / total_aa_counts for aa, count in overall_freqs.items()}
    position_freqs = [
        {aa: count / total_seqs for aa, count in pos_freq.items()}
        for pos_freq in position_freqs
    ]

    output_data = {
        "overall_freqs": overall_freqs,
        "position_freqs": position_freqs
    }

    msa_file = os.path.basename(msa_path)
    json_name = msa_file.replace(".seqmsa", ".seqmsa.json")
    json_path = os.path.join(output_dir, json_name)

    try:
        with open(json_path, 'w') as jf:
            json.dump(output_data, jf)
    except Exception as e:
        return f"[ERROR] Could not write {json_name}: {e}"

def precompute_and_save_all_frequencies_parallel(msa_dir, output_dir, max_workers=8):
    print(f"Scanning MSA files in {msa_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    msa_files = [
        os.path.join(msa_dir, f) for f in os.listdir(msa_dir) if f.endswith(".seqmsa")
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_msa_file, path, output_dir) for path in msa_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel processing"):
            msg = future.result()
            if msg:
                tqdm.write(msg)



if __name__ == "__main__":
    precompute_and_save_all_frequencies_parallel(
        msa_dir="E:/CAGI_data/MSA_folder",
        output_dir="E:/CAGI_data/PSIC_folder",
        max_workers=12
    )