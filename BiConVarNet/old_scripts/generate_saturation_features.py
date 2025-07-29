import os
import json
import math
import prody as pr
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Define amino acid groups including gap '-' as a subgroup
amino_acid_groups1 = {
    'hydrophobic': {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'},
    'polar': {'S', 'T', 'Y', 'N', 'Q', 'C', 'G'},
    'positive': {'K', 'R', 'H'},
    'negative': {'D', 'E'},
    'other': {'B', 'J', 'Z', 'X'},
    'gap': {'-'}
}

amino_acid_groups2 = {
    'smallest': {'G', 'A', 'S'},
    'small': {'T', 'C', 'P', 'D', 'N', 'V'},
    'medium': {'I', 'E', 'Q', 'L', 'H', 'M'},
    'large': {'F', 'Y', 'W', 'R', 'K'},
    'other': {'B', 'J', 'Z', 'X'},
    'gap': {'-'}
}

aa_codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def get_amino_acid_group(amino_acid, groups):
    for group, amino_acids in groups.items():
        if amino_acid in amino_acids:
            return group
    return 'unknown'

def calculate_psic(precomputed_folder, msa_filename, position, amino_acid, groups):
    precomputed_path = os.path.join(precomputed_folder, os.path.basename(msa_filename).replace('.pdb', '') + '.seqmsa.json')
    
    try:
        with open(precomputed_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading precomputed JSON file: {e}")
        return float('NaN'), float('NaN'), 'unknown'
    
    overall_freqs = data.get('overall_freqs', {})
    position_freqs = data.get('position_freqs', [])

    if not position_freqs or position < 1 or position > len(position_freqs):
        print("Position out of range or no position frequencies available")
        return float('NaN'), float('NaN'), 'unknown'
    
    position -= 1
    column_freqs = position_freqs[position]
    
    total_seqs = sum(column_freqs.values())
    observed_freq = column_freqs.get(amino_acid, 0) / total_seqs if total_seqs > 0 else 0
    
    total_aa_counts = sum(overall_freqs.values())
    expected_freq = overall_freqs.get(amino_acid, 0) / total_aa_counts if total_aa_counts > 0 else 0
    
    psic_score = float('-inf') if observed_freq == 0 or expected_freq == 0 else math.log(observed_freq / expected_freq)
    
    amino_acid_group = get_amino_acid_group(amino_acid, groups)
    if amino_acid_group == 'unknown':
        print(f"Unknown amino acid: {amino_acid}")
        return float('NaN'), float('NaN'), 'unknown'
    
    group_freq = sum(column_freqs.get(aa, 0) for aa in groups[amino_acid_group]) / total_seqs if total_seqs > 0 else 0
    group_expected_freq = sum(overall_freqs.get(aa, 0) for aa in groups[amino_acid_group]) / total_aa_counts if total_aa_counts > 0 else 0
    group_psic_score = float('-inf') if group_freq == 0 or group_expected_freq == 0 else math.log(group_freq / group_expected_freq)

    return psic_score, group_psic_score, amino_acid_group

def generate_saturation_mutagenesis_features_fasta(uniprot_id, sequence, precomputed_folder, output_file):
    # print(f"[{uniprot_id}] Generating saturation mutagenesis features...")

    output_data = []
    for i, wt_residue in enumerate(sequence):
        pos = i + 1  # 1-based index
        if wt_residue not in aa_codes:
            continue  # Skip unknowns
        for mut_residue in aa_codes:
            wt_psic, wt_group_psic, _ = calculate_psic(precomputed_folder, uniprot_id, pos, wt_residue, amino_acid_groups1)
            wt_size_psic, _, _ = calculate_psic(precomputed_folder, uniprot_id, pos, wt_residue, amino_acid_groups2)
            mut_psic, mut_group_psic, _ = calculate_psic(precomputed_folder, uniprot_id, pos, mut_residue, amino_acid_groups1)
            mut_size_psic, _, _ = calculate_psic(precomputed_folder, uniprot_id, pos, mut_residue, amino_acid_groups2)
            wt_mut_psic = wt_psic - mut_psic
            wt_mut_group_psic = wt_group_psic - mut_group_psic
            wt_mut_size_psic = wt_size_psic - mut_size_psic
            output_data.append([
                uniprot_id,
                pos,
                wt_residue,
                mut_residue,
                f"{wt_psic:.3f}",
                f"{wt_group_psic:.3f}",
                f"{wt_size_psic:.3f}",
                f"{mut_psic:.3f}",
                f"{mut_group_psic:.3f}",
                f"{mut_size_psic:.3f}",
                f"{wt_mut_psic:.3f}",
                f"{wt_mut_group_psic:.3f}",
                f"{wt_mut_size_psic:.3f}"
            ])

    with open(output_file, "w") as f:
        f.write("PDB_File\tResidue_Number\tWT_Residue\tMutation\tWT_Seq_PSIC\tWT_PhyChem_Group_Seq_PSIC\tWT_Size_Group_Seq_PSIC\tMut_Seq_PSIC\tMut_PhyChem_Group_Seq_PSIC\tMut_Size_Group_Seq_PSIC\tWT-Mut_Seq_PSIC\tWT-Mut_PhyChem_Group_Seq_PSIC\tWT-Mut_Size_Group_Seq_PSIC\n")
        for row in output_data:
            f.write("\t".join(map(str, row)) + "\n")

def run_batch_saturation_feature_gen(json_seq_path, precomputed_folder, output_dir):
    with open(json_seq_path, 'r') as f:
        uniprot_dict = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(precomputed_folder) if f.endswith(".seqmsa.json")]
    for json_file in tqdm(json_files, desc="Generating PSIC features"):
        uniprot_id = json_file.replace(".seqmsa.json", "")
        if uniprot_id not in uniprot_dict:
            print(f"[WARN] {uniprot_id} not found in sequence dict.")
            continue
        sequence = uniprot_dict[uniprot_id]
        output_file = os.path.join(output_dir, f"{uniprot_id}_saturation.tsv")
        generate_saturation_mutagenesis_features_fasta(uniprot_id, sequence, precomputed_folder, output_file)

def process_single_uniprot(args):
    uniprot_id, sequence, precomputed_folder, output_dir = args
    output_file = os.path.join(output_dir, f"{uniprot_id}_saturation.tsv")
    generate_saturation_mutagenesis_features_fasta(uniprot_id, sequence, precomputed_folder, output_file)
    
def run_batch_saturation_feature_gen_parallel(json_seq_path, precomputed_folder, output_dir, num_workers=None):
    with open(json_seq_path, 'r') as f:
        uniprot_dict = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(precomputed_folder) if f.endswith(".seqmsa.json")]

    task_list = []
    for json_file in json_files:
        uniprot_id = json_file.replace(".seqmsa.json", "")
        if uniprot_id not in uniprot_dict:
            print(f"[WARN] {uniprot_id} not found in sequence dict.")
            continue
        sequence = uniprot_dict[uniprot_id]
        task_list.append((uniprot_id, sequence, precomputed_folder, output_dir))

    print(f"Launching {len(task_list)} jobs with {num_workers or cpu_count()} workers...")
    with Pool(processes=num_workers or cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_single_uniprot, task_list), total=len(task_list)))


if __name__ == "__main__":
    run_batch_saturation_feature_gen_parallel(
        json_seq_path=r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\UniProtID_to_seq.json",
        precomputed_folder="E:/CAGI_data/PSIC_folder",
        output_dir="E:/CAGI_data/saturation_features",
        num_workers=12
    )