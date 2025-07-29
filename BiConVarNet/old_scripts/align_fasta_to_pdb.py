import pandas as pd
import os
import json
import prody as pr
from Bio import pairwise2
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

# 경로
tsv_path = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\rhapsody2_sav_db.tsv"
json_path = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\UniProtID_to_seq.json"
pdb_dir = r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset\alphafold_structures"
output_path = os.path.join(r"C:\Users\Kunny\Research\Dataset\Missense Variant dataset", "rhapsody2_sav_db_with_fastapos.tsv")
log_path = "fasta_mapping_fail_log.txt"

uniprot_seq_dict = None

def init_worker(shared_dict):
    global uniprot_seq_dict
    uniprot_seq_dict = shared_dict

# 3-letter → 1-letter amino acid 변환
aa_names = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Align 함수
def align_fasta_to_pdb(fasta_seq, pdb_seq):
    alignments = pairwise2.align.globalxx(fasta_seq, pdb_seq)
    a_fasta, a_pdb = alignments[0].seqA, alignments[0].seqB
    f2p, p2f = {}, {}
    f_idx = p_idx = 0
    for i in range(len(a_fasta)):
        if a_fasta[i] != '-':
            f_idx += 1
        if a_pdb[i] != '-':
            p_idx += 1
        if a_fasta[i] != '-' and a_pdb[i] != '-':
            f2p[f_idx] = p_idx
            p2f[p_idx] = f_idx
    return f2p, p2f

# PDB sequence 불러오기
def get_pdb_sequence_and_resnums(pdb_path):
    structure = pr.parsePDB(pdb_path, subset='ca')
    resnames = structure.getResnames()
    resnums = structure.getResnums()
    pdb_seq = [aa_names.get(r, 'X') for r in resnames]
    return ''.join(pdb_seq), resnums

# 한 row 처리 함수
def process_single_row(row):
    uid = row["UniProtID"]
    pdb_file = os.path.join(pdb_dir, row["StructureFile"])
    mut_pos_pdb_resnum = int(row["MutPos"])
    wt_aa = row["WT"]

    fasta_seq = uniprot_seq_dict.get(uid, "")
    if not fasta_seq:
        return None, (uid, "No FASTA seq")

    try:
        pdb_seq, resnums = get_pdb_sequence_and_resnums(pdb_file)
    except Exception as e:
        return None, (uid, f"PDB parse failed: {str(e)}")

    try:
        pdb_index = resnums.tolist().index(mut_pos_pdb_resnum) + 1
    except ValueError:
        return None, (uid, f"resnum {mut_pos_pdb_resnum} not in PDB")

    try:
        f2p, p2f = align_fasta_to_pdb(fasta_seq, pdb_seq)
        fasta_pos = p2f.get(pdb_index, None)
        if fasta_pos is None:
            return None, (uid, f"No mapping for PDB index {pdb_index}")
    except Exception as e:
        return None, (uid, f"Alignment failed: {str(e)}")

    try:
        if fasta_seq[fasta_pos - 1] != wt_aa:
            return fasta_pos, (uid, f"WT mismatch at FastaPos={fasta_pos}: expected {wt_aa}, got {fasta_seq[fasta_pos - 1]}")
    except Exception as e:
        return fasta_pos, (uid, f"WT check failed: {str(e)}")

    return fasta_pos, None

# 메인 실행
if __name__ == "__main__":
    df = pd.read_csv(tsv_path, sep="\t", header=None)
    df.columns = ["UniProtID", "StructureFile", "MutPos", "WT", "Mut", "Label"]

    rows = [row for _, row in df.iterrows()]
    print(f"Launching {len(rows)} jobs with {cpu_count()} workers...")

    with open(json_path) as f:
        seq_dict = json.load(f)

    manager = Manager()
    shared_dict = manager.dict(seq_dict)

    with Pool(processes=16, initializer=init_worker, initargs=(shared_dict,)) as pool:
        results = list(tqdm(pool.imap(process_single_row, rows), total=len(rows)))

    fasta_pos_list = []
    fail_logs = []

    for fasta_pos, fail in results:
        fasta_pos_list.append(fasta_pos)
        if fail:
            fail_logs.append(fail)

    df["FastaPos"] = fasta_pos_list
    df.to_csv(output_path, sep="\t", index=False)
    print(f"\n✅ 결과 TSV 저장 완료: {output_path}")

    with open(log_path, "w") as f:
        for uid, reason in fail_logs:
            f.write(f"{uid}\t{reason}\n")
    print(f"⚠️ 총 실패: {len(fail_logs)} → 로그 저장 완료: {log_path}")