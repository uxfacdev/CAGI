import os
import json

BASE_DIR = r"E:\CAGI_data"
JSONL_PATH = os.path.join(BASE_DIR, "chain_set.jsonl")
SPLIT_PATH = os.path.join(BASE_DIR, "chain_set_splits.json")
OUT_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# PDB writer (backbone only)
# ---------------------------
def write_pdb(coords, seq, chain_id, out_pdb):
    atom_order = ["N", "CA", "C", "O"]
    atom_serial = 1
    res_id = 1
    chain = chain_id.split(".")[-1]

    with open(out_pdb, "w") as f:
        for i, aa in enumerate(seq):
            for atom_name in atom_order:
                if atom_name not in coords:
                    continue
                x, y, z = coords[atom_name][i]
                f.write(
                    "ATOM  {atom:5d} {name:<4s}{res:>3s} {chain}{resid:4d}    "
                    "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {el:>2s}\n".format(
                        atom=atom_serial,
                        name=atom_name,
                        res=aa,
                        chain=chain,
                        resid=res_id,
                        x=x,
                        y=y,
                        z=z,
                        el=atom_name[0],
                    )
                )
                atom_serial += 1
            res_id += 1
        f.write("END\n")

# ---------------------------
# Load chain data
# ---------------------------
chain_data = {}
with open(JSONL_PATH) as f:
    for line in f:
        item = json.loads(line)
        chain_data[item["name"]] = item

# ---------------------------
# Load splits
# ---------------------------
with open(SPLIT_PATH) as f:
    splits = json.load(f)

split_map = {
    "train": splits["train"],
    "val": splits["validation"],
    "test": splits["test"],
}

# ---------------------------
# Build dataset
# ---------------------------
for split, chain_ids in split_map.items():
    split_dir = os.path.join(OUT_DIR, split)
    pdb_dir = os.path.join(split_dir, "pdbs")
    os.makedirs(pdb_dir, exist_ok=True)

    fasta_path = os.path.join(split_dir, "sequences.fasta")
    tsv_path = os.path.join(split_dir, "metadata.tsv")

    with open(fasta_path, "w") as fasta_f, open(tsv_path, "w") as tsv_f:
        tsv_f.write("ChainID\tPDBID\tChain\tSeqLen\tCATH\n")

        for chain_id in chain_ids:
            if chain_id not in chain_data:
                continue

            item = chain_data[chain_id]
            seq = item["seq"]
            cath = item.get("cath", "NA")

            # 🔥 핵심 수정 부분
            pdb_id, chain = chain_id.rsplit(".", 1)
            pdb_id = pdb_id.upper()

            # TSV
            tsv_f.write(
                f"{chain_id}\t{pdb_id}\t{chain}\t{len(seq)}\t{cath}\n"
            )

            # FASTA
            fasta_f.write(f">{chain_id}\n{seq}\n")

            # PDB
            out_pdb = os.path.join(pdb_dir, f"{chain_id}.pdb")
            write_pdb(item["coords"], seq, chain_id, out_pdb)

    print(f"[DONE] {split} split processed")

print("ALL DONE")
