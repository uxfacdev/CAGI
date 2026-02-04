from pathlib import Path

BASE = r"E:\CAGI_data\processed"

def sum_fasta_lengths(fasta_path):
    total = 0
    current_len = 0

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                total += current_len
                current_len = 0
            else:
                current_len += len(line)

    total += current_len  # 마지막 서열
    return total


for split in ["train", "val", "test"]:
    fasta = Path(BASE) / split / "sequences.fasta"
    total_len = sum_fasta_lengths(fasta)
    print(f"[{split}] total amino acids: {total_len}")

# 전체 합
total_all = sum(
    sum_fasta_lengths(Path(BASE) / split / "sequences.fasta")
    for split in ["train", "val", "test"]
)
print(f"[ALL] total amino acids: {total_all}")
