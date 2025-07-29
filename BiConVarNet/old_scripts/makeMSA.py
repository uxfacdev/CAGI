from multiprocessing import Pool
import subprocess
import os

# DIAMOND 실행 경로 및 입력 디렉토리 설정
diamond_path = r"C:\Users\Kunny\Desktop\정경건\tools\diamond.exe"
db_path = r"E:\CAGI_data\uniref90"
query_dir = r"E:\CAGI_data\fasta_files"
output_dir = r"E:\CAGI_data\blast_results"
os.makedirs(output_dir, exist_ok=True)

def run_diamond(uid, query_path, blast_path):
    tmp_path = blast_path + ".tmp"
    cmd = [
        diamond_path, "blastp",
        "--db", db_path,
        "--query", query_path,
        "--out", tmp_path,
        "--outfmt", "6",
        "--evalue", "0.001",
        "--max-target-seqs", "5000",
        "--threads", "6"
    ]
    print(f"[{uid}] Running DIAMOND...")

    result = subprocess.run(cmd)
    if result.returncode == 0 and os.path.exists(tmp_path):
        os.rename(tmp_path, blast_path)
    else:
        print(f"[{uid}] DIAMOND failed or incomplete output.")

if __name__ == "__main__":
    tasks = []
    for fname in os.listdir(query_dir):
        if fname.endswith(".fasta"):
            uid = fname[:-6]
            query_path = os.path.join(query_dir, fname)
            blast_path = os.path.join(output_dir, f"{uid}.m8")
            if os.path.exists(blast_path):
                continue  
            tasks.append((uid, query_path, blast_path))

    # 병렬 실행
    with Pool(processes=4) as pool:
        pool.starmap(run_diamond, tasks)
