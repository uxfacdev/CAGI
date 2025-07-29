# mmseqs_run_parallel.py

from multiprocessing import Pool
import subprocess
import os

mmseqs_path = "mmseqs"

base_path = os.path.expanduser("~/CAGI_data")
db_path = os.path.join(base_path, "uniref90_mmseqs")
query_dir = os.path.join(base_path, "fasta_files")
output_dir = os.path.join(base_path, "mmseqs_results")
tmp_root = os.path.join(base_path, "mmseqs_tmp")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_root, exist_ok=True)

failures = []

def run_mmseqs(uid, query_path, m8_path):
    tmp_query = os.path.join(tmp_root, f"{uid}_query")
    tmp_result = os.path.join(tmp_root, f"{uid}_result")
    tmp_tmpdir = os.path.join(tmp_root, f"{uid}_tmp")
    os.makedirs(tmp_tmpdir, exist_ok=True)

    try:
        subprocess.run([mmseqs_path, "createdb", query_path, tmp_query], check=True)

        subprocess.run([
            mmseqs_path, "search", tmp_query, db_path, tmp_result, tmp_tmpdir,
            "--threads", "6", "-e", "0.001", "--max-seqs", "5000"
        ], check=True)

        subprocess.run([
            mmseqs_path, "convertalis", tmp_query, db_path, tmp_result, m8_path
        ], check=True)

    except subprocess.CalledProcessError:
        print(f"[{uid}] MMseqs2 failed.")
        failures.append(uid)

if __name__ == "__main__":
    tasks = []
    for fname in os.listdir(query_dir):
        if fname.endswith(".fasta"):
            uid = fname[:-6]
            query_path = os.path.join(query_dir, fname)
            m8_path = os.path.join(output_dir, f"{uid}.m8")
            if os.path.exists(m8_path):
                continue
            tasks.append((uid, query_path, m8_path))

    with Pool(processes=1) as pool:
        pool.starmap(run_mmseqs, tasks)

    if failures:
        fail_log = os.path.join(output_dir, "failed.log")
        with open(fail_log, "w") as f:
            for uid in failures:
                f.write(uid + "\n")
        print(f"\n❗ {len(failures)} failures logged in {fail_log}")
    else:
        print("\n✅ All tasks completed successfully.")

