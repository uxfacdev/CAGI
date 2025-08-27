import os, re, gzip
import pandas as pd
from io import StringIO
from collections import defaultdict

# --- 설정: 여기만 수정하면 됨 ---------------------------------
SRC = r"E:\CAGI_data\dbNSFP5.1a_grch38.gz"
OUT_DIR = r"E:\CAGI_data\dbNSFP5.1a_grch38_splits"
ONLY_CHRS = {"X", "Y", "MT", "M"}   # X부터만 만들기 (Y, MT/M 포함)
OVERWRITE = True                    # 기존 결과파일 있으면 지우고 재생성
PREFIX = "dbNSFP5.1a"               # 출력 파일 접두어 (버전에 맞춤)
# ----------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# 1) 실제 헤더 읽기
with gzip.open(SRC, "rt") as fh:
    header = None
    for line in fh:
        if line.startswith("#"):
            header = line
        else:
            break
assert header, "헤더(#...)를 찾지 못했습니다."
COLS = header.lstrip("#").rstrip("\n").split("\t")

# 2) 저장할 컬럼만 추출
BASE_COLS = [
    "chr","pos(1-based)","ref","alt",
    "aaref","aaalt","aapos","genename",
    "Uniprot_acc","Uniprot_entry"
]
for c in ["Uniprot_aapos","Uniprot_aaref","Uniprot_aaalt"]:
    if c in COLS:
        BASE_COLS.append(c)
USECOLS = [c for c in BASE_COLS if c in COLS]
print(f"[info] 전체 {len(COLS)}개 중 선택 {len(USECOLS)}개:", USECOLS)

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1_\2", s)
    s = re.sub(r"[^0-9A-Za-z._-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s

# (선택) 기존 X/Y/MT 파일 삭제 후 새로 생성
if OVERWRITE:
    for c in ONLY_CHRS:
        for prefix in [PREFIX, "dbNSFP5.2a"]:  # 이전 오타 파일도 같이 정리
            p = os.path.join(OUT_DIR, f"{prefix}_{safe_name(c)}.tsv.gz")
            if os.path.exists(p):
                os.remove(p)

wrote_header = defaultdict(bool)
row_counts = defaultdict(int)
chrom_to_path = {}

chunksize = 3_000_000

# 3) 분할 저장 (X, Y, MT/M만 처리)
try:
    for chunk in pd.read_csv(
        SRC, sep="\t", compression="gzip",
        names=COLS, comment="#", usecols=USECOLS,
        dtype=str, low_memory=False, chunksize=chunksize
    ):
        chunk["chr"] = chunk["chr"].astype(str).str.strip()

        # X/Y/MT만 필터
        filt = chunk["chr"].isin(ONLY_CHRS)
        if not filt.any():
            continue
        chunk = chunk.loc[filt]

        for chrom, sub in chunk.groupby("chr", sort=False):
            # 출력 파일 경로(버전에 맞춰 5.1a)
            if chrom not in chrom_to_path:
                fn = f"{PREFIX}_{safe_name(chrom)}.tsv.gz"
                chrom_to_path[chrom] = os.path.join(OUT_DIR, fn)
            path = chrom_to_path[chrom]

            # 헤더 1회만 쓰기
            buf = StringIO()
            sub.to_csv(buf, sep="\t", index=False, header=not wrote_header[path])
            with gzip.open(path, "at", encoding="utf-8") as f:
                f.write(buf.getvalue())

            wrote_header[path] = True
            row_counts[path] += len(sub)

except Exception as e:
    print("[ERR] 조기 종료:", e)

# 4) 결과
print("✅ split 완료. 저장된 파일/행수:")
for path, n in sorted(row_counts.items()):
    print(f" - {os.path.basename(path):<25} ({n:,} rows)")
