import pandas as pd
import numpy as np

# -----------------------------
# config
# -----------------------------
df_path = r"C:\Users\Kunny\Research\Project\BiConVarNet\AnnotateAll\AlphaMissense_variants_final_for_backbone.tsv"
out_path = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\CLIP_\AlphaMissense_variants_clip_sampled.tsv"

POSITION_STRIDE = 30   # 대충 30 residue마다 1개
RANDOM_SEED = 42

# -----------------------------
# load
# -----------------------------
df = pd.read_csv(df_path, sep="\t")

# -----------------------------
# MutPos(pdb) → int
# -----------------------------

df = df[df["UniProtID"] != "Q8WZ42"]
df["MutPos(pdb)"] = df["MutPos(pdb)"].astype(float).astype(int)

# -----------------------------
# 정렬 (단백질별 residue 순서 보장)
# -----------------------------
df = df.sort_values(
    by=["UniProtID", "StructureFile", "MutPos(pdb)"]
).reset_index(drop=True)

rng = np.random.default_rng(RANDOM_SEED)
selected_rows = []

# -----------------------------
# 단백질 단위로 sampling
# -----------------------------
for (uniprot, pdb), g in df.groupby(["UniProtID", "StructureFile"]):
    last_selected_pos = -10**9

    for pos, g_pos in g.groupby("MutPos(pdb)"):
        if pos - last_selected_pos >= POSITION_STRIDE:
            # 같은 포지션에 여러 missense 중 1개 랜덤 선택
            row = g_pos.sample(n=1, random_state=RANDOM_SEED)
            selected_rows.append(row)
            last_selected_pos = pos

# -----------------------------
# 결과 DF
# -----------------------------
df_sampled = pd.concat(selected_rows, ignore_index=True)

# -----------------------------
# 저장
# -----------------------------
df_sampled.to_csv(out_path, sep="\t", index=False)

print(f"Original size : {len(df):,}")
print(f"Sampled size  : {len(df_sampled):,}")
print(f"Saved to      : {out_path}")
