import pandas as pd
import random

# ===============================
# 경로
# ===============================
tsv_path = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\CLIP_\filtered_variants_cleaned_final.tsv"
out_path = r"C:\Users\Kunny\Documents\GitHub\CAGI\EvoStructCLIP\CLIP_\filtered_variants_mutonly_aug.tsv"

# ===============================
# 설정
# ===============================
AUG_PER_VARIANT = 3  # 2~4로 조절 가능
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# ===============================
# 로드
# ===============================
df = pd.read_csv(tsv_path, sep="\t")
df = df.drop(columns=["Label"], errors="ignore")

# ===============================
# 기존 key set
# ===============================
existing_keys = set(
    zip(df["UniProtID"], df["MutPos"], df["WT"], df["Mut"])
)

augmented_rows = []

# ===============================
# Mut-only 증강
# ===============================
for _, row in df.iterrows():
    uid = row["UniProtID"]
    mut_pos = row["MutPos"]
    wt = row["WT"]
    original_mut = row["Mut"]

    # WT / 기존 Mut 제외
    candidate_muts = [
        aa for aa in AA_LIST if aa != wt and aa != original_mut
    ]

    random.shuffle(candidate_muts)

    n_aug = min(AUG_PER_VARIANT, len(candidate_muts))
    for mut in candidate_muts[:n_aug]:
        key = (uid, mut_pos, wt, mut)
        if key in existing_keys:
            continue

        augmented_rows.append({
            "UniProtID": uid,
            "MutPos": mut_pos,
            "WT": wt,
            "Mut": mut,
            "StructureFile": row["StructureFile"],
            "MutPos(pdb)": row["MutPos(pdb)"]
        })

        existing_keys.add(key)

# ===============================
# 병합 및 저장
# ===============================
aug_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
aug_df.to_csv(out_path, sep="\t", index=False)

print(f"Done: {len(df)} → {len(aug_df)} (mut-only augmentation)")
