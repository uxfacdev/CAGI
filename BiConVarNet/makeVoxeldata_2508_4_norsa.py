import os, pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

voxel_dir = r"E:\CAGI_data\voxel_cache_4"
out_dir   = r"E:\CAGI_data\voxel_cache_4_noRSA"
os.makedirs(out_dir, exist_ok=True)

def process_file(fname):
    if not fname.endswith(".pkl"):
        return None
    
    in_path  = os.path.join(voxel_dir, fname)
    out_path = os.path.join(out_dir, fname)

    # 이미 처리된 경우 skip
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return f"{fname}: skipped"

    try:
        with open(in_path, "rb") as f:
            data = pickle.load(f)

        feat = data["feature"]   # (1,7,7,7,48)
        # 채널 순서: [42 closeness | 2 Δpos | 1 pLDDT | 1 RSA | 1 ResDepth | 1 GNM]
        # RSA+ResDepth 제거: index 45, 46
        mask = [i for i in range(feat.shape[-1]) if i not in (45, 46)]
        feat_new = feat[..., mask]

        data["feature"] = feat_new.astype(np.float32)

        with open(out_path, "wb") as f:
            pickle.dump(data, f)

        return f"{fname}: done"
    except Exception as e:
        return f"{fname}: failed ({e})"

if __name__ == "__main__":
    fnames = os.listdir(voxel_dir)
    with ProcessPoolExecutor(max_workers=20) as ex:
        results = list(tqdm(ex.map(process_file, fnames), total=len(fnames)))

    # 로그 출력
    for r in results:
        if r is not None:
            print(r)
