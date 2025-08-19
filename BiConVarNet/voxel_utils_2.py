import numpy as np
from prody import parsePDB
from typing import Tuple

# Constants
AA3_LIST = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS',
    'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN',
    'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'UNK'
]  # 20 aa + 'UNK' = 21
ATOM_TYPES = ['CA', 'CB']
AA3_TO_INDEX = {aa: i for i, aa in enumerate(AA3_LIST)}

SCAN_RADIUS = 5.0  # Å
VOXEL_SIZE = 2.0   # Å
GRID_SIZE = 7      # 7x7x7 voxel grid

def get_voxel_centers(center_coord: np.ndarray) -> np.ndarray:
    half = GRID_SIZE // 2
    shifts = (np.arange(-half, half + 1, dtype=np.float32) * VOXEL_SIZE)
    gx, gy, gz = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    offsets = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    return center_coord.astype(np.float32) + offsets

def compute_atomic_distance_profile(
    pdb_path: str,
    mut_pos: int,
    add_seqpos_features: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      feature_grid: (1, 7, 7, 7, 63[+2]), mapping_grid: (1, 7, 7, 7)
    63ch = 42 closeness(CA/CB x 21) + 21 one-hot(CA-only 최근접 잔기)
    [+2ch] = Δpos 상대서열거리 (signed_norm, abs_norm)
    """
    # --- Load ---
    structure = parsePDB(pdb_path)
    atoms_all = structure.select('name CA CB')
    if atoms_all is None or len(atoms_all) == 0:
        raise ValueError("No CA or CB atoms found in structure.")

    coords_all   = atoms_all.getCoords().astype(np.float32)
    names_all    = atoms_all.getNames()
    resnames_all = atoms_all.getResnames()
    resnums_all  = atoms_all.getResnums().astype(np.int32)

    # --- Center at mut_pos CA ---
    ca_mask_mut = (names_all == 'CA') & (resnums_all == mut_pos)
    if not np.any(ca_mask_mut):
        raise ValueError(f"Mutation site {mut_pos} not found in CA atoms.")
    center_coord = coords_all[ca_mask_mut][0]

    voxel_centers = get_voxel_centers(center_coord)   # (343,3)
    V = voxel_centers.shape[0]
    R2 = np.float32(SCAN_RADIUS * SCAN_RADIUS)

    # --- Outputs ---
    base_feat_size = len(ATOM_TYPES) * len(AA3_LIST)  # 42
    extra = 2 if add_seqpos_features else 0
    total_feat = 63 + extra
    feature_grid = np.zeros((V, total_feat), dtype=np.float32)
    mapping_grid = np.full((V,), -1, dtype=np.int32)

    # --- Closeness index cache (CA/CB 모두) ---
    idx_cache = {}
    for atom_type in ATOM_TYPES:
        for aa in AA3_LIST:
            if atom_type == 'CB' and aa == 'GLY':
                idx_cache[(atom_type, aa)] = np.empty((0,), dtype=np.int32)
            else:
                mask = (names_all == atom_type) & (resnames_all == aa)
                idx_cache[(atom_type, aa)] = np.where(mask)[0].astype(np.int32)

    # --- CA-only 집합(최근접 잔기 판단용) ---
    ca_mask_all = (names_all == 'CA')
    coords_ca   = coords_all[ca_mask_all]
    resnames_ca = resnames_all[ca_mask_all]
    resnums_ca  = resnums_all[ca_mask_all]

    # 체인 길이 근사 (resnum 연속 가정 완화용: max-min+1)
    if resnums_ca.size == 0:
        raise ValueError("No CA atoms found.")
    L_est = max(1, int(resnums_ca.max() - resnums_ca.min() + 1))

    # --- CA-only 최근접 잔기 (원핫/매핑) ---
    diff_ca = voxel_centers[:, None, :] - coords_ca[None, :, :]      # (V,Nc,3)
    dist2_ca = np.einsum('vka,vka->vk', diff_ca, diff_ca)            # (V,Nc)
    nearest_ca_idx = dist2_ca.argmin(axis=1)                         # (V,)
    nearest_resnums_ca  = resnums_ca[nearest_ca_idx]                 # (V,)
    nearest_resnames_ca = resnames_ca[nearest_ca_idx]                # (V,)

    # mapping(resnum)
    mapping_grid[:] = nearest_resnums_ca

    # one-hot(21ch)
    for v in range(V):
        aa3 = nearest_resnames_ca[v]
        aa_idx = AA3_TO_INDEX.get(aa3, 20)
        feature_grid[v, base_feat_size + aa_idx] = 1.0  # channels 42..62

    # --- Δpos 2채널 추가(옵션) ---
    if add_seqpos_features:
        delta = nearest_resnums_ca.astype(np.int32) - int(mut_pos)
        signed_norm = np.clip(delta.astype(np.float32) / float(L_est), -1.0, 1.0)
        abs_norm = np.minimum(np.abs(signed_norm), 1.0)
        feature_grid[:, 63] = signed_norm
        feature_grid[:, 64] = abs_norm

    # --- Closeness(CA/CB 모두) 0..41 채우기 ---
    for atom_type_idx, atom_type in enumerate(ATOM_TYPES):
        for aa_idx, aa in enumerate(AA3_LIST):
            col = aa_idx + 21 * atom_type_idx  # 0..41
            idxs = idx_cache[(atom_type, aa)]
            if idxs.size == 0:
                continue
            sub_coords = coords_all[idxs]                                        # (K,3)
            diff = voxel_centers[:, None, :] - sub_coords[None, :, :]            # (V,K,3)
            dist2 = np.einsum('vka,vka->vk', diff, diff)                         # (V,K)
            dist2_masked = np.where(dist2 <= R2, dist2, np.inf).astype(np.float32)
            min_dist2 = np.min(dist2_masked, axis=1)                             # (V,)
            valid = np.isfinite(min_dist2)
            closeness = np.zeros((V,), dtype=np.float32)
            if np.any(valid):
                closeness[valid] = 1.0 - (np.sqrt(min_dist2[valid]) / SCAN_RADIUS)
            feature_grid[:, col] = closeness

    # --- Reshape ---
    feature_grid = feature_grid.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, total_feat)
    mapping_grid = mapping_grid.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)

    return feature_grid[None, ...], mapping_grid[None, ...]