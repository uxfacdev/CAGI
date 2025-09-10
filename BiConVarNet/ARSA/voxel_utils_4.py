import numpy as np
from typing import Tuple, Dict, Optional

AA3_LIST = [
    'ALA','CYS','ASP','GLU','PHE','GLY','HIS',
    'ILE','LYS','LEU','MET','ASN','PRO','GLN',
    'ARG','SER','THR','VAL','TRP','TYR','UNK'
]
ATOM_TYPES = ['CA', 'CB']
AA3_TO_INDEX = {aa: i for i, aa in enumerate(AA3_LIST)}

# 3↔1 letter 매핑/정규화
AA3_TO_AA1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I',
    'LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S',
    'THR':'T','VAL':'V','TRP':'W','TYR':'Y','UNK':'X'
}
_ALIAS3 = {"HSD":"HIS","HSE":"HIS","HSP":"HIS","HID":"HIS","HIE":"HIS","HIP":"HIS","MSE":"MET","SEC":"CYS","PYL":"LYS"}

MAX_ASA = {'ALA':129.0,'ARG':274.0,'ASN':195.0,'ASP':193.0,'CYS':167.0,'GLN':225.0,'GLU':223.0,'GLY':104.0,
           'HIS':224.0,'ILE':197.0,'LEU':201.0,'LYS':236.0,'MET':224.0,'PHE':240.0,'PRO':159.0,'SER':155.0,
           'THR':172.0,'TRP':285.0,'TYR':263.0,'VAL':174.0} 

def _norm_aa3(x: str) -> str:
    x = (x or "").upper()
    return _ALIAS3.get(x, x if x in AA3_TO_AA1 else "UNK")

SCAN_RADIUS = 5.0
VOXEL_SIZE  = 2.0
GRID_SIZE   = 7

def get_voxel_centers(center_coord: np.ndarray) -> np.ndarray:
    half = GRID_SIZE // 2
    shifts = (np.arange(-half, half + 1, dtype=np.float32) * VOXEL_SIZE)
    gx, gy, gz = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    offsets = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    return center_coord.astype(np.float32) + offsets

def _rsa_map_biopython(pdb_path: str, chain_id: Optional[str] = None,
                       use_standard: bool = True) -> Optional[Dict[int, float]]:
    from Bio.PDB import PDBParser, ShrakeRupley
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    sr = ShrakeRupley(); sr.compute(structure, level="R")

    # 체인별 {resnum: (resname3_norm, ASA)}
    chains = {}
    for model in structure:
        for chain in model:
            c = chain.id
            m = chains.setdefault(c, {})
            for res in chain:
                hetflag, resnum, icode = res.id
                if hetflag != " ":  # HETATM/WAT 제외
                    continue
                res3 = _norm_aa3(res.get_resname())
                asa  = float(res.xtra.get("EXP_ASA", 0.0))
                m[resnum] = (res3, asa)
    if not chains:
        return None

    # 대상 체인 선택
    if chain_id is not None and chain_id in chains:
        chosen = chains[chain_id]
    else:
        chosen = chains[next(iter(chains))]

    if use_standard:
        # 1) 표준 RSA 시도: ASA / MAX_ASA[res3]
        rsa = {}
        need_fallback = False
        asa_vals = []
        for rnum, (res3, asa) in chosen.items():
            asa_vals.append(asa)
            maxasa = MAX_ASA.get(res3)
            if maxasa and maxasa > 1e-6:
                rsa[rnum] = float(min(max(asa / maxasa, 0.0), 1.0))
            else:
                need_fallback = True

        if not need_fallback:
            return rsa  # 전부 표준 RSA 성공

        # 2) 일부 잔기에서 MAX_ASA 미매칭 → 해당 잔기만 분위수 폴백으로 채움
        asa_vals = np.array(asa_vals, dtype=np.float32)
        scale = float(np.percentile(asa_vals, 95)) if asa_vals.size else 1.0
        scale = scale if scale > 1e-6 else 1.0
        for rnum, (res3, asa) in chosen.items():
            if rnum not in rsa:
                rsa[rnum] = float(min(max(asa / scale, 0.0), 1.0))
        return rsa

    # 상대 RSA(현행 fallback)만 쓰고 싶을 때
    asa_vals = np.array([asa for (_, asa) in chosen.values()], dtype=np.float32)
    scale = float(np.percentile(asa_vals, 95)) if asa_vals.size else 1.0
    scale = scale if scale > 1e-6 else 1.0
    return {rnum: float(min(max(asa / scale, 0.0), 1.0))
            for rnum, (_, asa) in chosen.items()}


def _map_res_scalar_to_vox(mapping_grid: np.ndarray, res_scalar: Dict[int,float], default: float = 0.0) -> np.ndarray:
    g = mapping_grid[0]
    out = np.empty_like(g, dtype=np.float32)
    it = np.nditer(g, flags=['multi_index'])
    while not it.finished:
        r = int(it[0])
        out[it.multi_index] = float(res_scalar.get(r, default))
        it.iternext()
    return out[None, ..., None]

def compute_voxel_features_structural(
    pdb_path: str,
    mut_pos: int,
    chain_id: Optional[str] = None,
    add_dpos2: bool = True,
    add_plddt: bool = True,
    add_rsa: bool = False,
    add_resdepth: bool = False,
    add_gnm_msf: bool = True,
    wt1: Optional[str] = None,       # ✅ 추가: TSV의 WT 1-letter (옵션)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:  # ✅ QC dict 함께 반환
    """
    Returns:
      feature_grid: (1,7,7,7,C)
      mapping_grid: (1,7,7,7)
      qc: {
        'wt_expected': str|None,
        'obs_center_aa3': str,
        'obs_center_aa1': str,
        'center_maps_to_mut_pos': bool,
        'center_resnum': int,
        'wt_mismatch': bool,
        'notes': [str,...]
      }
    """
    from prody import parsePDB, GNM, calcSqFlucts

    st_all = parsePDB(pdb_path)

    sel_all = "protein and name CA CB"
    if chain_id is not None:
        sel_all = f"protein and chain {chain_id} and name CA CB"

    atoms_all = st_all.select(sel_all)
    if atoms_all is None or atoms_all.numAtoms() == 0:
        raise ValueError("No CA or CB atoms found with the given selection.")

    coords_all   = atoms_all.getCoords().astype(np.float32)
    names_all    = atoms_all.getNames()
    resnames_all = atoms_all.getResnames()
    resnums_all  = atoms_all.getResnums().astype(np.int32)
    chids_all    = atoms_all.getChids()

    # --- CA-only sets
    ca_mask_all = (names_all == 'CA')
    if not np.any(ca_mask_all):
        raise ValueError("No CA atoms found in selection.")
    ca_resnums  = resnums_all[ca_mask_all]
    ca_coords   = coords_all[ca_mask_all]
    ca_resnames = resnames_all[ca_mask_all]
    ca_chids    = chids_all[ca_mask_all]

    if chain_id is not None:
        hit = np.where((ca_resnums == mut_pos) & (ca_chids == chain_id))[0]
    else:
        hit = np.where(ca_resnums == mut_pos)[0]
    if hit.size == 0:
        raise ValueError(f"mut_pos {mut_pos} not found (check chain/resnum).")
    ci = int(hit[0])
    center_coord = ca_coords[ci]

    # --- voxel centers
    voxel_centers = get_voxel_centers(center_coord)
    V = voxel_centers.shape[0]
    R2 = np.float32(SCAN_RADIUS * SCAN_RADIUS)

    # --- 42ch closeness
    feat_42 = np.zeros((V, 42), dtype=np.float32)
    idx_cache = {}
    for atom_type in ATOM_TYPES:
        for aa in AA3_LIST:
            if atom_type == 'CB' and aa == 'GLY':
                idx_cache[(atom_type, aa)] = np.empty((0,), dtype=np.int32)
            else:
                mask = (names_all == atom_type) & (resnames_all == aa)
                idx_cache[(atom_type, aa)] = np.where(mask)[0].astype(np.int32)

    for atom_type_idx, atom_type in enumerate(ATOM_TYPES):
        for aa_idx, aa in enumerate(AA3_LIST):
            col = aa_idx + 21 * atom_type_idx
            idxs = idx_cache[(atom_type, aa)]
            if idxs.size == 0: continue
            sub_coords = coords_all[idxs]
            diff = voxel_centers[:, None, :] - sub_coords[None, :, :]
            dist2 = np.einsum('vka,vka->vk', diff, diff)
            dist2_masked = np.where(dist2 <= R2, dist2, np.inf).astype(np.float32)
            min_dist2 = np.min(dist2_masked, axis=1)
            valid = np.isfinite(min_dist2)
            closeness = np.zeros((V,), dtype=np.float32)
            if np.any(valid):
                closeness[valid] = 1.0 - (np.sqrt(min_dist2[valid]) / SCAN_RADIUS)
            feat_42[:, col] = closeness

    # --- CA-only nearest residue per voxel (mapping)
    diff_ca = voxel_centers[:, None, :] - ca_coords[None, :, :]
    dist2_ca = np.einsum('vka,vka->vk', diff_ca, diff_ca)
    nearest_ca_idx = dist2_ca.argmin(axis=1)
    nearest_resnums = ca_resnums[nearest_ca_idx]
    mapping_grid = nearest_resnums.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)

    # --- extras
    extra = []

    # Δpos 2ch
    if add_dpos2:
        L_est = int(max(1, ca_resnums.max() - ca_resnums.min() + 1))
        delta = nearest_resnums.astype(np.int32) - int(mut_pos)
        signed_norm = np.clip(delta.astype(np.float32) / float(L_est), -1.0, 1.0)[:, None]
        abs_norm    = np.minimum(np.abs(signed_norm), 1.0)
        extra += [signed_norm, abs_norm]

    # pLDDT
    if add_plddt:
        ca_sel = st_all.select("protein and name CA" + (f" and chain {chain_id}" if chain_id else ""))
        ca_betas = ca_sel.getBetas().astype(np.float32)
        plddt_by_resnum = {int(r): float(b) for r, b in zip(ca_resnums, ca_betas)}
        ch = _map_res_scalar_to_vox(mapping_grid[None, ...], plddt_by_resnum, default=0.0)
        extra.append(ch.reshape(-1,1))

    # RSA & ResidueDepth
    rsa_by_resnum: Optional[Dict[int,float]] = None
    if add_rsa or add_resdepth:
        rsa_by_resnum = _rsa_map_biopython(pdb_path, chain_id=chain_id, use_standard=True)
    if add_rsa:
        if rsa_by_resnum is None:
            local_ct = (dist2_ca <= (6.0**2)).sum(axis=1) - 1
            proxy = (1.0 / (1.0 + np.maximum(local_ct, 0)/10.0)).astype(np.float32)[:, None]
            extra.append(proxy)
        else:
            ch = _map_res_scalar_to_vox(mapping_grid[None, ...], rsa_by_resnum, default=0.0)
            extra.append(ch.reshape(-1,1))
    if add_resdepth:
        if rsa_by_resnum is None:
            local_ct = (dist2_ca <= (6.0**2)).sum(axis=1) - 1
            proxy = (1.0 / (1.0 + np.maximum(local_ct, 0)/10.0)).astype(np.float32)
            depth = (1.0 - proxy)[:, None]
            extra.append(depth)
        else:
            depth_by_resnum = {r: float(1.0 - v) for r, v in rsa_by_resnum.items()}
            ch = _map_res_scalar_to_vox(mapping_grid[None, ...], depth_by_resnum, default=0.0)
            extra.append(ch.reshape(-1,1))

    # GNM-MSF
    if add_gnm_msf:
        try:
            ca_sel = st_all.select("protein and name CA" + (f" and chain {chain_id}" if chain_id else ""))
            gnm = GNM('gnm'); gnm.buildKirchhoff(ca_sel, cutoff=10.0); gnm.calcModes()
            from prody import calcSqFlucts
            msf = calcSqFlucts(gnm[:]).astype(np.float32)
            msf_by_resnum = {int(r): float(v) for r, v in zip(ca_resnums, msf)}
            ch = _map_res_scalar_to_vox(mapping_grid[None, ...], msf_by_resnum, default=float(np.mean(msf)))
            extra.append(ch.reshape(-1,1))
        except Exception:
            deg = np.maximum((dist2_ca <= (10.0**2)).sum(axis=1) - 1, 1)
            msf_proxy = (1.0/deg).astype(np.float32)[:, None]
            extra.append(msf_proxy)

    # concat features
    feat = feat_42
    if extra:
        feat = np.concatenate([feat] + extra, axis=1)
    feature_grid = feat.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, -1)[None, ...]
    mapping_grid = mapping_grid[None, ...]

    # ---------- QC: 중앙 보셀 WT/매핑 확인 ----------
    obs_center_aa3 = _norm_aa3(ca_resnames[ci])
    obs_center_aa1 = AA3_TO_AA1.get(obs_center_aa3, "X")
    center_maps_to_mut_pos = (int(mapping_grid[0,3,3,3]) == int(mut_pos))
    wt_expected = (wt1.upper().strip() if isinstance(wt1, str) and wt1 else None)
    wt_mismatch = (wt_expected is not None and obs_center_aa1 != wt_expected)
    notes = []
    if not center_maps_to_mut_pos:
        notes.append("Center voxel mapping != mut_pos")
    if wt_mismatch:
        notes.append("WT mismatch (PDB CA != TSV WT)")

    qc = {
        "wt_expected": wt_expected,
        "obs_center_aa3": obs_center_aa3,
        "obs_center_aa1": obs_center_aa1,
        "center_maps_to_mut_pos": bool(center_maps_to_mut_pos),
        "center_resnum": int(mut_pos),
        "wt_mismatch": bool(wt_mismatch),
        "notes": notes,
    }
    return feature_grid, mapping_grid, qc
