import numpy as np
from prody import parsePDB
from typing import Tuple

# Constants
AA3_LIST = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS',
    'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN',
    'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'UNK'
] # 3글자 아미노산 코드 20개 + 'UNK' = 총 21개, 'UNK'는 알 수 없는 아미노산에 대한 placeholder


ATOM_TYPES = ['CA', 'CB']
AA3_TO_INDEX = {aa: i for i, aa in enumerate(AA3_LIST)}

SCAN_RADIUS = 5.0  # Å
VOXEL_SIZE = 2.0   # Å
GRID_SIZE = 7      # 7x7x7 voxel grid

def get_voxel_centers(center_coord: np.ndarray) -> np.ndarray:
    """Generate voxel center coordinates for the 7x7x7 grid centered at center_coord."""
    half_grid = GRID_SIZE // 2
    shifts = np.arange(-half_grid, half_grid + 1) * VOXEL_SIZE
    offsets = np.stack(np.meshgrid(shifts, shifts, shifts, indexing="ij"), axis=-1).reshape(-1, 3)
    return center_coord + offsets  # shape (343, 3)

def compute_atomic_distance_profile(pdb_path: str, mut_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a voxel grid of shape (1, 7, 7, 7, 63) representing atomic closeness profiles
    and one-hot residue identity, and a mapping grid (1, 7, 7, 7) indicating closest residue index.

    Args:
        pdb_path: Path to AlphaFold PDB file.
        mut_pos: 1-based mutation position.

    Returns:
        feature_grid: shape (1, 7, 7, 7, 63)
        mapping_grid: shape (1, 7, 7, 7)
    """
    structure = parsePDB(pdb_path)
    atoms = structure.select('name CA CB')
    if atoms is None or len(atoms) == 0:
        raise ValueError("No CA or CB atoms found in structure.")

    coords = atoms.getCoords()
    names = atoms.getNames()
    resnames = atoms.getResnames()
    resnums = atoms.getResnums()

    # Identify mutation center (CA of the mutated residue)
    ca_indices = [i for i, name in enumerate(names) if name == 'CA']
    ca_resnums = resnums[ca_indices]
    ca_target_idx = np.where(ca_resnums == mut_pos)[0]
    if len(ca_target_idx) == 0:
        raise ValueError(f"Mutation site {mut_pos} not found in CA atoms.")
    center_coord = coords[ca_indices[ca_target_idx[0]]]

    # Step 2: Define voxel grid centers
    voxel_centers = get_voxel_centers(center_coord)  # shape (343, 3)

    # Step 3: Init output arrays
    feature_grid = np.zeros((GRID_SIZE ** 3, 63), dtype=np.float32)
    mapping_grid = np.full((GRID_SIZE ** 3,), -1, dtype=np.int32)

    # Step 4: Compute features for each voxel
    for vi, vcenter in enumerate(voxel_centers):
        # 4-1: Closeness (42 features)
        for atom_type_idx, atom_type in enumerate(ATOM_TYPES):
            for aa_idx, aa in enumerate(AA3_LIST):
                if atom_type == 'CB' and aa == 'GLY':
                    continue  # GLY has no CB atom
                mask = np.logical_and(np.char.equal(names, atom_type), np.char.equal(resnames, aa))
                if not np.any(mask):
                    continue
                aa_coords = coords[mask]
                dists = np.linalg.norm(aa_coords - vcenter, axis=1)
                within_r = dists[dists <= SCAN_RADIUS]
                if len(within_r) == 0:
                    continue
                min_d = np.min(within_r)
                closeness = 1.0 - (min_d / SCAN_RADIUS)
                feature_grid[vi, aa_idx + 21 * atom_type_idx] = closeness

        # 4-2: Closest residue mapping
        dists_all = np.linalg.norm(coords - vcenter, axis=1)
        min_idx = np.argmin(dists_all)
        closest_resnum = resnums[min_idx]
        mapping_grid[vi] = closest_resnum

        # 4-3: One-hot residue type
        closest_resname = resnames[min_idx]
        aa_index = AA3_TO_INDEX.get(closest_resname, 20)  # Use 'UNK'=20 if unknown
        feature_grid[vi, 42 + aa_index] = 1.0  # Append at index 42-62

    # Step 5: Reshape
    feature_grid = feature_grid.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, 63)
    mapping_grid = mapping_grid.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)

    return feature_grid[np.newaxis], mapping_grid[np.newaxis]  # Add batch dim
