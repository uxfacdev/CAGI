import os
import numpy as np
import prody as pr
import pandas as pd
import warnings

AA1_TO_AA3 = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS','I':'ILE',
              'K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG','S':'SER',
              'T':'THR','V':'VAL','W':'TRP','Y':'TYR'} # 표준 1→3 letter 매핑. 문제 없음.
VOL = {'ALA':88.6,'CYS':108.5,'ASP':111.1,'GLU':138.4,'PHE':189.9,'GLY':60.1,'HIS':153.2,'ILE':166.7,
       'LYS':168.6,'LEU':166.7,'MET':162.9,'ASN':114.1,'PRO':112.7,'GLN':143.8,'ARG':173.4,'SER':89.0,
       'THR':116.1,'VAL':140.0,'TRP':227.8,'TYR':193.6} # 부피(Å³) 값들은 널리 쓰이는 표로, IMGT 레퍼런스에 같은 숫자
HYD = {'ALA':1.8,'CYS':2.5,'ASP':-3.5,'GLU':-3.5,'PHE':2.8,'GLY':-0.4,'HIS':-3.2,'ILE':4.5,
       'LYS':-3.9,'LEU':3.8,'MET':1.9,'ASN':-3.5,'PRO':-1.6,'GLN':-3.5,'ARG':-4.5,'SER':-0.8,
       'THR':-0.7,'VAL':4.2,'TRP':-0.9,'TYR':-1.3} # Kyte–Doolittle hydropathy 지수 그대로(예: ILE 4.5, ARG −4.5). 표준 값과 일치 (IMGT)
CHG = {'ASP':-1,'GLU':-1,'LYS':+1,'ARG':+1,'HIS':0} # HIS 0?
MAX_ASA = {'ALA':129.0,'ARG':274.0,'ASN':195.0,'ASP':193.0,'CYS':167.0,'GLN':225.0,'GLU':223.0,'GLY':104.0,
           'HIS':224.0,'ILE':197.0,'LEU':201.0,'LYS':236.0,'MET':224.0,'PHE':240.0,'PRO':159.0,'SER':155.0,
           'THR':172.0,'TRP':285.0,'TYR':263.0,'VAL':174.0} # “최대 용매 접근면적” 정상화 값, RSA 계산 표준으로 가장 많이 인용되는 스케일 중 하나

def _dihedral(p0,p1,p2,p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2
    b1 = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0,b1)*b1
    w = b2 - np.dot(b2,b1)*b1
    x = np.dot(v,w); y = np.dot(np.cross(b1,v), w)
    return np.arctan2(y,x)

def _delta_physchem(wt3, mut3):
    d_vol = VOL.get(mut3, 150.0) - VOL.get(wt3, 150.0)
    d_hyd = HYD.get(mut3, 0.0)   - HYD.get(wt3, 0.0)
    d_chg = CHG.get(mut3, 0)     - CHG.get(wt3, 0)

    return float(d_vol), float(d_hyd), float(d_chg)

def _normalize_resname(name: str) -> str:
    name = name.upper()
    alias = {
        # His variants (force to HIS)
        "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
        # Common alternates
        "MSE": "MET",  # Selenomethionine
        "SEC": "CYS",  # Selenocysteine
        "PYL": "LYS",  # Pyrrolysine (rare; map to LYS if encountered)
    }
    return alias.get(name, name)

def load_ddg_map(tsv_path: str, key_col="mut", value_col="ddg_fold") -> dict[str, float]:
    """
    ddg_fold.tsv를 읽어 변이키 -> 값 맵 생성.
    - key 예: 'IA23V;'  (세미콜론 포함)
    - 중복 키가 있으면 평균값으로 집계
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    if key_col not in df or value_col not in df:
        raise ValueError(f"columns '{key_col}', '{value_col}' not found in {tsv_path}")
    df = df[[key_col, value_col]].copy()
    df[key_col] = df[key_col].str.strip().str.upper()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    g = df.groupby(key_col, as_index=True)[value_col].mean()
    ddg = g.to_dict()
    # 키 편의상 세미콜론 제거 버전도 함께 매핑
    for k, v in list(ddg.items()):
        k2 = k.rstrip(";")
        ddg.setdefault(k2, v)
    return ddg

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def calc_msa_conservation(msa_seqs, mut_pos, wt, mut):
    col = [seq[mut_pos] for seq in msa_seqs if len(seq) > mut_pos]
    counts = {aa:0 for aa in AA_LIST}
    for aa in col:
        if aa in counts: counts[aa]+=1
    total = sum(counts.values())
    if total == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    freqs = np.array([counts[a]/total for a in AA_LIST], dtype=np.float32)
    p_wt, p_mut = freqs[AA_LIST.index(wt)], freqs[AA_LIST.index(mut)]

    # ΔPSIC
    psic = -np.log(p_mut+1e-8) + np.log(p_wt+1e-8)

    # Shannon entropy (raw)
    H = -np.sum(freqs * np.log(freqs+1e-8))

    return float(psic), float(H)


def _evoef2_key(wt1: str, pos: int, mut1: str, chain_id: str = "A", with_semicolon: bool = True) -> str:
    k = f"{wt1.upper()}{chain_id}{int(pos)}{mut1.upper()}"
    return k + ";" if with_semicolon else k

def extract_struct_S(
    pdb_path: str, mut_pos: int, wt1: str, mut1: str,
    neigh_r: float = 8.0, chain_id: str = 'A',
    ddg_map_fold: dict[str, float] | None = None,
    ddg_map_bind: dict[str, float] | None = None,
    uid: str | None = None, 
    msa_dict: dict | None = None
):

    wt3 = AA1_TO_AA3[wt1.upper()]
    mut3 = AA1_TO_AA3[mut1.upper()]

    st_all = pr.parsePDB(pdb_path)
    ca = st_all.select(f'protein and chain {chain_id} and name CA') # ProDy로 PDB 전체를 파싱, 체인 A의 알파탄소(CA)만 추출
    if ca is None:
        raise ValueError(f"No CA atoms found for chain {chain_id}.")

    ca_coords   = ca.getCoords().astype(np.float32) # 각 ca의 3d 좌표
    ca_resnums  = ca.getResnums().astype(int) # 잔기 번호
    ca_resnames = ca.getResnames()  
    betas       = ca.getBetas().astype(np.float32) # pLDDT (0-100)

    hit = np.where(ca_resnums == mut_pos)[0] # 변이 중심 잔기 찾기
    if hit.size == 0:
        raise ValueError(f"mut_pos {mut_pos} not found in chain {chain_id}.")
    ci = int(hit[0]); cpos = ca_coords[ci] # 변이 위치(mut_pos)와 같은 resnum을 가진 Cα 인덱스 ci를 찾고, 그 좌표 cpos를 잡음.

    obs3 = _normalize_resname(ca_resnames[ci])
    exp3 = AA1_TO_AA3[wt1.upper()]
    if obs3 != exp3:
        msg = (f"[WT mismatch] chain={chain_id} pos={mut_pos}: "
            f"expected WT={exp3} (from input '{wt1}'), but PDB has {obs3}. "
            "Check numbering/chain/WT letter.")
        
        WT_CHECK_MODE = "error"
        if WT_CHECK_MODE == "error":
            raise ValueError(msg)
        elif WT_CHECK_MODE == "warn":
            warnings.warn(msg)

    d_ca = np.linalg.norm(ca_coords - cpos, axis=1) # 각 Cα 좌표에서 중심 Cα 좌표 cpos를 뺀 벡터들의 길이를 구해서, 중심으로부터의 유클리드 거리(Å) 배열을 만든 거
    neigh_idx = np.where(d_ca <= neigh_r)[0]
    neigh_idx = neigh_idx[neigh_idx != ci]   # 중심 Cα와의 유클리드 거리로 반경 8Å 이웃을 구하고, 자기 자신은 제외

    # 1) pLDDT
    plddt_c = float(betas[ci]) # 센터: 해당 위치 pLDDT (betas[ci])
    if neigh_idx.size: # 주변 평균/최소/최대: 반경 8Å 내 이웃들만 대상으로 계산
        nb = betas[neigh_idx]
        plddt_mean, plddt_min, plddt_max = float(nb.mean()), float(nb.min()), float(nb.max())
    else:
        plddt_mean = plddt_min = plddt_max = plddt_c

    # 2) GNM-MSF
    try:
        from prody import GNM, calcSqFlucts
        gnm = GNM('gnm')
        gnm.buildKirchhoff(ca, cutoff=10.0) # Kirchhoff(접촉 그래프) 만들기
        
        gnm.calcModes()  # 고유값 / 고유벡터 계산
        msf = calcSqFlucts(gnm[:]).astype(np.float32) # 각 잔기의 MSF 구하기
    except Exception:
        deg = max((d_ca <= 10.0).sum() - 1, 1)
        msf = np.ones_like(d_ca, dtype=np.float32) * (1.0 / deg)
    msf_c    = float(msf[ci]) # 중심 잔기 값
    msf_mean = float(msf[neigh_idx].mean()) if neigh_idx.size else msf_c # 반경 8 Å 이웃(센터 제외)의 평균

    # 3) LRCO
    js = np.where(d_ca <= 8.0)[0]
    js = js[js != ci] # 접촉 기준 8Å 내부의 이웃 집합 js를 사용

    prot_len = int(ca_resnums.max())
    lrco = float(np.abs(ca_resnums[js] - ca_resnums[ci]).mean() / max(prot_len,1)) if js.size else 0.0 # 각 이웃 j에 대해 서열번호 차 |resnum_j − resnum_i|를 구해 평균.

    # 4) Δ물성
    d_vol, d_hyd, d_chg = _delta_physchem(wt3, mut3)

    # 6) Energetics: ddg_fold/ddg_bind 맵에서 채우기
    ddg_fold = np.nan
    # ddg_bind = np.nan

    if ddg_map_fold is not None:
        # UniProt별 맵인지 확인
        if isinstance(ddg_map_fold, dict) and uid in ddg_map_fold and isinstance(ddg_map_fold[uid], dict):
            local_map = ddg_map_fold[uid]
        else:
            local_map = ddg_map_fold

        k1 = _evoef2_key(wt1, mut_pos, mut1, chain_id, with_semicolon=True)
        k2 = k1.rstrip(";")
        ddg_fold = float(local_map.get(k1, local_map.get(k2, np.nan)))
    # if ddg_map_bind is not None:
    #     k1 = _evoef2_key(wt1, mut_pos, mut1, chain_id, with_semicolon=True)
    #     k2 = k1.rstrip(";")
        # ddg_bind = float(ddg_map_bind.get(k1, ddg_map_bind.get(k2, np.nan)))

    # 7) Backbone torsion
    def _pick(name, resnum):
        sel = st_all.select(f"protein and chain {chain_id} and resnum {resnum} and name {name}")
        return sel.getCoords()[0] if (sel is not None and sel.numAtoms()>0) else None
    pN   = _pick('N',  mut_pos)
    pCA  = _pick('CA', mut_pos)
    pC   = _pick('C',  mut_pos)
    pCm1 = _pick('C',  mut_pos - 1)
    pNp1 = _pick('N',  mut_pos + 1)
    phi = psi = 0.0
    if pN is not None and pCA is not None and pC is not None:
        if pCm1 is not None: phi = float(_dihedral(pCm1, pN, pCA, pC))
        if pNp1 is not None: psi = float(_dihedral(pN, pCA, pC, pNp1))
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)

    # # 8) Residue depth
    # res_depth = float(1.0 - rsa_center) if not np.isnan(rsa_center) else np.nan

    # 9) Gly/Pro
    gly_switch = 1.0 if (AA1_TO_AA3[wt1.upper()]=='GLY') ^ (AA1_TO_AA3[mut1.upper()]=='GLY') else 0.0
    pro_switch = 1.0 if (AA1_TO_AA3[wt1.upper()]=='PRO') ^ (AA1_TO_AA3[mut1.upper()]=='PRO') else 0.0

    # inside extract_struct_S
    psic, H = np.nan, np.nan
    if msa_dict is not None and uid in msa_dict:
        first_elem = msa_dict[uid][0][0]
        if isinstance(first_elem, str):
            # 원본 msa_dict 버전
            msa_seqs = [s for _, s in msa_dict[uid]]
            psic, H = calc_msa_conservation(msa_seqs, mut_pos-1, wt1, mut1)
        elif isinstance(first_elem, dict):
            # precomputed 버전
            counts, total = msa_dict[uid][mut_pos-1]
            if total > 0:
                p_wt = counts.get(wt1, 0) / total
                p_mut = counts.get(mut1, 0) / total
                psic = -np.log(p_mut+1e-8) + np.log(p_wt+1e-8)
                freqs = np.array([counts[a]/total for a in AA_LIST], dtype=np.float32)
                H = -np.sum(freqs * np.log(freqs+1e-8))

    S = np.array([
        plddt_c, plddt_mean, plddt_min, plddt_max,
        msf_c, msf_mean,
        lrco,
        d_vol, d_hyd, d_chg,
        ddg_fold,
        sin_phi, cos_phi, sin_psi, cos_psi,
        gly_switch, pro_switch, psic, H
    ], dtype=np.float32)

    names = [
        "plddt_center","plddt_neigh_mean","plddt_neigh_min","plddt_neigh_max",
        "msf_center","msf_neigh_mean",
        "lrco",
        "d_volume","d_hydropathy","d_charge",
        "ddg_fold",
        "sin_phi","cos_phi","sin_psi","cos_psi",
        "gly_switch","pro_switch", "delta_psic","shannon"
    ]
    return S, names