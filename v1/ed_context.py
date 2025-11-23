# ed_context.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
import logging

logger = logging.getLogger("EDCtx")

try:
    from xtb.interface import Calculator, Param, CalculationMethod
    HAS_XTB = True
except Exception:
    HAS_XTB = False
    #logger.warning("xtb-python 不可用，将使用 Gasteiger 电荷作为近似。")

def _embed_if_needed(m):
    if m.GetNumConformers() == 0:
        AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    return m

def compute_partial_charges(mol, method="auto"):
    # 优先 xTB，失败或未安装则用 Gasteiger（非常快）
    if method == "auto" and HAS_XTB:
        try:
            m = Chem.AddHs(Chem.Mol(mol))
            _embed_if_needed(m)
            conf = m.GetConformer()
            xyz = np.asarray(conf.GetPositions(), float)
            nums = np.asarray([a.GetAtomicNum() for a in m.GetAtoms()], int)
            calc = Calculator(CalculationMethod.GFN2xTB, nums, xyz)
            calc.set_settings(Param())
            res = calc.singlepoint()
            ch = np.array(res.get_charges(), dtype=np.float64)[:mol.GetNumAtoms()]
            return np.nan_to_num(ch, 0.0, 0.0, 0.0)
        except Exception as e:
            logger.warning(f"xTB 计算失败，改用 Gasteiger: {e}")

    m = Chem.Mol(mol)
    _embed_if_needed(m)
    rdPartialCharges.ComputeGasteigerCharges(m)
    charges = []
    for a in m.GetAtoms():
        try:
            charges.append(float(a.GetProp('_GasteigerCharge')))
        except Exception:
            charges.append(0.0)
    return np.nan_to_num(np.array(charges, dtype=np.float64), 0.0, 0.0, 0.0)

def fibonacci_sphere(n=64):
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2*i/n)
    theta = np.pi * (1 + 5**0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)

def esp_at_points(mol, points, charges=None, eps=1e-6):
    conf = mol.GetConformer()
    xyz = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=np.float64)
    if charges is None:
        charges = compute_partial_charges(mol)
    diff = points[:, None, :] - xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) + eps
    esp = (charges[None, :] / dist).sum(axis=1)
    return esp.astype(np.float32)

def _centers_auto(m):
    # 若有 dummy(原子号0)就用 dummy 坐标；否则用分子质心
    conf = m.GetConformer()
    d_idx = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(d_idx) > 0:
        centers = []
        for i in d_idx:
            p = conf.GetAtomPosition(i)
            centers.append(np.array([p.x, p.y, p.z], dtype=np.float64))
        return centers
    pos = np.array([conf.GetAtomPosition(i) for i in range(m.GetNumAtoms())], dtype=np.float64)
    return [pos.mean(axis=0)]

def build_ed_context_vector(
    mol,
    radii=(1.5, 2.5, 3.5),
    n_dir=64,
    stats=("mean","std","posfrac","p10","p90"),
    charge_method="auto"
):
    """
    返回一个长度 = len(radii)*len(stats) 的小向量（默认 3*5=15 维）。
    """
    m = Chem.Mol(mol)
    _embed_if_needed(m)
    dirs = fibonacci_sphere(n_dir)
    charges = compute_partial_charges(m, method=charge_method)
    centers = _centers_auto(m)
    per_center = []
    for c in centers:
        feat_r = []
        for r in radii:
            pts = c[None, :] + r * dirs
            vals = esp_at_points(m, pts, charges=charges)
            vn = vals / (np.linalg.norm(vals) + 1e-8)  # 每个壳层做幅度归一
            fs = []
            for s in stats:
                if s == "mean": fs.append(float(vn.mean()))
                elif s == "std": fs.append(float(vn.std()))
                elif s == "posfrac": fs.append(float((vn > 0).mean()))
                elif s == "p10": fs.append(float(np.percentile(vn, 10)))
                elif s == "p90": fs.append(float(np.percentile(vn, 90)))
            feat_r.extend(fs)
        per_center.append(np.array(feat_r, dtype=np.float32))
    if not per_center:
        return np.zeros((len(radii)*len(stats),), dtype=np.float32)
    vec = np.mean(np.stack(per_center, axis=0), axis=0)  # 多连接点平均
    return vec  # numpy [D]