import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.utils import to_undirected, to_dense_adj
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors, BRICS, GetMolFrags
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.Chem import QED
try:
    from rdkit.Contrib.SA_Score import sascorer
    def calculate_sa_score(mol): return sascorer.calculateScore(mol)
except ImportError:
    sascorer = None
    def calculate_sa_score(mol): return 5.0
from rdkit import Chem, DataStructs, RDLogger, rdBase, RDConfig
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import logging
import pickle
import requests
import matplotlib.pyplot as plt
import math
from tabulate import tabulate

# 最终引入正确的EGNN实现
from egnn_pytorch import EGNN

# =============================================================================
# 1. 基础设置
# =============================================================================
RDLogger.logger().setLevel(RDLogger.CRITICAL)
logger = logging.getLogger("BioIsostericDiffusion")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler("bioisosteric_diffusion_final_v34.log"); fh.setFormatter(formatter); logger.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(formatter); logger.addHandler(sh)

seed = 42; np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# =============================================================================
# 2. 核心模型与损失函数定义
# =============================================================================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07): super().__init__(); self.temperature = temperature
    def forward(self, features, labels):
        features = torch.nn.functional.normalize(features, dim=1); sim_matrix = torch.matmul(features, features.T) / self.temperature; mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float() - torch.eye(features.size(0), device=device)
        exp_sim = torch.exp(sim_matrix); sum_exp = torch.sum(exp_sim * (1 - torch.eye(exp_sim.size(0), device=device)), dim=1, keepdim=True); log_prob = sim_matrix - torch.log(sum_exp + 1e-9); mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9); return -mean_log_prob.mean()

class EdgePredictor(nn.Module):
    def __init__(self, node_in_channels, hidden_channels):
        super().__init__(); self.node_conv1 = GCNConv(node_in_channels, hidden_channels); self.node_conv2 = GCNConv(hidden_channels, hidden_channels); self.mlp = nn.Sequential(nn.Linear(2 * hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_channels, 1))
    def forward(self, x, edge_index):
        h = F.relu(self.node_conv1(x, edge_index)); h = self.node_conv2(h, edge_index); start_nodes = h[edge_index[0]]; end_nodes = h[edge_index[1]]; edge_representation = torch.cat([start_nodes, end_nodes], dim=-1); return self.mlp(edge_representation).squeeze(-1)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, time):
        half_dim = self.dim // 2; embeddings = math.log(10000) / (half_dim - 1); embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]; embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1); return embeddings

class EGNNWrapper(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=4):
        super().__init__(); self.embedding_in = nn.Linear(in_dim, hidden_dim); self.egnn_layers = nn.ModuleList([EGNN(dim = hidden_dim) for _ in range(n_layers)]); self.embedding_out = nn.Linear(hidden_dim, out_dim)
    def forward(self, x, pos, edge_index):
        h = self.embedding_in(x); adj_mat = to_dense_adj(edge_index, max_num_nodes=x.size(0)).bool()
        h = h.unsqueeze(0); pos = pos.unsqueeze(0)
        for egnn in self.egnn_layers: h, pos = egnn(feats=h, coors=pos, adj_mat=adj_mat)
        h = h.squeeze(0); return self.embedding_out(h)

class EGNNPharmacophoreEncoder(nn.Module):
    def __init__(self, pharm_feature_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__(); self.egnn_net = EGNNWrapper(in_dim=pharm_feature_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, n_layers=n_layers); self.output_proj = nn.Linear(hidden_dim, out_dim)
    def forward(self, pharm_feats, pharm_pos):
        num_points = pharm_feats.size(0)
        if num_points == 0: return torch.zeros(self.output_proj.out_features, device=pharm_feats.device)
        if num_points < 2: point_embeddings = self.egnn_net.embedding_in(pharm_feats)
        else:
            src, dst = [], []
            for i in range(num_points):
                for j in range(num_points):
                    if i != j: src.append(i); dst.append(j)
            edge_index = torch.tensor([src, dst], device=pharm_feats.device, dtype=torch.long)
            point_embeddings = self.egnn_net(pharm_feats, pharm_pos, edge_index)
        global_embedding = point_embeddings.mean(dim=0); return self.output_proj(global_embedding)

class JointDiffusionGenerator(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_atom_types, time_dim=64, n_layers=4, num_timesteps=100):
        super().__init__(); self.num_timesteps = num_timesteps; self.num_atom_types = num_atom_types
        self.time_embed = nn.Sequential(SinusoidalPositionalEmbedding(time_dim), nn.Linear(time_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.context_embed = nn.Linear(context_dim, hidden_dim)
        self.denoising_net = EGNNWrapper(in_dim=num_atom_types + hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, n_layers=n_layers)
        self.pos_noise_predictor = nn.Linear(hidden_dim, 3)
        self.x_noise_predictor = nn.Linear(hidden_dim, num_atom_types)
        betas = self.cosine_beta_schedule(num_timesteps); self.register_buffer('betas', betas); alphas = 1. - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0)); self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod)); self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1; x = torch.linspace(0, timesteps, steps, device=device); alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]; betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]); return torch.clip(betas, 0.0001, 0.9999)
    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1); sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        if x_start.dim() == 3: sqrt_alpha_t = sqrt_alpha_t.unsqueeze(-1); sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.unsqueeze(-1)
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
    def forward(self, data, context, atom_type_map):
        x_numeric, pos, edge_index = data.x, data.pos, data.edge_index
        atom_indices = torch.tensor([atom_type_map.get(num.item(), self.num_atom_types - 1) for num in x_numeric[:, 0]], device=device, dtype=torch.long)
        x_one_hot = F.one_hot(atom_indices, num_classes=self.num_atom_types).float()
        t = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        pos_noise, x_noise = torch.randn_like(pos), torch.randn_like(x_one_hot)
        noisy_pos, noisy_x = self.q_sample(pos, t, pos_noise), self.q_sample(x_one_hot, t, x_noise)
        t_emb = self.time_embed(t); c_emb = self.context_embed(context)
        combined_cond = (t_emb + c_emb).expand(noisy_x.size(0), -1)
        combined_feats = torch.cat([noisy_x, combined_cond], dim=-1)
        h_denoised = self.denoising_net(combined_feats, noisy_pos, edge_index)
        predicted_pos_noise, predicted_x_noise = self.pos_noise_predictor(h_denoised), self.x_noise_predictor(h_denoised)
        loss_pos = F.mse_loss(predicted_pos_noise, pos_noise); loss_x = F.mse_loss(predicted_x_noise, x_noise)
        return loss_pos + 1.0 * loss_x
    @torch.no_grad()
    def p_sample_loop(self, data_template, context, atom_type_map):
        self.eval(); num_nodes = data_template.num_nodes
        pos = torch.randn((num_nodes, 3), device=device); x = torch.randn((num_nodes, self.num_atom_types), device=device)
        alphas = 1. - self.betas
        for i in tqdm(reversed(range(self.num_timesteps)), desc="扩散生成", total=self.num_timesteps, leave=False):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            t_emb = self.time_embed(t); c_emb = self.context_embed(context)
            combined_cond = (t_emb + c_emb).expand(num_nodes, -1)
            combined_feats = torch.cat([x, combined_cond], dim=-1)
            h_denoised = self.denoising_net(combined_feats, pos, data_template.edge_index)
            predicted_pos_noise = self.pos_noise_predictor(h_denoised)
            predicted_x_noise = self.x_noise_predictor(h_denoised)
            alpha_t = alphas[i]
            pos = (1 / torch.sqrt(alpha_t)) * (pos - (self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i]) * predicted_pos_noise)
            x = (1 / torch.sqrt(alpha_t)) * (x - (self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i]) * predicted_x_noise)
            if i > 0:
                pos += torch.sqrt(self.betas[i]) * torch.randn_like(pos); x += torch.sqrt(self.betas[i]) * torch.randn_like(x)
        final_atom_indices = torch.argmax(x, dim=1)
        inv_atom_type_map = {v: k for k, v in atom_type_map.items()}
        final_atomic_nums = [inv_atom_type_map.get(idx.item(), 0) for idx in final_atom_indices]
        self.train(); return final_atomic_nums, pos.cpu().numpy()

# =============================================================================
# 3. 数据处理与特征工程模块
# =============================================================================
class ChEMBLDataset(Dataset):
    def __init__(self, data_dir): self.data_dir = data_dir; self.molecules = self._load_molecules(); self.graph_data_list = [d for d in [self._mol_to_graph_data(m) for m in tqdm(self.molecules, desc="转换分子为图")] if d]
    def _load_molecules(self):
        mols, files = [], [f for f in os.listdir(self.data_dir) if f.endswith(('.sdf', '.mol'))]
        for f in tqdm(files, desc="加载分子数据"):
            try:
                suppl = Chem.SDMolSupplier(os.path.join(self.data_dir, f))
                for mol in suppl:
                    if mol is not None:
                        if mol.GetNumConformers() == 0: AllChem.EmbedMolecule(mol)
                        mols.append(mol)
            except: logger.warning(f"加载 {f} 失败")
        logger.info(f"成功加载 {len(mols)} 个分子"); return mols
    @staticmethod
    def _mol_to_graph_data(mol):
        try:
            x = torch.tensor([[a.GetAtomicNum(), a.GetDegree(), a.GetTotalValence(), a.GetFormalCharge(), a.GetIsAromatic()*1, a.GetHybridization().real] for a in mol.GetAtoms()], dtype=torch.float)
            edge_index = to_undirected(torch.tensor([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=torch.long).t()) if mol.GetNumBonds() > 0 else torch.empty((2, 0), dtype=torch.long)
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
            return Data(x=x, edge_index=edge_index, pos=pos, mol=mol)
        except Exception: return None
    def __len__(self): return len(self.graph_data_list)
    def __getitem__(self, idx): return self.graph_data_list[idx]
class SelfSupervisedIsostereMiner(nn.Module):
    def __init__(self, fragment_dim, hidden_dim):
        super().__init__(); self.fragment_encoder = nn.Sequential(nn.Linear(fragment_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, hidden_dim)); self.isostere_kb = {}; self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fragment_dim)
    def forward(self, fragment_features): return self.fragment_encoder(fragment_features)
    @staticmethod
    def _split_molecule_with_brics(mol: Chem.Mol):
        if not mol: return [];
        try: return [m for m in (Chem.MolFromSmiles(s) for s in BRICS.BRICSDecompose(mol)) if m]
        except: return []
    def _get_fragment_features_as_fp(self, fragment: Chem.Mol):
        if not fragment: return None
        emol = Chem.EditableMol(fragment); [emol.RemoveAtom(idx) for idx in sorted([a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)]
        sane_frag = emol.GetMol()
        if sane_frag:
            try: Chem.SanitizeMol(sane_frag); return torch.from_numpy(np.array(self.fpgen.GetFingerprint(sane_frag), dtype=np.float32))
            except: return None
        return None
    def _get_morgan_fingerprint(self, mol: Chem.Mol):
        if not mol: return None
        emol = Chem.EditableMol(mol); [emol.RemoveAtom(idx) for idx in sorted([a.GetIdx() for a in mol.GetAtoms() if not a.GetAtomicNum()], reverse=True)]
        sane_frag = emol.GetMol()
        if sane_frag:
            try: Chem.SanitizeMol(sane_frag); return self.fpgen.GetFingerprint(sane_frag)
            except: return None
        return None
    def update_kb(self, fragments):
        logger.info("正在更新知识库..."); self.eval()
        with torch.no_grad():
            for frag in tqdm(fragments, "编码片段"):
                feat = self._get_fragment_features_as_fp(frag)
                if feat is not None: self.isostere_kb[Chem.MolToSmiles(frag)] = (self.forward(feat.unsqueeze(0).to(device)).squeeze(0).cpu(), frag)
        logger.info(f"知识库更新完毕，含 {len(self.isostere_kb)} 个片段"); self.train()
    def get_candidates_from_kb(self, frag_to_replace, top_k=5):
        if not self.isostere_kb: return []
        frag_fp = self._get_morgan_fingerprint(frag_to_replace)
        if frag_fp is None: return []
        candidates = []
        for smiles, (embedding, mol) in self.isostere_kb.items():
            cand_fp = self._get_morgan_fingerprint(mol)
            if cand_fp is None: continue
            sim = DataStructs.TanimotoSimilarity(frag_fp, cand_fp)
            if sim < 0.999: candidates.append((sim, mol))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:top_k]]
class MultiObjectiveEvaluator:
    def __init__(self):
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    def get_shape_similarity(self, mol1, mol2):
        try:
            if mol1.GetNumConformers() == 0: AllChem.EmbedMolecule(mol1)
            if mol2.GetNumConformers() == 0: AllChem.EmbedMolecule(mol2)
            usr_desc1 = rdMolDescriptors.GetUSR(mol1); usr_desc2 = rdMolDescriptors.GetUSR(mol2)
            dist = np.linalg.norm(np.array(usr_desc1) - np.array(usr_desc2)); return 1 / (1 + dist)
        except: return 0.0
    def evaluate(self, generated_mol, original_mol):
        try:
            if generated_mol is None: raise ValueError("生成的分子为None")
            if original_mol is None: raise ValueError("原始分子为None")
            Chem.SanitizeMol(generated_mol); Chem.SanitizeMol(original_mol)
        except Exception as e: return {'qed': 0.0, 'sa_score': 0.0, 'similarity': 0.0, 'shape_similarity': 0.0, 'composite_score': 0.0, 'valid': False}
        qed_score = QED.qed(generated_mol)
        try: sa_score_raw = calculate_sa_score(generated_mol); sa_score = np.clip((10.0 - sa_score_raw) / 9.0, 0, 1)
        except: sa_score = 0.0
        try: fp1 = self.fpgen.GetFingerprint(original_mol); fp2 = self.fpgen.GetFingerprint(generated_mol); similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        except: similarity = 0.0
        shape_similarity = self.get_shape_similarity(generated_mol, original_mol)
        composite_score = (0.3 * qed_score) + (0.3 * sa_score) + (0.2 * similarity) + (0.2 * shape_similarity)
        return {'qed': qed_score, 'sa_score': sa_score, 'similarity': similarity, 'shape_similarity': shape_similarity, 'composite_score': composite_score, 'valid': True}
# =============================================================================
# 4. 主模型：BioIsostericDiffusion
# =============================================================================
class BioIsostericDiffusion(nn.Module):
    def __init__(self, atom_dim, fragment_fp_dim, hidden_dim, lr, atom_types, pharm_factory):
        super().__init__()
        self.atom_types = atom_types; self.num_atom_types = len(atom_types); self.atom_type_map = {z: i for i, z in enumerate(atom_types)}
        self.pharm_factory = pharm_factory
        pharm_feat_dim = len(pharm_factory.GetFeatureFamilies())
        self.site_identifier = EdgePredictor(node_in_channels=atom_dim, hidden_channels=hidden_dim).to(device)
        self.isostere_miner = SelfSupervisedIsostereMiner(fragment_fp_dim, hidden_dim).to(device)
        self.context_encoder = EGNNWrapper(in_dim=atom_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, n_layers=4).to(device)
        self.pharmacophore_encoder = EGNNPharmacophoreEncoder(pharm_feature_dim=pharm_feat_dim, hidden_dim=hidden_dim, out_dim=hidden_dim).to(device)
        # 扩散模型的 context_dim 现在是 hidden_dim * 3
        self.generative_model = JointDiffusionGenerator(
            hidden_dim=hidden_dim, 
            context_dim=hidden_dim * 3, # geom + chem + stats
            num_atom_types=self.num_atom_types
        ).to(device)
        #self.generative_model = JointDiffusionGenerator(hidden_dim=hidden_dim, context_dim=hidden_dim * 2, num_atom_types=self.num_atom_types).to(device)
        self.evaluator = MultiObjectiveEvaluator()
        self.contrastive_criterion = NTXentLoss(); self.site_id_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(list(self.parameters()), lr=lr)
        self.site_id_loss_history = []; self.pretrain_loss_history = []; self.diffusion_loss_history = []
        # 【新增】统计向量编码器
        self.stats_encoder = nn.Linear(stats_dim, hidden_dim).to(device)

        # 扩散模型的 context_dim 现在是 hidden_dim * 3
        self.generative_model = JointDiffusionGenerator(
            hidden_dim=hidden_dim, 
            context_dim=hidden_dim * 3, # geom + chem + stats
            num_atom_types=self.num_atom_types
        ).to(device)
    # 【新增】构建统计数据库的辅助函数
    def _build_replacement_statistics(self, training_pairs):
        logger.info("构建替换规则的统计数据库...")
        stats_db = {}
        evaluator = MultiObjectiveEvaluator() # 临时评估器
        
        for mol_A, mol_B in tqdm(training_pairs, "统计性质变化"):
            # (这是一个简化的逻辑, 实际需要通过BRICS识别精确的 frag_A, frag_B)
            # 假设我们通过某种方式得到了 frag_A_smiles, frag_B_smiles
            key = (frag_A_smiles, frag_B_smiles)
            
            scores_A = evaluator.evaluate(mol_A, mol_A) # Self-evaluation to get properties
            scores_B = evaluator.evaluate(mol_B, mol_A)
            
            delta_qed = scores_B['qed'] - scores_A['qed']
            delta_sa = scores_B['sa_score'] - scores_A['sa_score']

            if key not in stats_db:
                stats_db[key] = {'count': 0, 'delta_qed': [], 'delta_sa': []}
            
            stats_db[key]['count'] += 1
            stats_db[key]['delta_qed'].append(delta_qed)
            stats_db[key]['delta_sa'].append(delta_sa)

        # 计算平均值
        avg_stats_db = {}
        for key, value in stats_db.items():
            avg_stats_db[key] = [
                np.mean(value['delta_qed']), 
                np.mean(value['delta_sa'])
            ]
        return avg_stats_db
    def get_3d_pharmacophore_points(self, mol):
        try:
            feats = self.pharm_factory.GetFeaturesForMol(mol)
            if not feats: return None, None
            feat_dim = len(self.pharm_factory.GetFeatureFamilies()); family_map = {fam: i for i, fam in enumerate(self.pharm_factory.GetFeatureFamilies())}
            pharm_feats_list, pharm_pos_list = [], []
            for feat in feats:
                feat_family = feat.GetFamily()
                if feat_family in family_map:
                    one_hot = np.zeros(feat_dim); one_hot[family_map[feat_family]] = 1
                    pharm_feats_list.append(one_hot); pharm_pos_list.append(list(feat.GetPos()))
            if not pharm_feats_list: return None, None
            return torch.tensor(pharm_feats_list, dtype=torch.float), torch.tensor(pharm_pos_list, dtype=torch.float)
        except Exception: return None, None
    def _prepare_fragment_for_diffusion(self, frag):
        if frag is None or not (1 < frag.GetNumAtoms() < 50): return None
        try:
            emol = Chem.EditableMol(frag); [emol.RemoveAtom(idx) for idx in sorted([a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)]
            mol_no_dummy = emol.GetMol()
            if mol_no_dummy is None or mol_no_dummy.GetNumAtoms() == 0: return None
            if not all(a.GetAtomicNum() in self.atom_types for a in mol_no_dummy.GetAtoms()): return None
            mol_with_hs = Chem.AddHs(mol_no_dummy)
            if AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv3()) == -1:
                if AllChem.EmbedMolecule(mol_with_hs, useRandomCoords=True) == -1: return None
            mol_final = Chem.RemoveHs(mol_with_hs)
            return ChEMBLDataset._mol_to_graph_data(mol_final)
        except Exception: return None
    def build_mol_from_atoms(self, atomic_nums, coords):
        mol = Chem.RWMol()
        for atomic_num in atomic_nums:
            if atomic_num > 0: mol.AddAtom(Chem.Atom(int(atomic_num)))
        if mol.GetNumAtoms() == 0: return None
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()): conf.SetAtomPosition(i, coords[i])
        mol.AddConformer(conf)
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                dist = np.linalg.norm(coords[i] - coords[j])
                try:
                    r_i = Chem.GetPeriodicTable().GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
                    r_j = Chem.GetPeriodicTable().GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
                    if 0.1 < dist < (r_i + r_j) * 0.7: mol.AddBond(i, j, Chem.BondType.SINGLE)
                except: continue
        try: Chem.SanitizeMol(mol); return mol.GetMol()
        except: return None
    def stitch_fragments(self, core_frag_with_dummy, new_frag):
        try:
            core_no_dummy_h = AllChem.ReplaceSubstructs(core_frag_with_dummy, Chem.MolFromSmarts('[#0]'), Chem.MolFromSmarts('[H]'), replaceAll=True)[0]
            core_no_dummy = Chem.RemoveHs(core_no_dummy_h)
            if new_frag is None: return None
            Chem.SanitizeMol(core_no_dummy); Chem.SanitizeMol(new_frag)
            recombined_mols_2d = list(BRICS.BRICSBuild([core_no_dummy, new_frag]))
            if not recombined_mols_2d: return None
            final_mol_2d = recombined_mols_2d[0]; final_mol_2d = Chem.AddHs(final_mol_2d)
            core_ref = Chem.Mol(core_frag_with_dummy)
            if core_ref.GetNumConformers() == 0: AllChem.EmbedMolecule(core_ref)
            Chem.SanitizeMol(core_ref)
            if AllChem.ConstrainedEmbed(final_mol_2d, core_ref, useTethers=True) == -1: return None
            AllChem.MMFFOptimizeMolecule(final_mol_2d)
            return Chem.RemoveHs(final_mol_2d)
        except Exception: return None
    @staticmethod
    def _create_edge_labels(data):
        mol = data.mol
        if (not mol) or (mol.GetNumBonds() == 0): return torch.zeros(data.edge_index.size(1), dtype=torch.float)
        try: brics_bonds_indices = {bond[0] for bond in BRICS.FindBRICSBonds(mol)}
        except: return torch.zeros(data.edge_index.size(1), dtype=torch.float)
        labels = []
        for i in range(data.edge_index.size(1) // 2):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            bond = mol.GetBondBetweenAtoms(u, v)
            is_brics_bond = bond is not None and bond.GetIdx() in brics_bonds_indices
            labels.append(1.0 if is_brics_bond else 0.0)
        labels_tensor = torch.tensor(labels, dtype=torch.float); return torch.cat([labels_tensor, labels_tensor], dim=0)
    def train_site_identifier(self, dataset, epochs=10, batch_size=32):
        logger.info("阶段一：开始训练位点识别器...")
        dataloader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True); self.site_identifier.train()
        for name, param in self.named_parameters(): param.requires_grad = 'site_identifier' in name
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, f"Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad(); batch = batch.to(device)
                edge_logits = self.site_identifier(batch.x, batch.edge_index)
                edge_labels = torch.cat([self._create_edge_labels(g.to('cpu')) for g in batch.to_data_list()], dim=0)
                loss = self.site_id_criterion(edge_logits, edge_labels.to(device))
                loss.backward(); self.optimizer.step(); total_loss += loss.item()
            avg_loss = total_loss / len(dataloader) if dataloader else 0
            logger.info(f"位点识别器 Epoch {epoch+1}, 平均损失: {avg_loss:.4f}"); self.site_id_loss_history.append(avg_loss)
        for param in self.parameters(): param.requires_grad = True
    def encode_context(self, data):
        graph_list = data.to_data_list(); encoded_graphs = []
        for graph in graph_list:
            graph = graph.to(device)
            h_updated = self.context_encoder(graph.x, graph.pos, graph.edge_index)
            batch_vec = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            encoded_graphs.append(global_mean_pool(h_updated, batch_vec))
        return torch.cat(encoded_graphs, dim=0)
    def _build_training_pairs(self, dataset, threshold=0.7, sample_size=1000):
        logger.info("为预训练构建正样本对..."); frag_bank, fp_bank, unique_smiles = [], [], set()
        for data in tqdm(dataset, "1/2: 构建片段银行"):
            if data and data.mol:
                for frag in self.isostere_miner._split_molecule_with_brics(data.mol):
                    smiles = Chem.MolToSmiles(frag)
                    if smiles not in unique_smiles:
                        fp = self.isostere_miner._get_morgan_fingerprint(frag)
                        if fp: unique_smiles.add(smiles); frag_bank.append(frag); fp_bank.append(fp)
        if len(frag_bank) < 2: return []
        positive_pairs = []
        for i in tqdm(range(len(frag_bank)), "2/2: 采样并构建正样本对"):
            indices = list(range(i)) + list(range(i + 1, len(frag_bank)))
            sample_indices = np.random.choice(indices, min(len(indices), sample_size), replace=False)
            sims = DataStructs.BulkTanimotoSimilarity(fp_bank[i], [fp_bank[j] for j in sample_indices])
            for k, sim in enumerate(sims):
                if sim >= threshold: positive_pairs.append((frag_bank[i], frag_bank[sample_indices[k]]))
        pair_smiles, unique_pairs = set(), []
        for f1, f2 in positive_pairs:
            key = tuple(sorted((Chem.MolToSmiles(f1), Chem.MolToSmiles(f2))))
            if key not in pair_smiles: pair_smiles.add(key); unique_pairs.append((f1, f2))
        logger.info(f"构建完成: 找到 {len(unique_pairs)} 个正样本对。"); return unique_pairs
    def pretrain_isostere_miner(self, dataset, epochs=10, batch_size=32):
        logger.info("阶段二：开始预训练 Isostere Miner...");
        for name, param in self.named_parameters(): param.requires_grad = 'isostere_miner' in name
        training_pairs = self._build_training_pairs(dataset)
        if not training_pairs: logger.error("未能构建训练对，跳过预训练。"); return
        dataloader = DataLoader(training_pairs, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, f"Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()
                features, labels, label_idx = [], [], 0
                for f1, f2 in batch:
                    feat1, feat2 = self.isostere_miner._get_fragment_features_as_fp(f1), self.isostere_miner._get_fragment_features_as_fp(f2)
                    if feat1 is not None and feat2 is not None: features.extend([feat1, feat2]); labels.extend([label_idx, label_idx]); label_idx += 1
                if len(features) < 2: continue
                embs = self.isostere_miner(torch.stack(features).to(device))
                loss = self.contrastive_criterion(embs, torch.tensor(labels, dtype=torch.long).to(device))
                loss.backward(); self.optimizer.step(); total_loss += loss.item()
            avg_loss = total_loss/len(dataloader) if dataloader and len(dataloader) > 0 else 0
            logger.info(f"预训练 Epoch {epoch+1}, 平均损失: {avg_loss:.4f}")
            self.pretrain_loss_history.append(avg_loss)
        for param in self.parameters(): param.requires_grad = True
        self._build_final_isostere_kb(dataset)
    def _build_final_isostere_kb(self, dataset):
        logger.info("构建最终知识库..."); all_frags, unique_smiles = [], set()
        for data in dataset:
            if data and data.mol:
                for frag in self.isostere_miner._split_molecule_with_brics(data.mol):
                    smiles = Chem.MolToSmiles(frag)
                    if smiles not in unique_smiles: unique_smiles.add(smiles); all_frags.append(frag)
        self.isostere_miner.update_kb(all_frags)
    def train_generative_model(self, dataset, epochs=50, batch_size=8):
        logger.info(f"阶段三：开始训练 (集成3D药效团条件)...");
        logger.info("为扩散模型准备'目标-条件'训练对...")
        training_pairs = self._build_training_pairs(dataset, threshold=0.7)
        processed_pairs = [pair for pair in tqdm(map(lambda p: (self._prepare_fragment_for_diffusion(p[0]), self._prepare_fragment_for_diffusion(p[1])), training_pairs), total=len(training_pairs), desc="净化和转换训练对") if pair[0] and pair[1]]
        if not processed_pairs: logger.error("未能创建训练对，跳过。"); return
        def collate_fn_diffusion(batch):
            batch_data_A = Batch.from_data_list([item[0] for item in batch]); batch_data_B = Batch.from_data_list([item[1] for item in batch]); return batch_data_A, batch_data_B
        pair_dataloader = DataLoader(processed_pairs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_diffusion, drop_last=True)
        self.generative_model.train(); self.pharmacophore_encoder.train(); self.context_encoder.train()
        for name, param in self.named_parameters(): param.requires_grad = 'generative_model' in name or 'context_encoder' in name or 'pharmacophore_encoder' in name
        for epoch in range(epochs):
            total_loss = 0
            for target_batch, condition_batch in tqdm(pair_dataloader, f"Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()
                target_batch, condition_batch = target_batch.to(device), condition_batch.to(device)
                with torch.no_grad():
                    self.context_encoder.eval(); geom_context_tensor = self.encode_context(condition_batch); self.context_encoder.train()
                chem_context_list = []
                for data in condition_batch.to('cpu').to_data_list():
                    pharm_feats, pharm_pos = self.get_3d_pharmacophore_points(data.mol)
                    if pharm_feats is not None and pharm_pos is not None and pharm_feats.size(0) > 0:
                        with torch.no_grad():
                            self.pharmacophore_encoder.eval()
                            chem_context = self.pharmacophore_encoder(pharm_feats.to(device), pharm_pos.to(device))
                            self.pharmacophore_encoder.train()
                        chem_context_list.append(chem_context)
                    else: chem_context_list.append(torch.zeros(self.pharmacophore_encoder.output_proj.out_features, device=device))
                chem_context_tensor = torch.stack(chem_context_list, dim=0)
                final_context = torch.cat([geom_context_tensor, chem_context_tensor], dim=1)
                loss_batch = 0; target_list = target_batch.to_data_list()
                for i in range(len(target_list)):
                    loss = self.generative_model(target_list[i], final_context[i].unsqueeze(0), self.atom_type_map)
                    if loss is not None and not torch.isnan(loss) and not torch.isinf(loss): loss_batch += loss
                if isinstance(loss_batch, int) and loss_batch == 0: continue
                (loss_batch / len(target_list)).backward(); self.optimizer.step(); total_loss += loss_batch.item()
            avg_loss = total_loss / len(processed_pairs) if processed_pairs else 0
            logger.info(f"扩散模型 Epoch {epoch+1}, 平均损失: {avg_loss:.4f}")
            self.diffusion_loss_history.append(avg_loss)
        for param in self.parameters(): param.requires_grad = True
    def generate_isosteres(self, mol: Chem.Mol, top_k_bonds: int = 1, top_k_fragments: int = 3):
        logger.info(f"\n" + "="*50); logger.info(f"开始为分子 {Chem.MolToSmiles(mol)} 进行3D生成与评估..."); self.eval()
        if mol.GetNumConformers() == 0: AllChem.EmbedMolecule(mol)
        data = ChEMBLDataset._mol_to_graph_data(mol)
        if data is None: logger.error("无法转换分子为图数据。"); return []
        data = data.to(device)
        with torch.no_grad():
            edge_logits = self.site_identifier(data.x, data.edge_index); edge_scores = torch.sigmoid(edge_logits)
        num_original_bonds = data.edge_index.size(1) // 2; bond_scores = edge_scores[:num_original_bonds].cpu().numpy()
        candidate_indices = np.argsort(bond_scores)[::-1][:top_k_bonds]
        logger.info(f"识别出 Top-{len(candidate_indices)} 替换位点 (键索引): {candidate_indices.tolist()}")
        all_results = []; original_bonds = list(mol.GetBonds())
        for bond_idx in candidate_indices:
            bond_to_break = original_bonds[bond_idx]
            try:
                frags_with_dummy = BRICS.BreakBRICSBonds(mol, bonds=[(bond_to_break.GetBeginAtomIdx(), bond_to_break.GetEndAtomIdx())])
                frags_mols_with_dummy = list(GetMolFrags(frags_with_dummy, asMols=True, sanitizeFrags=True))
                if len(frags_mols_with_dummy) != 2: continue
            except: continue
            frag_to_replace_wd, core_frag_wd = (frags_mols_with_dummy[0], frags_mols_with_dummy[1]) if frags_mols_with_dummy[0].GetNumAtoms() < frags_mols_with_dummy[1].GetNumAtoms() else (frags_mols_with_dummy[1], frags_mols_with_dummy[0])
            isostere_candidates = self.isostere_miner.get_candidates_from_kb(frag_to_replace_wd, top_k=top_k_fragments)
            if not isostere_candidates: continue
            core_data = self._prepare_fragment_for_diffusion(core_frag_wd)
            if core_data is None: continue
            with torch.no_grad(): geom_context = self.encode_context(core_data.to(device))
            for isostere_cand in isostere_candidates:
                with torch.no_grad():
                    isostere_data = self._prepare_fragment_for_diffusion(isostere_cand)
                    if isostere_data is None: continue
                    pharm_feats, pharm_pos = self.get_3d_pharmacophore_points(isostere_data.mol)
                    if pharm_feats is None or pharm_pos is None: continue
                    chem_context = self.pharmacophore_encoder(pharm_feats.to(device), pharm_pos.to(device)).unsqueeze(0)
                final_context = torch.cat([geom_context, chem_context], dim=1)
                template_data = self._prepare_fragment_for_diffusion(frag_to_replace_wd)
                if template_data is None: continue
                generated_atomic_nums, generated_pos = self.generative_model.p_sample_loop(template_data.to(device), final_context, self.atom_type_map)
                new_frag = self.build_mol_from_atoms(generated_atomic_nums, generated_pos)
                if not new_frag: logger.warning("从原子云构建分子失败。"); continue
                try:
                    new_mol = self.stitch_fragments(core_frag_wd, new_frag)
                    if not new_mol: continue
                except Exception as e: logger.warning(f"3D拼接或优化失败: {e}"); continue
                scores = self.evaluator.evaluate(new_mol, mol)
                all_results.append({'new_mol':new_mol, 'new_mol_smiles': Chem.MolToSmiles(new_mol), 'original_fragment_smiles': Chem.MolToSmiles(Chem.RemoveHs(frag_to_replace_wd)), 'candidate_isostere_smiles': Chem.MolToSmiles(isostere_cand), **scores})
        return sorted(all_results, key=lambda x: x.get('composite_score', 0), reverse=True)
# =============================================================================
# 5. 可视化与主执行函数
# =============================================================================
def plot_loss_curves(pretrain_losses, diffusion_losses, site_id_losses):
    try: plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
    except: logger.warning("中文字体 'SimHei' 未找到，图例可能显示不正确。")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6)); fig.suptitle('模型三阶段训练损失曲线', fontsize=16)
    axes[0].plot(site_id_losses, label='位点识别器 (BCE)', color='green'); axes[0].set_title('阶段一: 位点识别器训练'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('平均损失'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(pretrain_losses, label='等排体挖掘器 (NTXent)', color='blue'); axes[1].set_title('阶段二: 等排体挖掘器预训练'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('平均损失'); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(diffusion_losses, label='3D扩散模型', color='red'); axes[2].set_title('阶段三: 3D联合生成模型训练'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('平均损失'); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig("training_loss_curves.png"); logger.info("训练损失曲线图已保存为 training_loss_curves.png"); plt.show()
def main():
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'); pharm_factory = AllChem.BuildFeatureFactory(fdef_name)
    data_dir = "./chembl_data_sample"
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.listdir(data_dir):
        logger.info("数据目录为空，下载示例分子...");
        try: r = requests.get("https://files.rcsb.org/ligands/view/ASN_ideal.sdf"); r.raise_for_status(); open(os.path.join(data_dir, "aspirin.sdf"), 'wb').write(r.content); logger.info("已下载 'aspirin.sdf'。")
        except Exception as e: logger.error(f"下载失败: {e}"); return
    
    full_dataset = ChEMBLDataset(data_dir)
    if not full_dataset or len(full_dataset) == 0: logger.error("数据集为空或加载失败。"); return
    
    indices = list(range(len(full_dataset))); np.random.shuffle(indices)
    split_point = int(0.8 * len(full_dataset))
    train_indices, test_indices = indices[:split_point], indices[split_point:]
    train_dataset = [full_dataset[i] for i in train_indices]
    test_dataset = [full_dataset[i] for i in test_indices]
    
    logger.info(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")
    
    atom_types_list = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    
    model = BioIsostericDiffusion(
        atom_dim=6, fragment_fp_dim=2048, hidden_dim=128, lr=1e-4, atom_types=atom_types_list, pharm_factory=pharm_factory
    )
    
    model.train_site_identifier(train_dataset, epochs=2, batch_size=32)
    model.pretrain_isostere_miner(train_dataset, epochs=2, batch_size=32)
    model.train_generative_model(train_dataset, epochs=5, batch_size=8)

    logger.info("所有训练阶段完成。")
    with open("isostere_kb_final.pkl", "wb") as f: pickle.dump(model.isostere_miner.isostere_kb, f)
    logger.info("等排体知识库已保存。")
    
    if test_dataset:
        test_mol_data = test_dataset[np.random.randint(len(test_dataset))]
        results = model.generate_isosteres(test_mol_data.mol)
        if results:
            results_df = pd.DataFrame(results)
            display_columns = [col for col in results_df.columns if col != 'new_mol']
            display_df = results_df[display_columns]
            logger.info("生成与评估结果 (Top 5):")
            try:
                print(tabulate(display_df.head(), headers='keys', tablefmt='psql'))
            except ImportError:
                print(display_df.head())
        else:
            logger.warning("未能为示例分子生成任何有效的等排体。")

    plot_loss_curves(model.pretrain_loss_history, model.diffusion_loss_history, model.site_id_loss_history)
if __name__ == "__main__":
    main()