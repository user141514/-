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
from torch_geometric.utils import to_undirected, to_dense_adj, to_dense_batch

from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors, Draw, GetMolFrags
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.Chem import QED
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

try:
    from rdkit.Contrib.SA_Score import sascorer


    def calculate_sa_score(mol):
        return sascorer.calculateScore(mol)
except ImportError:
    sascorer = None


    def calculate_sa_score(mol):
        return 5.0

from rdkit import Chem, DataStructs, RDLogger, rdBase, RDConfig
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import logging
import pickle
import requests
import matplotlib.pyplot as plt
import math
from tabulate import tabulate

# 引入 AccFG 库
try:
    from accfg import AccFG
except ImportError:
    AccFG = None
    print("CRITICAL ERROR: AccFG library not found. Please install it using 'pip install accfg'.")

# 引入 EGNN 实现
try:
    from egnn_pytorch import EGNN
except ImportError:
    EGNN = None
    print("CRITICAL ERROR: egnn_pytorch library not found. Please install it using 'pip install egnn_pytorch'.")

# 导入简化的 ED 上下文构造器
from ed_context import build_ed_context_vector


# =============================================================================
# 1. 基础设置 (移入 main 函数)
# =============================================================================

# =============================================================================
# 2. 核心辅助函数
# =============================================================================
def get_functional_groups_with_accfg(smiles_string, analyzer):
    if not analyzer or not smiles_string: return []
    try:
        fgs, _ = analyzer.run(smiles_string, show_atoms=True, show_graph=True, canonical=False)
        return [(str(name), tuple(map(int, idx))) for name, matches in fgs.items() for idx in matches] if fgs else []
    except Exception as e:
        logging.error(f"AccFG 在处理 {smiles_string} 时出错: {e}")
        return []


def fragment_by_subgraph_construction(mol, functional_groups_info):
    if not functional_groups_info or mol.GetNumConformers() == 0:
        return []

    original_conf = mol.GetConformer()
    fragment_pairs = []
    all_atom_indices = set(range(mol.GetNumAtoms()))

    for group_name, fg_indices in functional_groups_info:
        try:
            fg_indices_set = set(int(i) for i in fg_indices)
            if not fg_indices_set: continue
            core_indices_set = all_atom_indices - fg_indices_set
            if not core_indices_set: continue

            attachment_points = []
            for bond in mol.GetBonds():
                begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if (begin_idx in fg_indices_set and end_idx in core_indices_set):
                    attachment_points.append({'fg_atom': begin_idx, 'core_atom': end_idx})
                elif (end_idx in fg_indices_set and begin_idx in core_indices_set):
                    attachment_points.append({'fg_atom': end_idx, 'core_atom': begin_idx})

            if not attachment_points: continue

            fg_rwmol = Chem.RWMol()
            old_to_new_fg_map = {old_idx: fg_rwmol.AddAtom(mol.GetAtomWithIdx(old_idx)) for old_idx in fg_indices_set}
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in fg_indices_set and bond.GetEndAtomIdx() in fg_indices_set:
                    fg_rwmol.AddBond(old_to_new_fg_map[bond.GetBeginAtomIdx()], old_to_new_fg_map[bond.GetEndAtomIdx()],
                                     bond.GetBondType())

            core_rwmol = Chem.RWMol()
            old_to_new_core_map = {old_idx: core_rwmol.AddAtom(mol.GetAtomWithIdx(old_idx)) for old_idx in
                                   core_indices_set}
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in core_indices_set and bond.GetEndAtomIdx() in core_indices_set:
                    core_rwmol.AddBond(old_to_new_core_map[bond.GetBeginAtomIdx()],
                                       old_to_new_core_map[bond.GetEndAtomIdx()], bond.GetBondType())

            fg_conf = Chem.Conformer(len(fg_indices_set) + len(attachment_points))
            core_conf = Chem.Conformer(len(core_indices_set) + len(attachment_points))

            for old_idx, new_idx in old_to_new_fg_map.items():
                fg_conf.SetAtomPosition(new_idx, original_conf.GetAtomPosition(old_idx))
            for old_idx, new_idx in old_to_new_core_map.items():
                core_conf.SetAtomPosition(new_idx, original_conf.GetAtomPosition(old_idx))

            for i, point in enumerate(attachment_points):
                dummy_map_num = i + 1

                dummy_fg = Chem.Atom(0);
                dummy_fg.SetAtomMapNum(dummy_map_num)
                dummy_idx_fg = fg_rwmol.AddAtom(dummy_fg)
                fg_rwmol.AddBond(old_to_new_fg_map[point['fg_atom']], dummy_idx_fg, Chem.BondType.SINGLE)
                fg_conf.SetAtomPosition(dummy_idx_fg, original_conf.GetAtomPosition(point['core_atom']))

                dummy_core = Chem.Atom(0);
                dummy_core.SetAtomMapNum(dummy_map_num)
                dummy_idx_core = core_rwmol.AddAtom(dummy_core)
                core_rwmol.AddBond(old_to_new_core_map[point['core_atom']], dummy_idx_core, Chem.BondType.SINGLE)
                core_conf.SetAtomPosition(dummy_idx_core, original_conf.GetAtomPosition(point['fg_atom']))

            fg_mol = fg_rwmol.GetMol()
            core_mol = core_rwmol.GetMol()

            if fg_mol: fg_mol.AddConformer(fg_conf, assignId=True)
            if core_mol: core_mol.AddConformer(core_conf, assignId=True)

            if Chem.SanitizeMol(fg_mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE or \
                    Chem.SanitizeMol(core_mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
                continue

            fragment_pairs.append((core_mol, fg_mol))
        except Exception:
            continue
    return fragment_pairs


def simple_collate_fn(batch):
    return batch


def collate_fn_diffusion(batch):
    batch_targets = Batch.from_data_list([item[0] for item in batch])
    batch_conditions = Batch.from_data_list([item[1] for item in batch])
    return batch_targets, batch_conditions


# =============================================================================
# 3. 核心模型与损失函数定义
# =============================================================================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07, device='cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(self.device) - torch.eye(features.size(0),
                                                                                                      device=self.device)
        exp_sim = torch.exp(sim_matrix)
        sum_exp = torch.sum(exp_sim * (1 - torch.eye(exp_sim.size(0), device=self.device)), dim=1, keepdim=True)
        log_prob = sim_matrix - torch.log(sum_exp + 1e-9)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        return -mean_log_prob.mean()


class EdgePredictor(nn.Module):
    def __init__(self, node_in_channels, hidden_channels):
        super().__init__()
        self.node_conv1 = GCNConv(node_in_channels, hidden_channels)
        self.node_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden_channels, 1))

    def forward(self, x, edge_index):
        h = F.relu(self.node_conv1(x, edge_index))
        h = self.node_conv2(h, edge_index)
        start_nodes = h[edge_index[0]]
        end_nodes = h[edge_index[1]]
        edge_representation = torch.cat([start_nodes, end_nodes], dim=-1)
        return self.mlp(edge_representation).squeeze(-1)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, device='cpu'):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EGNNWrapper(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=4):
        super().__init__()
        self.embedding_in = nn.Linear(in_dim, hidden_dim)
        self.egnn_layers = nn.ModuleList([EGNN(dim=hidden_dim) for _ in range(n_layers)])
        self.embedding_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, pos, edge_index, batch_vec=None):
        if batch_vec is None:
            return self.forward_single(x, pos, edge_index)
        else:
            return self.forward_batched(x, pos, edge_index, batch_vec)

    def forward_single(self, x, pos, edge_index):
        h = self.embedding_in(x)
        adj_mat = to_dense_adj(edge_index, max_num_nodes=x.size(0)).bool()
        h = h.unsqueeze(0)
        pos = pos.unsqueeze(0)
        for egnn in self.egnn_layers:
            h, pos = egnn(feats=h, coors=pos, adj_mat=adj_mat)
        h = h.squeeze(0)
        return self.embedding_out(h)

    def forward_batched(self, x, pos, edge_index, batch_vec):
        h0 = self.embedding_in(x)
        h_dense, mask = to_dense_batch(h0, batch_vec)
        p_dense, _ = to_dense_batch(pos, batch_vec)
        adj = to_dense_adj(edge_index, batch=batch_vec).bool()

        h = h_dense
        p = p_dense
        for egnn in self.egnn_layers:
            h, p = egnn(feats=h, coors=p, adj_mat=adj)

        out_dense = self.embedding_out(h)
        out = out_dense[mask]
        return out


class EGNNPharmacophoreEncoder(nn.Module):
    def __init__(self, pharm_feature_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__()
        self.egnn_net = EGNNWrapper(in_dim=pharm_feature_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                                    n_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, pharm_feats, pharm_pos):
        num_points = pharm_feats.size(0)
        if num_points == 0:
            return torch.zeros(self.output_proj.out_features, device=pharm_feats.device)
        if num_points < 2:
            point_embeddings = self.egnn_net.embedding_in(pharm_feats)
        else:
            src, dst = [], []
            for i in range(num_points):
                for j in range(num_points):
                    if i != j: src.append(i); dst.append(j)
            edge_index = torch.tensor([src, dst], device=pharm_feats.device, dtype=torch.long)
            point_embeddings = self.egnn_net(pharm_feats, pharm_pos, edge_index)
        global_embedding = point_embeddings.mean(dim=0)
        return self.output_proj(global_embedding)


class EDContextEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, v):
        return self.net(v)


class JointDiffusionGenerator(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_atom_types, device, time_dim=64, n_layers=4, num_timesteps=100):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_atom_types = num_atom_types
        self.device = device
        self.time_embed = nn.Sequential(SinusoidalPositionalEmbedding(time_dim, device=device),
                                        nn.Linear(time_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.context_embed = nn.Linear(context_dim, hidden_dim)
        self.denoising_net = EGNNWrapper(in_dim=num_atom_types + hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                                         n_layers=n_layers)
        self.pos_noise_predictor = nn.Linear(hidden_dim, 3)
        self.x_noise_predictor = nn.Linear(hidden_dim, num_atom_types)
        betas = self.cosine_beta_schedule(num_timesteps)
        self.register_buffer('betas', betas)
        alphas = 1. - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t_index, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t_index].unsqueeze(-1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_index].unsqueeze(-1)
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

    def forward(self, batch, context, atom_type_map):
        x_numeric, pos, edge_index, bvec = batch.x, batch.pos, batch.edge_index, batch.batch

        atom_indices = torch.tensor([atom_type_map.get(int(num.item()), self.num_atom_types - 1)
                                     for num in x_numeric[:, 0]], device=self.device, dtype=torch.long)
        x_one_hot = F.one_hot(atom_indices, num_classes=self.num_atom_types).float()

        B = int(bvec.max().item()) + 1
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        t_per_node = t[bvec]

        pos_noise = torch.randn_like(pos)
        x_noise = torch.randn_like(x_one_hot)
        noisy_pos = self.q_sample(pos, t_per_node, pos_noise)
        noisy_x = self.q_sample(x_one_hot, t_per_node, x_noise)

        t_emb = self.time_embed(t)
        c_emb = self.context_embed(context)
        cond_graph = (t_emb + c_emb)
        cond_node = cond_graph[bvec]

        combined_feats = torch.cat([noisy_x, cond_node], dim=-1)

        h_denoised = self.denoising_net(combined_feats, noisy_pos, edge_index, bvec)
        predicted_pos_noise = self.pos_noise_predictor(h_denoised)
        predicted_x_noise = self.x_noise_predictor(h_denoised)

        loss_pos = F.mse_loss(predicted_pos_noise, pos_noise)
        loss_x = F.mse_loss(predicted_x_noise, x_noise)
        return loss_pos + loss_x

    @torch.no_grad()
    def p_sample_loop(self, data_template, context, atom_type_map):
        self.eval()
        num_nodes = data_template.num_nodes
        pos = torch.randn((num_nodes, 3), device=self.device)
        x = torch.randn((num_nodes, self.num_atom_types), device=self.device)
        alphas = 1. - self.betas
        for i in tqdm(reversed(range(self.num_timesteps)), desc="扩散生成", total=self.num_timesteps, leave=False):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            t_emb = self.time_embed(t)
            c_emb = self.context_embed(context)
            combined_cond = (t_emb + c_emb).expand(num_nodes, -1)
            combined_feats = torch.cat([x, combined_cond], dim=-1)
            h_denoised = self.denoising_net(combined_feats, pos, data_template.edge_index)
            predicted_pos_noise = self.pos_noise_predictor(h_denoised)
            predicted_x_noise = self.x_noise_predictor(h_denoised)
            alpha_t = alphas[i]
            pos = (1 / torch.sqrt(alpha_t)) * (
                        pos - (self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i]) * predicted_pos_noise)
            x = (1 / torch.sqrt(alpha_t)) * (
                        x - (self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i]) * predicted_x_noise)
            if i > 0:
                pos += torch.sqrt(self.betas[i]) * torch.randn_like(pos)
                x += torch.sqrt(self.betas[i]) * torch.randn_like(x)
        final_atom_indices = torch.argmax(x, dim=1)
        inv_atom_type_map = {v: k for k, v in atom_type_map.items()}
        final_atomic_nums = [inv_atom_type_map.get(idx.item(), 0) for idx in final_atom_indices]
        self.train()
        return final_atomic_nums, pos.cpu().numpy()


# =============================================================================
# 4. 数据处理与特征工程模块
# =============================================================================
class ChEMBLDataset(Dataset):
    def __init__(self, data_dir, max_mols=None):
        self.data_dir = data_dir
        self.graph_data_list = []
        files = [f for f in os.listdir(self.data_dir) if f.endswith(('.sdf', '.mol'))]

        desc = "加载并转换分子为图"
        pbar = tqdm(total=max_mols, desc=desc) if max_mols is not None else tqdm(desc=desc)

        molecules_processed = 0
        for f in files:
            if max_mols is not None and molecules_processed >= max_mols:
                break

            try:
                suppl = Chem.SDMolSupplier(os.path.join(self.data_dir, f), removeHs=False)
                for mol in suppl:
                    if max_mols is not None and molecules_processed >= max_mols:
                        break

                    if mol is None:
                        continue

                    graph_data = self._mol_to_graph_data(mol)
                    if graph_data:
                        self.graph_data_list.append(graph_data)
                        molecules_processed += 1
                        pbar.update(1)

            except Exception as e:
                logging.warning(f"处理文件 {f} 时出错: {e}")

        pbar.close()
        if max_mols is not None and pbar.n < max_mols:
            pbar.n = max_mols
            pbar.refresh()

        logging.info(f"成功加载并转换 {len(self.graph_data_list)} 个分子。")

    @staticmethod
    def _mol_to_graph_data(mol):
        try:
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

            Chem.SanitizeMol(mol)

            x = torch.tensor([[a.GetAtomicNum(), a.GetDegree(), a.GetTotalValence(), a.GetFormalCharge(),
                               a.GetIsAromatic() * 1, a.GetHybridization().real] for a in mol.GetAtoms()],
                             dtype=torch.float)
            edge_index = to_undirected(torch.tensor([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
                                                    dtype=torch.long).t()) if mol.GetNumBonds() > 0 else torch.empty(
                (2, 0), dtype=torch.long)
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

            return Data(x=x, edge_index=edge_index, pos=pos, mol=mol)
        except Exception as e:
            smiles = Chem.MolToSmiles(Chem.MolFromSmarts(Chem.MolToSmarts(mol))) if mol else "N/A"
            logging.warning(f"转换分子 {smiles} 为图失败: {e}")
            return None

    def __len__(self):
        return len(self.graph_data_list)

    def __getitem__(self, idx):
        return self.graph_data_list[idx]


class SelfSupervisedIsostereMiner(nn.Module):
    def __init__(self, fragment_dim, hidden_dim, device='cpu'):
        super().__init__()
        self.fragment_encoder = nn.Sequential(nn.Linear(fragment_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                              nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, hidden_dim))
        self.isostere_kb = {}
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fragment_dim)
        self.device = device

    def _get_fragment_features_as_fp(self, fragment: Chem.Mol):
        if not fragment: return None
        try:
            fp = self.fpgen.GetFingerprint(fragment)
            return torch.from_numpy(np.array(fp, dtype=np.float32))
        except Exception as e:
            logging.warning(f"计算指纹FP失败 for {Chem.MolToSmiles(fragment)}: {e}")
            return None

    def _get_morgan_fingerprint(self, mol: Chem.Mol):
        if not mol: return None
        try:
            return self.fpgen.GetFingerprint(mol)
        except Exception as e:
            logging.warning(f"计算指纹失败 for {Chem.MolToSmiles(mol)}: {e}")
            return None

    def update_kb(self, fragments):
        logging.info("正在更新知识库...")
        unique_fragments = {Chem.MolToSmiles(f): f for f in fragments}.values()
        logging.info(f"从 {len(fragments)} 个输入中得到 {len(unique_fragments)} 个唯一官能团。")
        for frag in tqdm(unique_fragments, "为知识库缓存官能团"):
            self.isostere_kb[Chem.MolToSmiles(frag)] = frag
        logging.info(f"知识库更新完毕，含 {len(self.isostere_kb)} 个片段")

    def get_candidates_from_kb(self, frag_to_replace, top_k=5):
        if not self.isostere_kb: return []
        frag_fp = self._get_morgan_fingerprint(frag_to_replace)
        if frag_fp is None: return []

        candidates = []
        for smiles, mol in self.isostere_kb.items():
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
            usr_desc1 = rdMolDescriptors.GetUSR(mol1)
            usr_desc2 = rdMolDescriptors.GetUSR(mol2)
            dist = np.linalg.norm(np.array(usr_desc1) - np.array(usr_desc2))
            return 1 / (1 + dist)
        except:
            return 0.0

    def evaluate(self, generated_mol, original_mol):
        try:
            if generated_mol is None: raise ValueError("生成的分子为None")
            if original_mol is None: raise ValueError("原始分子为None")
            Chem.SanitizeMol(generated_mol)
            Chem.SanitizeMol(original_mol)
        except Exception as e:
            return {'qed': 0.0, 'sa_score': 0.0, 'similarity': 0.0, 'shape_similarity': 0.0, 'composite_score': 0.0,
                    'valid': False}
        qed_score = QED.qed(generated_mol)
        try:
            sa_score_raw = calculate_sa_score(generated_mol)
            sa_score = np.clip((10.0 - sa_score_raw) / 9.0, 0, 1)
        except:
            sa_score = 0.0
        try:
            fp1 = self.fpgen.GetFingerprint(original_mol)
            fp2 = self.fpgen.GetFingerprint(generated_mol)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            similarity = 0.0
        shape_similarity = self.get_shape_similarity(generated_mol, original_mol)
        composite_score = (0.3 * qed_score) + (0.3 * sa_score) + (0.2 * similarity) + (0.2 * shape_similarity)
        return {'qed': qed_score, 'sa_score': sa_score, 'similarity': similarity, 'shape_similarity': shape_similarity,
                'composite_score': composite_score, 'valid': True}


# =============================================================================
# 5. 主模型：BioIsostericDiffusion
# =============================================================================
class BioIsostericDiffusion(nn.Module):
    def __init__(self, atom_dim, fragment_fp_dim, hidden_dim, lr, atom_types, pharm_factory, device, accfg_analyzer):
        super().__init__()
        self.device = device
        self.accfg_analyzer = accfg_analyzer
        self.atom_types = atom_types
        self.num_atom_types = len(atom_types)
        self.atom_type_map = {z: i for i, z in enumerate(atom_types)}
        self.pharm_factory = pharm_factory
        pharm_feat_dim = len(pharm_factory.GetFeatureFamilies())

        self.site_identifier = EdgePredictor(node_in_channels=atom_dim, hidden_channels=hidden_dim).to(device)
        self.isostere_miner = SelfSupervisedIsostereMiner(fragment_fp_dim, hidden_dim, device=device).to(device)
        self.context_encoder = EGNNWrapper(in_dim=atom_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, n_layers=4).to(
            device)
        self.pharmacophore_encoder = EGNNPharmacophoreEncoder(pharm_feature_dim=pharm_feat_dim, hidden_dim=hidden_dim,
                                                              out_dim=hidden_dim).to(device)

        self.ed_vec_cfg = dict(radii=(1.5, 2.5, 3.5), n_dir=64, stats=("mean", "std", "posfrac", "p10", "p90"))
        self.ed_vec_dim = len(self.ed_vec_cfg["radii"]) * len(self.ed_vec_cfg["stats"])
        self.ed_context_encoder = EDContextEncoder(in_dim=self.ed_vec_dim, hidden_dim=hidden_dim,
                                                   out_dim=hidden_dim).to(device)

        self.generative_model = JointDiffusionGenerator(
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            num_atom_types=self.num_atom_types,
            device=self.device
        ).to(device)

        self.ln_geom = nn.LayerNorm(hidden_dim).to(device)
        self.ln_pharm = nn.LayerNorm(hidden_dim).to(device)
        self.ln_ed = nn.LayerNorm(hidden_dim).to(device)

        # 固定 alpha 为 0.1
        self._ed_alpha_raw = nn.Parameter(torch.tensor(-2.1972246), requires_grad=False)

        self.evaluator = MultiObjectiveEvaluator()
        self.contrastive_criterion = NTXentLoss(device=self.device)
        self.site_id_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.site_id_loss_history = []
        self.pretrain_loss_history = []
        self.diffusion_loss_history = []

    def ed_alpha(self):
        return torch.sigmoid(self._ed_alpha_raw)

    def get_3d_pharmacophore_points(self, mol):
        try:
            feats = self.pharm_factory.GetFeaturesForMol(mol)
            if not feats: return None, None
            feat_dim = len(self.pharm_factory.GetFeatureFamilies())
            family_map = {fam: i for i, fam in enumerate(self.pharm_factory.GetFeatureFamilies())}
            pharm_feats_list, pharm_pos_list = [], []
            for feat in feats:
                feat_family = feat.GetFamily()
                if feat_family in family_map:
                    one_hot = np.zeros(feat_dim, dtype=np.float32)
                    one_hot[family_map[feat_family]] = 1.0
                    pharm_feats_list.append(one_hot)
                    pharm_pos_list.append(list(feat.GetPos()))
            if not pharm_feats_list: return None, None

            pharm_feats_tensor = torch.from_numpy(np.array(pharm_feats_list, dtype=np.float32))
            pharm_pos_tensor = torch.from_numpy(np.array(pharm_pos_list, dtype=np.float32))
            return pharm_feats_tensor, pharm_pos_tensor
        except Exception:
            return None, None

    def _prepare_fragment_for_diffusion(self, frag):
        if frag is None or not (0 < frag.GetNumAtoms() < 50): return None
        try:
            mol_no_dummy_list = AllChem.ReplaceSubstructs(frag, Chem.MolFromSmarts('[#0]'), Chem.MolFromSmarts('[H]'),
                                                          replaceAll=True)
            if not mol_no_dummy_list: return None
            mol_no_dummy = mol_no_dummy_list[0]

            if mol_no_dummy is None: return None
            mol_no_dummy = Chem.RemoveHs(mol_no_dummy)
            if mol_no_dummy.GetNumAtoms() == 0: return None

            if not all(a.GetAtomicNum() in self.atom_types for a in mol_no_dummy.GetAtoms()): return None
            mol_with_hs = Chem.AddHs(mol_no_dummy)
            if AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv3()) == -1:
                if AllChem.EmbedMolecule(mol_with_hs, useRandomCoords=True) == -1: return None
            mol_final = Chem.RemoveHs(mol_with_hs)
            return ChEMBLDataset._mol_to_graph_data(mol_final)
        except Exception as e:
            logging.warning(f"预处理片段 {Chem.MolToSmiles(frag)} 失败: {e}")
            return None

    def build_mol_from_atoms(self, atomic_nums, coords):
        mol = Chem.RWMol()
        for atomic_num in atomic_nums:
            if atomic_num > 0: mol.AddAtom(Chem.Atom(int(atomic_num)))
        if mol.GetNumAtoms() == 0: return None

        coords = np.array(coords, dtype=np.float64)

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, list(coords[i]))

        mol.AddConformer(conf)
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                dist = np.linalg.norm(coords[i] - coords[j])
                try:
                    r_i = Chem.GetPeriodicTable().GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
                    r_j = Chem.GetPeriodicTable().GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
                    if 0.1 < dist < (r_i + r_j) * 0.7: mol.AddBond(i, j, Chem.BondType.SINGLE)
                except:
                    continue
        try:
            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except:
            return None

    def stitch_fragments(self, core_frag_with_dummy, new_frag_with_dummy):
        """
            将核心骨架与新生成的片段拼接，并进行稳健的 3D 构象生成。
            集成 Plan A (坐标映射) 和 Plan B (刚性对齐) 策略。
        """
        # 1. 准备分子 (从 RDKit 对象复制，避免修改原对象)
        core_mol = Chem.Mol(core_frag_with_dummy)
        frag_mol = Chem.Mol(new_frag_with_dummy)

        # =================================================
        # 步骤 A: 准备完美的 Core 3D 模板
        # =================================================
        # 替换 * 为 H，解决 UFF 力场报错
        try:
            core_clean = \
            Chem.ReplaceSubstructs(core_mol, Chem.MolFromSmarts('[#0]'), Chem.MolFromSmarts('[H]'), replaceAll=True)[0]
            core_clean = Chem.RemoveHs(core_clean)
            core_clean = Chem.AddHs(core_clean)
            # 生成 Core 的 3D 坐标
            if AllChem.EmbedMolecule(core_clean, AllChem.ETKDGv3()) == -1:
                return None
        except Exception:
            return None

        # =================================================
        # 步骤 B: 图编辑拼接 (Graph Editing)
        # =================================================
        try:
            # 寻找连接点
            dummy_core_idx, anchor_core_idx = -1, -1
            for atom in core_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    dummy_core_idx = atom.GetIdx()
                    neighbors = atom.GetNeighbors()
                    if neighbors: anchor_core_idx = neighbors[0].GetIdx()
                    break

            dummy_frag_idx, anchor_frag_idx = -1, -1
            for atom in frag_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    dummy_frag_idx = atom.GetIdx()
                    neighbors = atom.GetNeighbors()
                    if neighbors: anchor_frag_idx = neighbors[0].GetIdx()
                    break

            if dummy_core_idx == -1 or dummy_frag_idx == -1:
                return None

            # 合并与连线
            combined = Chem.CombineMols(core_mol, frag_mol)
            rw_mol = Chem.RWMol(combined)
            offset = core_mol.GetNumAtoms()
            rw_mol.AddBond(anchor_core_idx, anchor_frag_idx + offset, Chem.BondType.SINGLE)

            # 删除虚原子 (从大到小删)
            atoms_to_remove = sorted([dummy_core_idx, dummy_frag_idx + offset], reverse=True)
            for idx in atoms_to_remove:
                rw_mol.RemoveAtom(idx)

            stitched_mol = rw_mol.GetMol()
            Chem.SanitizeMol(stitched_mol)
        except Exception:
            return None

        # =================================================
        # 步骤 C: 智能 3D 生成 (Plan A + Plan B)
        # =================================================
        final_mol = Chem.AddHs(stitched_mol)

        # 建立原子映射: Core_Clean -> Final_Mol
        atom_map = []
        coord_map = {}
        conf_core = core_clean.GetConformer()

        for atom in core_mol.GetAtoms():
            old_idx = atom.GetIdx()
            if old_idx == dummy_core_idx: continue

            # 计算索引映射
            new_idx = old_idx if old_idx < dummy_core_idx else old_idx - 1

            # 只映射重原子，提高成功率
            if atom.GetAtomicNum() > 1:
                # 假设 ReplaceSubstructs 保留了重原子顺序
                core_clean_idx = old_idx
                atom_map.append((new_idx, core_clean_idx))
                coord_map[new_idx] = conf_core.GetAtomPosition(core_clean_idx)

        # --- 暂时关闭 RDKit 错误日志，避免 Plan A 失败时的红字干扰 ---
        lg = RDLogger.logger()
        original_level = lg.GetLevel()
        lg.setLevel(RDLogger.CRITICAL)

        try:
            # Plan A: 强行坐标映射
            res = AllChem.EmbedMolecule(final_mol, coordMap=coord_map, useRandomCoords=True, clearConfs=True)

            # Plan B: 自由嵌入 + 刚性对齐 (如果 Plan A 失败)
            if res == -1:
                res = AllChem.EmbedMolecule(final_mol, AllChem.ETKDGv3())
                if res != -1:
                    try:
                        rdMolAlign.AlignMol(final_mol, core_clean, atomMap=atom_map)
                    except:
                        return None  # 对齐失败
                else:
                    return None  # 彻底生成失败
        finally:
            # 恢复日志级别
            lg.setLevel(original_level)

        if res == -1: return None

        # =================================================
        # 步骤 D: 约束优化 (钉死骨架)
        # =================================================
        try:
            ff = AllChem.MMFFGetMoleculeForceField(final_mol, AllChem.MMFFGetMoleculeProperties(final_mol))
            if ff:
                for new_idx, _ in atom_map:
                    ff.AddFixedPoint(new_idx)  # 钉死骨架
                ff.Minimize(maxIts=1000)  # 优化侧链

            return Chem.RemoveHs(final_mol)
        except Exception:
            return None
    @staticmethod
    def _create_edge_labels(data, accfg_analyzer):
        mol = data.mol
        if (not mol) or (mol.GetNumBonds() == 0): return torch.zeros(data.edge_index.size(1), dtype=torch.float)

        smiles = Chem.MolToSmiles(mol)
        fg_info = get_functional_groups_with_accfg(smiles, accfg_analyzer)
        breakable_bonds_indices = set()
        for group_name, fg_indices in fg_info:
            fg_indices_set = set(fg_indices)
            for atom_idx in fg_indices:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() not in fg_indices_set:
                        bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                        breakable_bonds_indices.add(bond.GetIdx())

        labels = []
        for i in range(data.edge_index.size(1) // 2):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            bond = mol.GetBondBetweenAtoms(u, v)
            is_breakable = bond is not None and bond.GetIdx() in breakable_bonds_indices
            labels.append(1.0 if is_breakable else 0.0)

        labels_tensor = torch.tensor(labels, dtype=torch.float)
        return torch.cat([labels_tensor, labels_tensor], dim=0)

    def train_site_identifier(self, dataset, epochs=10, batch_size=32, num_workers=0):
        if epochs == 0:
            logging.info("跳过阶段一：位点识别器训练。")
            return
        logging.info("阶段一：开始训练位点识别器...")
        dataloader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.site_identifier.train()
        for name, param in self.named_parameters(): param.requires_grad = 'site_identifier' in name
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, f"Epoch {epoch + 1}/{epochs}"):
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                edge_logits = self.site_identifier(batch.x, batch.edge_index)
                edge_labels = torch.cat([g.edge_labels for g in batch.to_data_list()], dim=0).to(self.device)
                if edge_labels.numel() == 0: continue
                loss = self.site_id_criterion(edge_logits, edge_labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader) if dataloader else 0
            logging.info(f"位点识别器 Epoch {epoch + 1}, 平均损失: {avg_loss:.4f}")
            self.site_id_loss_history.append(avg_loss)
        for param in self.parameters(): param.requires_grad = True

    def encode_context(self, batch):
        batch = batch.to(self.device)
        h_updated = self.context_encoder(batch.x, batch.pos, batch.edge_index, batch.batch)
        return global_mean_pool(h_updated, batch.batch)

    def _build_training_pairs(self, dataset, threshold=0.7, sample_size=1000):
        logging.info("为预训练构建正样本对 (使用 AccFG 和去重优化)...")
        frag_bank, fp_bank, unique_frag_smiles = [], [], set()

        for data in tqdm(dataset, "1/2: 从数据集中提取并去重官能团"):
            if data and data.mol:
                smiles = Chem.MolToSmiles(data.mol)
                fg_info = get_functional_groups_with_accfg(smiles, self.accfg_analyzer)
                if fg_info:
                    fragment_pairs = fragment_by_subgraph_construction(data.mol, fg_info)
                    for _, fg in fragment_pairs:
                        if fg.GetNumHeavyAtoms() < 2 or len(GetMolFrags(fg)) > 1:
                            continue
                        fg_smiles = Chem.MolToSmiles(fg)
                        if fg_smiles not in unique_frag_smiles:
                            fp = self.isostere_miner._get_morgan_fingerprint(fg)
                            if fp:
                                unique_frag_smiles.add(fg_smiles)
                                frag_bank.append(fg)
                                fp_bank.append(fp)

        if len(frag_bank) < 2:
            logging.warning("未能找到足够的唯一片段来构建训练对。")
            return []
        logging.info(f"提取完成，共找到 {len(frag_bank)} 个唯一的、连通的官能团片段。")

        positive_pairs = []
        for i in tqdm(range(len(frag_bank)), "2/2: 采样并构建正样本对"):
            indices = list(range(i)) + list(range(i + 1, len(frag_bank)))
            sample_indices = np.random.choice(indices, min(len(indices), sample_size), replace=False)
            sims = DataStructs.BulkTanimotoSimilarity(fp_bank[i], [fp_bank[j] for j in sample_indices])
            for k, sim in enumerate(sims):
                if sim >= threshold: positive_pairs.append((frag_bank[i], frag_bank[sample_indices[k]]))

        unique_pair_keys, unique_pairs = set(), []
        for f1, f2 in positive_pairs:
            key = tuple(sorted((Chem.MolToSmiles(f1), Chem.MolToSmiles(f2))))
            if key not in unique_pair_keys:
                unique_pair_keys.add(key)
                unique_pairs.append((f1, f2))

        logging.info(f"构建完成: 找到 {len(unique_pairs)} 个唯一的正样本对。")
        return unique_pairs

    def pretrain_isostere_miner(self, dataset, epochs=10, batch_size=32, num_workers=0):
        if epochs == 0:
            logging.info("跳过阶段二：Isostere Miner 预训练，直接构建知识库。")
            self._build_final_isostere_kb(dataset)
            return
        logging.info("阶段二：开始预训练 Isostere Miner...")
        for name, param in self.named_parameters(): param.requires_grad = 'isostere_miner' in name
        training_pairs = self._build_training_pairs(dataset, threshold=0.6)
        if not training_pairs:
            logging.warning("未能构建训练对，跳过预训练。")
            self._build_final_isostere_kb(dataset)
            return
        dataloader = DataLoader(training_pairs, batch_size=batch_size, shuffle=True, collate_fn=simple_collate_fn,
                                num_workers=num_workers)
        self.isostere_miner.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, f"Epoch {epoch + 1}/{epochs}"):
                self.optimizer.zero_grad()
                features, labels, label_idx = [], [], 0
                for f1, f2 in batch:
                    feat1, feat2 = self.isostere_miner._get_fragment_features_as_fp(
                        f1), self.isostere_miner._get_fragment_features_as_fp(f2)
                    if feat1 is not None and feat2 is not None:
                        features.extend([feat1, feat2])
                        labels.extend([label_idx, label_idx])
                        label_idx += 1
                if len(features) < 2: continue
                embs = self.isostere_miner.fragment_encoder(torch.stack(features).to(self.device))
                loss = self.contrastive_criterion(embs, torch.tensor(labels, dtype=torch.long).to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader) if dataloader and len(dataloader) > 0 else 0
            logging.info(f"预训练 Epoch {epoch + 1}, 平均损失: {avg_loss:.4f}")
            self.pretrain_loss_history.append(avg_loss)
        for param in self.parameters(): param.requires_grad = True
        self._build_final_isostere_kb(dataset)

    def _build_final_isostere_kb(self, dataset):
        logging.info("基于AccFG构建最终官能团知识库...")
        all_frags = set()
        for data in tqdm(dataset, "从数据集中提取官能团"):
            if data and data.mol:
                smiles = Chem.MolToSmiles(data.mol)
                fg_info = get_functional_groups_with_accfg(smiles, self.accfg_analyzer)
                if fg_info:
                    fragment_pairs = fragment_by_subgraph_construction(data.mol, fg_info)
                    for _, fg in fragment_pairs:
                        if fg and len(GetMolFrags(fg)) == 1:
                            all_frags.add(Chem.MolToSmiles(fg))
        self.isostere_miner.update_kb([Chem.MolFromSmiles(s) for s in all_frags])

    def train_generative_model(self, dataset, epochs=50, batch_size=8, num_workers=0):
        logging.info(f"阶段三：开始训练 (基于'变换'任务)...")

        training_pairs = self._build_training_pairs(dataset, threshold=0.75, sample_size=64)

        processed_pairs = [
            pair for pair in tqdm(
                map(lambda p: (self._prepare_fragment_for_diffusion(p[0]), self._prepare_fragment_for_diffusion(p[1])),
                    training_pairs),
                total=len(training_pairs),
                desc="净化和转换'变换'数据对"
            ) if pair[0] and pair[1]
        ]

        if not processed_pairs:
            logging.error("未能创建有效的训练对，跳过扩散模型训练。")
            return

        cond_graphs = [p[1] for p in processed_pairs]
        precompute_contexts_for_dataset([p[0] for p in processed_pairs], self.pharm_factory, self.ed_vec_cfg,
                                        self.accfg_analyzer)
        precompute_contexts_for_dataset(cond_graphs, self.pharm_factory, self.ed_vec_cfg, self.accfg_analyzer)

        pair_dataloader = DataLoader(
            processed_pairs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_diffusion,
            drop_last=True, num_workers=num_workers, pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(num_workers > 0)
        )

        self.generative_model.train();
        self.pharmacophore_encoder.train();
        self.context_encoder.train();
        self.ed_context_encoder.train()
        for name, param in self.named_parameters():
            param.requires_grad = 'generative_model' in name or 'context_encoder' in name or 'pharmacophore_encoder' in name or 'ed_context_encoder' in name

        for epoch in range(epochs):
            total_loss = 0
            if len(pair_dataloader) == 0: continue
            for target_batch, condition_batch in tqdm(pair_dataloader, f"Epoch {epoch + 1}/{epochs}"):
                self.optimizer.zero_grad()
                target_batch, condition_batch = target_batch.to(self.device), condition_batch.to(self.device)

                with torch.no_grad():
                    self.context_encoder.eval()
                    geom_context_tensor = self.encode_context(condition_batch)
                    self.context_encoder.train()

                cond_list = condition_batch.to_data_list()

                chem_context_list = []
                for data in cond_list:
                    pf = getattr(data, 'pharm_feats', None)
                    pp = getattr(data, 'pharm_pos', None)
                    if pf is None or pp is None:
                        pf, pp = self.get_3d_pharmacophore_points(data.mol)
                        data.pharm_feats, data.pharm_pos = pf, pp
                    if pf is not None and pp is not None and pf.size(0) > 0:
                        with torch.no_grad():
                            self.pharmacophore_encoder.eval()
                            chem_context = self.pharmacophore_encoder(pf.to(self.device), pp.to(self.device))
                            self.pharmacophore_encoder.train()
                        chem_context_list.append(chem_context)
                    else:
                        chem_context_list.append(
                            torch.zeros(self.pharmacophore_encoder.output_proj.out_features, device=self.device))
                chem_context_tensor = torch.stack(chem_context_list, dim=0)

                ed_vec_list = []
                for d in cond_list:
                    v = getattr(d, 'ed_vec', None)
                    if v is None:
                        v_np = build_ed_context_vector(d.mol, **self.ed_vec_cfg)
                        v = torch.from_numpy(v_np.astype(np.float32))
                        d.ed_vec = v
                    ed_vec_list.append(v)
                ed_vec_tensor = torch.stack(ed_vec_list, dim=0).to(self.device)
                ed_vec_tensor = (ed_vec_tensor - self.ed_mu) / (self.ed_sigma + 1e-6)
                ed_vec_tensor = ed_vec_tensor.clamp_(-3.0, 3.0)
                ed_context_tensor = self.ed_context_encoder(ed_vec_tensor)

                alpha_target = self.ed_alpha()
                warmup_epochs = max(1, int(0.2 * epochs))
                alpha = alpha_target * (epoch + 1) / warmup_epochs if epoch < warmup_epochs else alpha_target

                final_context = self.ln_geom(geom_context_tensor) \
                                + self.ln_pharm(chem_context_tensor) \
                                + float(alpha.item()) * self.ln_ed(ed_context_tensor)

                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    loss = self.generative_model(target_batch, final_context, self.atom_type_map)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in self.parameters() if p.requires_grad), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            avg_loss = total_loss / len(pair_dataloader) if pair_dataloader else 0
            logging.info(f"扩散模型 Epoch {epoch + 1}, 平均损失: {avg_loss:.4f}")
            self.diffusion_loss_history.append(avg_loss)

        for param in self.parameters(): param.requires_grad = True

    def generate_isosteres(self, mol: Chem.Mol, top_k_fragments: int = 5):
        logging.info(f"\n" + "=" * 50)
        logging.info(f"开始为分子 {Chem.MolToSmiles(mol)} 进行生成与评估...")
        self.eval()

        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)

        smiles = Chem.MolToSmiles(mol)
        fg_info = get_functional_groups_with_accfg(smiles, self.accfg_analyzer)

        if not fg_info:
            logging.warning("AccFG 未能识别出任何可替换的官能团。")
            return []

        logging.info(f"AccFG 识别出 {len(fg_info)} 个官能团，将逐一尝试替换。")
        fragment_pairs = fragment_by_subgraph_construction(mol, fg_info)

        all_results = []
        for core_frag_wd, frag_to_replace_wd in fragment_pairs:
            logging.info(f"--- 尝试替换片段: {Chem.MolToSmiles(frag_to_replace_wd)} ---")
            isostere_candidates = self.isostere_miner.get_candidates_from_kb(frag_to_replace_wd, top_k=top_k_fragments)

            if not isostere_candidates:
                logging.info("知识库中未找到合适的候选片段。")
                continue

            logging.info(
                f"找到 {len(isostere_candidates)} 个候选片段: {[Chem.MolToSmiles(c) for c in isostere_candidates]}")

            core_data = self._prepare_fragment_for_diffusion(core_frag_wd)
            if core_data is None: continue

            if not hasattr(core_data, 'pharm_feats') or not hasattr(core_data, 'ed_vec'):
                if not hasattr(core_data, 'pharm_feats'):
                    feats, pos = self.get_3d_pharmacophore_points(core_data.mol)
                    core_data.pharm_feats = feats
                    core_data.pharm_pos = pos
                if not hasattr(core_data, 'ed_vec'):
                    v = build_ed_context_vector(core_data.mol, **self.ed_vec_cfg)
                    core_data.ed_vec = torch.from_numpy(v.astype(np.float32))

            core_batch = Batch.from_data_list([core_data])
            with torch.no_grad():
                geom_context = self.encode_context(core_batch)

                if core_data.pharm_feats is not None and core_data.pharm_pos is not None:
                    chem_context = self.pharmacophore_encoder(core_data.pharm_feats.to(self.device),
                                                              core_data.pharm_pos.to(self.device)).unsqueeze(0)
                else:
                    chem_context = torch.zeros((1, self.pharmacophore_encoder.output_proj.out_features),
                                               device=self.device)

                v = core_data.ed_vec.to(self.device)
                v = (v - self.ed_mu) / (self.ed_sigma + 1e-6)
                v = v.clamp_(-3.0, 3.0)
                ed_context = self.ed_context_encoder(v.unsqueeze(0))

                alpha = self.ed_alpha()
                final_context = self.ln_geom(geom_context) \
                                + self.ln_pharm(chem_context) \
                                + float(alpha.item()) * self.ln_ed(ed_context)

            for isostere_cand in isostere_candidates:
                if len(GetMolFrags(isostere_cand)) > 1:
                    logging.warning(f"跳过非连通的候选片段: {Chem.MolToSmiles(isostere_cand)}")
                    continue

                template_data = self._create_template_from_candidate(isostere_cand)
                if template_data is None: continue

                generated_atomic_nums, generated_pos = self.generative_model.p_sample_loop(
                    template_data.to(self.device), final_context, self.atom_type_map)
                new_frag_no_dummy = self.build_mol_from_atoms(generated_atomic_nums, generated_pos)
                if not new_frag_no_dummy: continue

                new_frag_with_dummy = Chem.Mol(isostere_cand)

                real_atom_indices = [a.GetIdx() for a in new_frag_with_dummy.GetAtoms() if a.GetAtomicNum() != 0]
                if len(real_atom_indices) != len(generated_pos):
                    logging.warning(
                        f"原子数量不匹配: 模板有 {len(real_atom_indices)} 个, 模型生成了 {len(generated_pos)} 个坐标。跳过此候选。")
                    continue

                conf = Chem.Conformer(new_frag_with_dummy.GetNumAtoms())
                for i, atom_idx in enumerate(real_atom_indices):
                    position = [float(coord) for coord in generated_pos[i]]
                    conf.SetAtomPosition(atom_idx, position)

                new_frag_with_dummy.AddConformer(conf, assignId=True)

                new_mol = self.stitch_fragments(core_frag_wd, new_frag_with_dummy)
                if not new_mol: continue

                scores = self.evaluator.evaluate(new_mol, mol)
                if scores['valid']:
                    all_results.append({
                        'new_mol': new_mol,
                        'new_mol_smiles': Chem.MolToSmiles(new_mol),
                        'original_fragment_smiles': Chem.MolToSmiles(frag_to_replace_wd),
                        'candidate_isostere_smiles': Chem.MolToSmiles(isostere_cand),
                        **scores
                    })

        return sorted(all_results, key=lambda x: x.get('composite_score', 0), reverse=True)


# =============================================================================
# 6. 可视化与主执行函数
# =============================================================================
def plot_loss_curves(pretrain_losses, diffusion_losses, site_id_losses):
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        logging.warning("中文字体 'SimHei' 未找到，图例可能显示不正确。")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('模型三阶段训练损失曲线', fontsize=16)
    axes[0].plot(site_id_losses, label='位点识别器 (BCE)', color='green')
    axes[0].set_title('阶段一: 位点识别器训练')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('平均损失')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(pretrain_losses, label='等排体挖掘器 (NTXent)', color='blue')
    axes[1].set_title('阶段二: 等排体挖掘器预训练')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('平均损失')
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(diffusion_losses, label='3D扩散模型', color='red')
    axes[2].set_title('阶段三: 3D联合生成模型训练')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('平均损失')
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("training_loss_curves_accfg.png")
    logging.info("训练损失曲线图已保存为 training_loss_curves_accfg.png")
    plt.show()


def run_comprehensive_test(model):
    logging.info("\n" + "#" * 60)
    logging.info("### 开始对固定分子列表进行综合性测试 ###")
    logging.info("#" * 60 + "\n")
    test_smiles_list = ["c1ccccc1C(=O)O", "O=C(C)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"]
    for i, smiles in enumerate(test_smiles_list):
        logging.info(f"\n--- 测试分子 {i + 1}/{len(test_smiles_list)}: {smiles} ---")
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue
        with torch.no_grad():
            results = model.generate_isosteres(mol)
        if results:
            logging.info(f"成功为分子 '{smiles}' 生成 {len(results)} 个等排体。")
            results_df = pd.DataFrame(results)
            display_df = results_df[[c for c in results_df.columns if c != 'new_mol']]
            logging.info("生成与评估结果 (Top 3):")
            try:
                print(tabulate(display_df.head(3), headers='keys', tablefmt='psql'))
            except ImportError:
                print(display_df.head(3).to_string())
        else:
            logging.warning(f"未能为分子 '{smiles}' 生成任何有效的等排体。")


def precompute_contexts_for_dataset(dataset_list, pharm_factory, ed_cfg, accfg_analyzer):
    from copy import deepcopy
    logging.info("开始为数据集预计算上下文特征...")
    for data in tqdm(dataset_list, desc="预计算特征"):
        mol = data.mol
        # 1) 药效团点
        try:
            feats = pharm_factory.GetFeaturesForMol(mol)
            fams = pharm_factory.GetFeatureFamilies()
            fam_map = {fam: i for i, fam in enumerate(fams)}
            feat_dim = len(fams)
            pharm_feats_list, pharm_pos_list = [], []
            for feat in feats:
                fam = feat.GetFamily()
                if fam in fam_map:
                    one_hot = np.zeros(feat_dim, dtype=np.float32)
                    one_hot[fam_map[fam]] = 1.0
                    pharm_feats_list.append(one_hot)
                    pharm_pos_list.append(list(feat.GetPos()))
            if pharm_feats_list:
                data.pharm_feats = torch.from_numpy(np.array(pharm_feats_list, dtype=np.float32))
                data.pharm_pos = torch.from_numpy(np.array(pharm_pos_list, dtype=np.float32))
            else:
                data.pharm_feats = None
                data.pharm_pos = None
        except Exception:
            data.pharm_feats = None
            data.pharm_pos = None

        # 2) ED 上下文向量
        try:
            v = build_ed_context_vector(mol, **ed_cfg)
            data.ed_vec = torch.from_numpy(v.astype(np.float32))
        except Exception:
            ed_vec_dim = len(ed_cfg['radii']) * len(ed_cfg['stats'])
            data.ed_vec = torch.zeros(ed_vec_dim, dtype=torch.float32)

        # 3) 边标签 (用于位点识别器)
        try:
            data.edge_labels = BioIsostericDiffusion._create_edge_labels(deepcopy(data), accfg_analyzer)
        except Exception:
            data.edge_labels = torch.zeros(data.edge_index.size(1), dtype=torch.float)
    logging.info("上下文特征预计算完成。")
    return dataset_list


def compute_ed_stats(dataset_list, ed_dim):
    vecs = []
    for d in dataset_list:
        v = getattr(d, 'ed_vec', None)
        if v is not None:
            vecs.append(v.numpy())
    if not vecs:
        mu = np.zeros((ed_dim,), dtype=np.float32)
        sigma = np.ones((ed_dim,), dtype=np.float32)
    else:
        arr = np.stack(vecs, axis=0).astype(np.float32)
        mu = arr.mean(axis=0)
        sigma = arr.std(axis=0) + 1e-6
    return torch.from_numpy(mu), torch.from_numpy(sigma)


def main():
    # --- 性能优化：AMP 和 TF32 设置 ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # --- 初始化代码 ---
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    logger = logging.getLogger("BioIsostericDiffusion_Final_Integrated")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler("bioisosteric_diffusion_final.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    accfg_analyzer = None
    if AccFG:
        accfg_analyzer = AccFG(print_load_info=False)
        logger.info("AccFG 分析器已成功初始化。")

    # --- 极速测试模式配置 ---
    TEST_MODE = 0

    if not AccFG or not EGNN:
        logger.error("核心依赖 (AccFG 或 egnn_pytorch) 未找到，程序终止。")
        return

    if TEST_MODE:
        logger.info("=" * 20 + " 运行在极速测试模式 " + "=" * 20)
        num_mols_to_use, site_id_epochs, pretrain_epochs, diffusion_epochs = 512, 0, 1, 5
        diffusion_batch_size = 16
    else:
        logger.info("=" * 20 + " 运行在完整模式 " + "=" * 20)
        num_mols_to_use, site_id_epochs, pretrain_epochs, diffusion_epochs = None, 10, 10, 50
        diffusion_batch_size = 8

    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    pharm_factory = AllChem.BuildFeatureFactory(fdef_name)
    data_dir = "./chembl_data_sample"

    cache_file = f"precomputed_dataset_{num_mols_to_use}.pkl"

    if os.path.exists(cache_file):
        logger.info(f"发现缓存文件，直接从 '{cache_file}' 加载预计算好的数据集...")
        with open(cache_file, 'rb') as f:
            data_to_split = pickle.load(f)
        logger.info("数据集加载完成。")
    else:
        logger.info("未发现缓存文件，开始加载并预计算数据集...")
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        if not os.listdir(data_dir):
            logger.info("数据目录为空，下载示例分子...");
            try:
                r = requests.get("https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL1201587.sdf")
                r.raise_for_status()
                with open(os.path.join(data_dir, "example_multimol.sdf"), 'wb') as f:
                    f.write(r.content)
                logger.info("已下载示例分子 'example_multimol.sdf'。")
            except Exception as e:
                logger.error(f"下载失败: {e}"); return

        logger.info(f"加载原始数据集 (最多 {num_mols_to_use or '所有'} 个分子)...")
        full_dataset = ChEMBLDataset(data_dir, max_mols=num_mols_to_use)
        if not full_dataset or len(full_dataset) == 0: logger.error("数据集为空或加载失败。"); return

        ed_cfg = dict(radii=(1.5, 2.5, 3.5), n_dir=64, stats=("mean", "std", "posfrac", "p10", "p90"))
        precompute_contexts_for_dataset(full_dataset.graph_data_list, pharm_factory, ed_cfg, accfg_analyzer)

        data_to_split = full_dataset.graph_data_list

        logger.info(f"预计算完成，将数据集缓存到 '{cache_file}'...")
        with open(cache_file, 'wb') as f:
            pickle.dump(data_to_split, f)

    generator = torch.Generator().manual_seed(42)
    if len(data_to_split) < 5:
        train_dataset, test_dataset = data_to_split, data_to_split
    else:
        train_size = int(0.8 * len(data_to_split))
        train_dataset, test_dataset = torch.utils.data.random_split(data_to_split,
                                                                    [train_size, len(data_to_split) - train_size],
                                                                    generator=generator)

    logger.info(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

    atom_types_list = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]

    model = BioIsostericDiffusion(
        atom_dim=6, fragment_fp_dim=2048, hidden_dim=128, lr=1e-4,
        atom_types=atom_types_list, pharm_factory=pharm_factory,
        device=device, accfg_analyzer=accfg_analyzer
    )

    ed_cfg_stats = dict(radii=(1.5, 2.5, 3.5), n_dir=64, stats=("mean", "std", "posfrac", "p10", "p90"))
    ed_mu, ed_sigma = compute_ed_stats([d for d in train_dataset],
                                       ed_dim=len(ed_cfg_stats['radii']) * len(ed_cfg_stats['stats']))
    model.register_buffer('ed_mu', ed_mu.to(device))
    model.register_buffer('ed_sigma', ed_sigma.to(device))

    num_workers = max(1, os.cpu_count() // 2) if device.type == 'cuda' else 0

    model.train_site_identifier(train_dataset, epochs=site_id_epochs, batch_size=32, num_workers=num_workers)
    model.pretrain_isostere_miner(train_dataset, epochs=pretrain_epochs, batch_size=32, num_workers=num_workers)
    model.train_generative_model(train_dataset, epochs=diffusion_epochs, batch_size=diffusion_batch_size,
                                 num_workers=num_workers)

    logger.info("所有训练阶段完成。")
    with open("isostere_kb_final.pkl", "wb") as f:
        pickle.dump(model.isostere_miner.isostere_kb, f)
    logger.info("等排体知识库已保存。")

    run_comprehensive_test(model)
    plot_loss_curves(model.pretrain_loss_history, model.diffusion_loss_history, model.site_id_loss_history)


if __name__ == "__main__":
    main()
