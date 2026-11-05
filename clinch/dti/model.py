# -*- coding: utf-8 -*-
"""Interaction predictor, inference only.

Every class and function below is lifted verbatim from the training script used to fit
the released checkpoints, so the module reproduces the saved architecture exactly. Only
the training machinery and its heavy imports were left out.

Do not reformat the definitions. They have to stay byte-identical to the versions the
state dictionaries were written from.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, TransformerConv, global_add_pool
from transformers import AutoModel, AutoTokenizer


# --- constants -----------------------------------------------------------
EPSILON = 1e-9

# --- protein model name --------------------------------------------------
PROTEIN_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# --- amino-acid property table -------------------------------------------
AA_PHYSCHEM_PROPS = {
    'A': [-0.17, -0.41, 0, 1, 1, 0], 'R': [-1.88, 0.48, 1, 5, 1, 0],
    'N': [-1.38, -0.27, 0, 2, 2, 0], 'D': [-1.38, -0.34, -1, 1, 3, 0],
    'C': [0.22, -0.24, 0, 1, 1, 0], 'Q': [-1.38, -0.06, 0, 2, 2, 0],
    'E': [-1.38, -0.03, -1, 1, 3, 0], 'G': [-0.56, -0.73, 0, 1, 1, 0],
    'H': [-1.22, 0.13, 0, 2, 2, 1], 'I': [1.00, 0.40, 0, 1, 1, 0],
    'L': [0.83, 0.40, 0, 1, 1, 0], 'K': [-1.56, 0.38, 1, 3, 1, 0],
    'M': [0.44, 0.22, 0, 1, 1, 0], 'F': [0.33, 0.60, 0, 1, 1, 1],
    'P': [-1.67, -0.15, 0, 0, 1, 0], 'S': [-0.67, -0.40, 0, 2, 2, 0],
    'T': [-0.58, -0.19, 0, 2, 2, 0], 'W': [-0.83, 1.00, 0, 2, 1, 1],
    'Y': [-0.44, 0.64, 0, 2, 2, 1], 'V': [0.94, 0.18, 0, 1, 1, 0],
    'X': [0.0, 0.0, 0, 0, 0, 0],
}
PHYSCHEM_DIM_BASE = len(list(AA_PHYSCHEM_PROPS.values())[0])
PHYSCHEM_DIM = PHYSCHEM_DIM_BASE + 1

# --- local hydrophobicity ------------------------------------------------
def _get_local_hydrophobicity(aa_sequence_list: list[str], window_size: int = 5) -> list[float]:
    hydrophobicities = [AA_PHYSCHEM_PROPS.get(aa, AA_PHYSCHEM_PROPS["X"])[0] for aa in aa_sequence_list]
    local_hydros = []
    half_window = window_size // 2
    for i in range(len(hydrophobicities)):
        start = max(0, i - half_window)
        end = min(len(hydrophobicities), i + half_window + 1)
        window_hydro = hydrophobicities[start:end]
        local_hydros.append(sum(window_hydro) / len(window_hydro))
    return local_hydros

# --- Data Splitting ---

# --- atom list -----------------------------------------------------------
ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H"]

# --- graph featurisation -------------------------------------------------
def _atom_type_onehot(a: Chem.Atom) -> list[int]:
    s = a.GetSymbol()
    return [int(s == t) for t in ATOM_LIST] + [int(s not in ATOM_LIST)]

def _atom_feat(a: Chem.Atom) -> np.ndarray:
    hyb_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    hyb = [int(a.GetHybridization() == h) for h in hyb_list]
    return np.concatenate([
        np.array([a.GetAtomicNum(), a.GetTotalNumHs(), a.GetIsAromatic(), a.GetFormalCharge(), a.GetDegree(), a.IsInRing()], dtype=np.float32),
        _atom_type_onehot(a), hyb,
    ])

def _bond_feat(b: Chem.Bond) -> np.ndarray:
    return np.array([b.GetBondTypeAsDouble(), b.GetIsConjugated(), b.IsInRing()], dtype=np.float32)

def clean_protein_sequence(sequence: str) -> str:
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    # Correctly handle potential float inputs
    seq = "".join([char for char in str(sequence).upper() if char in valid_aas])
    return " ".join(list(seq))

def smiles_to_graph(smi: str) -> Data | None:
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    atoms = [_atom_feat(a) for a in m.GetAtoms()]
    edge_idx, e_attr = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = _bond_feat(b)
        edge_idx += [(i, j), (j, i)]; e_attr += [bf, bf]
    return Data(x=torch.tensor(np.asarray(atoms), dtype=torch.float32), 
                edge_index=torch.tensor(edge_idx, dtype=torch.long).t().contiguous(), 
                edge_attr=torch.tensor(np.asarray(e_attr), dtype=torch.float32) if edge_idx else torch.empty((0, 3)), 
                smiles=smi)

# --- dataset and collate -------------------------------------------------
class DtiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_cols: dict, protein_max_len=2000, attention_supervision=False):
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.protein_max_len = protein_max_len
        self.attention_supervision = attention_supervision

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi, seq_raw = row["ligand_smiles"], row["protein_sequence"]
        
        graph_data = smiles_to_graph(smi)
        if graph_data is None: return None

        seq = clean_protein_sequence(seq_raw)
        
        outputs = {'graph': graph_data, 'seq': seq}
        
        # Add global interaction count probabilities
        for itype, cols in self.target_cols.items():
            outputs[f'y_{itype}'] = torch.tensor(row[cols].values.astype(np.float32))

        # Add attention supervision targets if enabled
        if self.attention_supervision:
            for itype in self.target_cols.keys():
                # Defensively check for column names
                pocket_col_name = f"{itype}_pocket"
                if pocket_col_name not in row:
                    pocket_col_name = f"{itype.capitalize()}_pocket"

                contact_vector = torch.zeros(self.protein_max_len, dtype=torch.float32)
                if pocket_col_name in row and pd.notna(row[pocket_col_name]):
                    try:
                        # Indices are 1-based in the CSV
                        indices = [
                            int(i.strip())
                            for i in re.split(r'[;,]', str(row[pocket_col_name]))
                            if i.strip()
                        ]
                        # Convert to 0-based for tensor, ensuring they are within bounds
                        valid_indices = [i - 1 for i in indices if 0 < i <= self.protein_max_len]
                        if valid_indices:
                            contact_vector[valid_indices] = 1.0
                    except (ValueError, AttributeError):
                        logging.warning(f"Could not parse pocket indices for SMILES {smi}. Using all-zero vector.")
                        pass 
                outputs[f'y_contact_{itype}'] = contact_vector
        
        return outputs

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    graph_data_list = [item['graph'] for item in batch]
    seq_list = [item['seq'] for item in batch]
    
    graph_batch = Batch.from_data_list(graph_data_list)
    graph_batch.protein_sequence = seq_list
    
    first_item = batch[0]
    for key in first_item:
        if key.startswith(('y_', 'y_contact_')):
            y_list = [item[key] for item in batch]
            setattr(graph_batch, key, torch.stack(y_list))

    return graph_batch

# --- Model Architecture ---

# --- ligand encoder ------------------------------------------------------
class LigandGNN(nn.Module):
    def __init__(self, in_dim, hidden_size, num_layers, heads, dropout):
        super().__init__()
        self.transformer = TransformerConv(in_dim, hidden_size, heads=heads, concat=False, dropout=dropout)
        self.bn0 = nn.BatchNorm1d(hidden_size)
        self.res_gine = nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)), edge_dim=3) for _ in range(num_layers - 1)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_layers - 1)])
    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.transformer(x, edge_index)
        x = self.bn0(x)
        for gine, bn in zip(self.res_gine, self.bn):
            x = gine(x, edge_index, edge_attr)
            x = bn(x)
        return global_add_pool(x, batch_idx)

# --- protein encoder -----------------------------------------------------
class ProteinEncoder(nn.Module):
    _class_cache: dict = {}
    def __init__(self, model_name=PROTEIN_MODEL_NAME, device="cpu", max_len=2000):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.esm_embedding_dim = self.model.config.hidden_size
        self.output_dim = self.esm_embedding_dim + PHYSCHEM_DIM
        self.max_len = max_len
    def _compute_embeddings(self, unique_seqs: list) -> None:                                                                                                                                                                                                                                                         
        """Run ESM-2 + physchem for uncached sequences and populate _class_cache."""
        if not unique_seqs:                                                                                                                                                                                                                                                                                           
            return
        # padding=True pads only to the longest seq in this call, NOT to max_len=2000.                                                                                                                                                                                                                                
        # HIV-1 protease is 99 AA -> ~101 tokens instead of 2000 -> ~400x less attention work.                                                                                                                                                                                                                        
        inputs = self.tokenizer(                                                                                                                                                                                                                                                                                      
            unique_seqs, return_tensors="pt",                                                                                                                                                                                                                                                                         
            padding=True, truncation=True, max_length=self.max_len                                                                                                                                                                                                                                                    
        ).to(self.device)                                                                                                                                                                                                                                                                                             
        with torch.no_grad():                                                                                                                                                                                                                                                                                         
            outputs = self.model(**inputs)
        esm_embeddings = outputs.last_hidden_state  # [n, padded_len, 1280]                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                        
        physchem_tensors = []                                                                                                                                                                                                                                                                                         
        for seq in unique_seqs:                                                                                                                                                                                                                                                                                       
            seq_aas_list = seq.split()                                                                                                                                                                                                                                                                                
            base_props = [AA_PHYSCHEM_PROPS.get(aa, AA_PHYSCHEM_PROPS["X"]) for aa in seq_aas_list]                                                                                                                                                                                                                   
            local_hydros = _get_local_hydrophobicity(seq_aas_list)                                                                                                                                                                                                                                                    
            combined_props = [base_props[i] + [local_hydros[i]] for i in range(len(seq_aas_list))]                                                                                                                                                                                                                    
            physchem_tensors.append(torch.tensor(combined_props, dtype=torch.float32))                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                        
        padded_physchem = nn.utils.rnn.pad_sequence(physchem_tensors, batch_first=True, padding_value=0)                                                                                                                                                                                                              
        pad_amount = esm_embeddings.shape[1] - padded_physchem.shape[1]                                                                                                                                                                                                                                               
        padded_physchem_full = F.pad(padded_physchem, (0, 0, 1, pad_amount - 1), 'constant', 0).to(self.device)                                                                                                                                                                                                       
        combined = torch.cat([esm_embeddings, padded_physchem_full], dim=-1)  # [n, padded_len, 1287]                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                        
        for i, seq in enumerate(unique_seqs):                                                                                                                                                                                                                                                                         
            ProteinEncoder._class_cache[seq] = combined[i].detach().cpu()
      
    def forward(self, sequences: list) -> torch.Tensor:
        uncached = list(dict.fromkeys(s for s in sequences if s not in ProteinEncoder._class_cache))                                                                                                                                                                                                                  
        if uncached:                                                                                                                                                                                                                                                                                                  
            self._compute_embeddings(uncached)
                                                                                                                                                                                                                                                                                                                        
        # Retrieve all from cache and pad to the longest in this batch                                                                                                                                                                                                                                                
        tensors = [ProteinEncoder._class_cache[s] for s in sequences]                                                                                                                                                                                                                                                 
        max_len = max(t.shape[0] for t in tensors)                                                                                                                                                                                                                                                                    
        embed_dim = tensors[0].shape[1]                                                                                                                                                                                                                                                                               
        batch_out = torch.zeros(len(sequences), max_len, embed_dim, device=self.device)
        for i, t in enumerate(tensors):                                                                                                                                                                                                                                                                               
            batch_out[i, :t.shape[0]] = t.to(self.device)
        return batch_out

#    def forward(self, sequences: list[str]):
 #       inputs = self.tokenizer(sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len).to(self.device)
  #      with torch.no_grad(): outputs = self.model(**inputs)
   #     esm_embeddings = outputs.last_hidden_state
    #    
    #    physchem_tensors = []
    #    for seq in sequences:
     #       seq_aas_list = seq.split(); base_props = [AA_PHYSCHEM_PROPS.get(aa, AA_PHYSCHEM_PROPS["X"]) for aa in seq_aas_list]
     #       local_hydros = _get_local_hydrophobicity(seq_aas_list)
     #       combined_props = [base_props[i] + [local_hydros[i]] for i in range(len(seq_aas_list))]
     #       physchem_tensors.append(torch.tensor(combined_props, dtype=torch.float32))
     #
#        padded_physchem = nn.utils.rnn.pad_sequence(physchem_tensors, batch_first=True, padding_value=0)
#        # Correct padding: Pad to match the ESM embeddings length, taking special tokens into account
#        pad_amount = esm_embeddings.shape[1] - padded_physchem.shape[1]
#        # ESM adds <cls> at the beginning, so we pad our physchem properties to align with the residues
#        padded_physchem_full = F.pad(padded_physchem, (0, 0, 1, pad_amount - 1), 'constant', 0).to(self.device)
#        return torch.cat([esm_embeddings, padded_physchem_full], dim=-1)

# --- cross attention -----------------------------------------------------
class SingleCrossAttention(nn.Module):
    """A single attention head for one task."""
    def __init__(self, hidden_size, protein_embedding_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(protein_embedding_dim, hidden_size)
        self.v_proj = nn.Linear(protein_embedding_dim, hidden_size)
    
    def forward(self, ligand_emb, protein_res_emb):
        q = self.q_proj(ligand_emb).unsqueeze(1)
        k, v = self.k_proj(protein_res_emb), self.v_proj(protein_res_emb)
        # Attention scores (logits) for supervision
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.hidden_size)
        # Softmax for creating the context vector
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_probs, v).squeeze(1)
        return context, attn_scores.squeeze(1) # Return raw scores (logits) for supervision loss

# --- dti model -----------------------------------------------------------
class DTI_Model(nn.Module):
    def __init__(self, cfg, in_dim, n_outputs: dict, device, protein_max_len=2000):
        super().__init__()
        h = cfg['hidden_size']
        self.interaction_types = list(n_outputs.keys())
        self.attention_supervision = cfg.get('attention_supervision', False)
        
        self.ligand_encoder = LigandGNN(in_dim, h, cfg['num_layers'], cfg['heads'], cfg['dropout'])
        self.protein_encoder = ProteinEncoder(device=device, max_len=protein_max_len)
        protein_embedding_dim = self.protein_encoder.output_dim

        # Create a separate attention head for each interaction type
        self.attention_heads = nn.ModuleDict()
        self.fc_towers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()

        for itype in self.interaction_types:
            self.attention_heads[itype] = SingleCrossAttention(h, protein_embedding_dim)
            # Each head gets its own processing tower
            self.fc_towers[itype] = nn.Sequential(
                nn.Linear(h * 2, h), 
                nn.ReLU(), 
                nn.Dropout(cfg['dropout'])
            )
            self.output_layers[itype] = nn.Linear(h, n_outputs[itype])

    def forward(self, data):
        ligand_embedding = self.ligand_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        protein_residue_embeddings = self.protein_encoder(data.protein_sequence)

        global_preds, local_preds = {}, {}
        for itype in self.interaction_types:
            protein_context, attn_logits = self.attention_heads[itype](ligand_embedding, protein_residue_embeddings)
            
            combined_embedding = torch.cat([ligand_embedding, protein_context], dim=1)
            shared_embedding = self.fc_towers[itype](combined_embedding)
            
            # Global prediction for interaction count
            global_preds[itype] = F.log_softmax(self.output_layers[itype](shared_embedding), dim=1)
            # Local prediction (attention logits) for supervision
            local_preds[itype] = attn_logits
            
        return global_preds, local_preds

# --- Loss & Training ---
