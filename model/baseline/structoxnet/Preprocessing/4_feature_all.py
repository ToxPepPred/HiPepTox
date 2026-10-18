# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 9:00
# @Author  : Jiao Shihu
# @Email   : shihujiao@163.com


import pickle
import torch
import numpy as np
import torch_cluster
import pandas as pd
from Bio.PDB import PDBParser


# In[2]:


_amino_acids = lambda x: {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                          'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'LIG': 20}.get(x, 21)  # 'LIG' for small-molecule ligand
RESTYPE_3to1 = lambda x: {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                          'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}.get(x, '<unk>')  # strange residues (e.g., CA, BET)
RESTYPE_3to1_PROTTRANS = lambda x: {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                                    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}.get(x, 'X')  # strange residues (e.g., CA, BET)


# In[3]:


def read_fasta_file(filepath):
    with open(filepath, "r") as file:
        lines = [line.strip() for line in file]
        sequences_dict = {lines[i][1:]: lines[i+1] for i in range(0, len(lines), 2)}
    return sequences_dict


# In[4]:


def extract_ca_coordinates(file_path):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('structure', file_path)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    resname = residue.get_resname()
                    ca_coords.append([resname, ca_atom.coord[0], ca_atom.coord[1], ca_atom.coord[2]])
    return pd.DataFrame(ca_coords, columns=['resname', 'x', 'y', 'z'])


# In[5]:


def _normalize(tensor, dim=-1):
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


# In[6]:


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device) #在D_min和D_max之间取D_count个值，包括首尾
    D_mu = D_mu.view([1, -1]) #reshape成[1,D_count]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1) #[edge_index.shape[1], 1],升维

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2) #[edge_index.shape[1], D_count],就是把每一个距离用D_count这么个维度的向量表示
    return RBF


# In[7]:


def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]] #对应两个节点的坐标相减，[edge_index.shape[1], 3]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)
    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2) #先对E_vectors先做norm,再在中间加个维度1，[edge_index.shape[1], 1, 3]
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v)) #  维度不变化，前者[edge_index.shape[1], D_count]，后者[edge_index.shape[1], 1, 3]
    return edge_s, edge_v


# In[8]:


def cal_geo_feature(df, edge_cutoff=8, num_rbf=16, connection='rball',level='residue'):
    
    coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(), dtype=torch.float32)
    if level == 'residue':
        nodes = torch.as_tensor(list(map(_amino_acids, df.resname)), dtype=torch.long) #把残基映射成数字
    else:
        nodes = torch.as_tensor(list(map(_element_mapping, df.element)), dtype=torch.long)

    # some proteins are added by HIS or miss some residues
    if connection == 'knn':
        edge_index = torch_cluster.knn_graph(coords, k=10) 
    else:
        edge_index = torch_cluster.radius_graph(coords, edge_cutoff) 
    #

    edge_s, edge_v = _edge_features(coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)  # use RBF to represent distance
    
    return (coords, nodes, edge_index, edge_s, edge_v)


# In[ ]:


def get_geo_feature(seqencefile, featurefile, pdbfile, plmfile):
    sequences = read_fasta_file(seqencefile)
    
    erro_list = []
    with open(featurefile, 'rb') as file:
        feature_data = pickle.load(file)

    with open(plmfile, 'rb') as plmf:
        plm_data = pickle.load(plmf)
     
    peptide_names = feature_data.keys()
    df_feature = {}
    
    for name in peptide_names:
        df = extract_ca_coordinates(pdbfile+name+'.pdb')
        data = cal_geo_feature(df) 
        seq = ''.join([RESTYPE_3to1_PROTTRANS(i) for i in list(df['resname'])])
        
        if seq != sequences[name]:
            erro_list.append(sequences[name])
            
        hand_feature = torch.from_numpy(np.array(feature_data[name])).float()
        plm_data_float = torch.from_numpy(plm_data[name]).float()  # Assuming plm_data[name] is already a tensor
        node_feature = torch.cat((hand_feature, plm_data_float), dim=1)    

        data = data + (node_feature,)
        df_feature[name] = data
        
        
    return df_feature, erro_list


# In[ ]:


test_data, erro_test = get_geo_feature('./inputs/test_sequences.txt', './outputs/test_features.pkl', './inputs/pdb/', './outputs/test_plm_representation.pkl')


# In[ ]:


with open('./outputs/combined_test.pkl', 'wb') as f:
    pickle.dump(test_data, f)


# In[ ]:




