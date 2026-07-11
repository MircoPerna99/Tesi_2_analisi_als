import torch
import esm
import math
import numpy as np
from tqdm import tqdm
import os
import shutil
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator,MACCSkeys,Descriptors, rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler
from utils import DTADataset


def scaler_global_drug_features(drug_dict):
    keys = list(drug_dict.keys())
    physchem_matrix = np.stack([drug_dict[k][0] for k in keys]) 

    scaler_drug = StandardScaler()
    scaler_drug.fit(physchem_matrix)

    physchem_scaled = scaler_drug.transform(physchem_matrix)

    drug_dict_scaled = {
        k: (torch.tensor(physchem_scaled[i], dtype=torch.float), drug_dict[k][1])
        for i, k in enumerate(keys)
    }
    
    return drug_dict_scaled


def scaler_global_target_features(prot_dict):
    keys_prot = list(prot_dict.keys())
    global_matrix = np.stack([prot_dict[k][0] for k in keys_prot])
    scaler_prot_global = StandardScaler()
    scaler_prot_global.fit(global_matrix)
    global_scaled = scaler_prot_global.transform(global_matrix)

    prot_dict_scaled = {
        k: (torch.tensor(global_scaled[i], dtype=torch.float), 
            prot_dict[k][1]) 
        for i, k in enumerate(keys_prot)
    }
    return prot_dict_scaled


def extract_protein_features(sequence):
    clean_seq = ''.join([str(aa) for aa in sequence if str(aa) in 'ACDEFGHIKLMNPQRSTVWY'])
    
    analysis = ProteinAnalysis(clean_seq)
    
    mw = analysis.molecular_weight()
    pi = analysis.isoelectric_point()
    gravy = analysis.gravy()                        
    instability = analysis.instability_index()
    length = len(clean_seq)
    
    aa_composition = list(analysis.get_amino_acids_percent().values())
    
    helix, turn, sheet = analysis.secondary_structure_fraction()
    
    global_props = [mw, pi, gravy, instability, length, helix, turn, sheet]
    
    features_global_props = np.array(global_props, dtype=np.float32)
    features_aa = np.array(aa_composition, dtype=np.float32)
    return (torch.tensor(features_global_props, dtype=torch.float),torch.tensor(features_aa, dtype=torch.float))

def extract_drug_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)      # H-bond donors
    hba = rdMolDescriptors.CalcNumHBA(mol)      # H-bond acceptors
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    
    maccs = list(MACCSkeys.GenMACCSKeys(mol))
    
    physchem = [mw, logp, tpsa, hbd, hba, rotatable, aromatic_rings, stereocenters]
    
    features_physchem = np.array(physchem, dtype=np.float32)
    features_maccs = np.array(maccs, dtype=np.float32)
    return (torch.tensor(features_physchem, dtype=torch.float),torch.tensor(features_maccs, dtype=torch.float))

def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# one ont encoding
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(seq):
    residue_feature = []
    for residue in seq:
        # replace some rare residue with 'X'
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]

    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature


def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_size = mol.GetNumAtoms()

    mol_features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        mol_features.append(feature / sum(feature))

    edges = []
    bond_type_np = np.zeros((mol_size, mol_size))
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        bond_type_np[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetBondTypeAsDouble()
        bond_type_np[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetBondTypeAsDouble()
        # bond_type.append(bond.GetBondTypeAsDouble())
    g = nx.Graph(edges).to_directed()
    # print('@@@@@@@@@@@@@@@@@')
    # print(np.array(edges).shape,'edges')
    # print(np.array(g).shape,'g')

    mol_adj = np.zeros((mol_size, mol_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    # print(np.array(mol_adj).shape,'mol_adj')
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))

    bond_edge_index = []
    bond_type = []
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        bond_edge_index.append([i, j])
        bond_type.append(bond_type_np[i, j])
    # print(bond_edge_index)
    # print('smile_to_graph')
    # print('mol_features',np.array(mol_features).shape)
    # print('bond_edge_index',np.array(bond_edge_index).shape)
    # print('bond_type',np.array(bond_type).shape)
    return mol_size, mol_features, bond_edge_index, bond_type
    # return mol_size, mol_features, bond_edge_index


# target sequence to target graph
def sequence_to_graph(target_key, target_sequence, distance_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)
    # print('***',(os.path.abspath(os.path.join(distance_dir, target_key + '.npy'))))
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    distance_map = np.load(contact_map_file)
    # the neighbor residue should have a edge
    # add self loop
    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    # print(distance_map)
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold
    # print(len(index_row))
    # print(len(index_col))
    # print(len(index_row_))
    # print(len(index_col_))
    # print(distance_map.shape)
    # print((len(index_row) * 1.0) / (distance_map.shape[0] * distance_map.shape[1]))
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege
        target_edge_distance.append(distance_map[i, j])  # edge weight
    target_feature = seq_feature(target_sequence)
    # residue_distance = np.array(target_edge_distance)  # consistent with edge
    # print('target_feature', target_feature.shape)
    # print(target_edge_index)
    # print('target_edge_index', np.array(target_edge_index).shape)
    # print('residue_distance', residue_distance.shape)
    # return target_size, target_feature, residue_edge_index, residue_distance
    return target_size, target_feature, target_edge_index, target_edge_distance



class DataManager():
    def __init__(self):
        self.save_dir = 'pre_process/distance_map/'
        if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
    
    
    def protein_graph_construct(self,proteins):
        # Load ESM-1b model
        # torch.set_grad_enabled(False)
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S() #model pre-trained for proteins, alphaber the dictionar of the token 
        batch_converter = alphabet.get_batch_converter()
        target_graph = {}

        count = 0
        for index in tqdm(range(len(proteins))): #is a library for progress bar
            pro_id=proteins[index][0]

            data = []            
            seq = proteins[index][1].decode('utf-8').strip()
            if len(seq) <= 1000:
                data.append((pro_id, seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                contact_map = results["contacts"][0].numpy()
                target_graph[pro_id] = contact_map
            else:
                contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
                interval = 500
                i = math.ceil(len(seq) / interval)
            #     # ======================
            #     # =                    =
            #     # =                    =
            #     # =          ======================
            #     # =          =*********=          =
            #     # =          =*********=          =
            #     # ======================          =
            #     #            =                    =
            #     #            =                    =
            #     #            ======================
            #     # where * is the overlapping area
            #     # subsection seq contact map prediction
                for s in range(i):
                    start = s * interval  # sub seq predict start
                    end = min((s + 2) * interval, len(seq))  # sub seq predict end
                    sub_seq_len = end - start

                    # prediction
                    temp_seq = seq[start:end]
                    temp_data = []
                    temp_data.append((pro_id, temp_seq))
                    batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                    with torch.no_grad():
                        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    # insert into the global contact map
                    row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                    row = row + start
                    col = col + start
                    contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                        0].numpy()
                    contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
                    if end == len(seq):
                        break
                target_graph[pro_id] = contact_prob_map

            np.save(self.save_dir + pro_id + '.npy', target_graph[pro_id])
            count += 1

    def delete_graph(self):
        if os.path.exists(self.save_dir ):
            shutil.rmtree(self.save_dir )
            
    def prepare_set(self, drug_target_list):
        drug_smiles = []
        pro_seqs = {}
        pro_keys = []
        for element in drug_target_list:
            drug_smiles.append(element[1])
            pro_keys.append(element[2])
            if(not element[2] in pro_seqs):
                pro_seqs[element[2]]=element[3].decode('utf-8').strip()

        drug_smiles = list(set(drug_smiles))
        pro_keys = list(set(pro_keys))

        # create target graph
        global_feature_protein = {}
        target_graph = {}
        for i in tqdm(range(len(pro_keys))):
            key = pro_keys[i]
            seq = pro_seqs[key]
            g_t = sequence_to_graph(key, seq, self.save_dir)
            target_graph[key] = g_t
            global_feature_protein[key] = extract_protein_features(seq)

        global_feature_protein = scaler_global_target_features(global_feature_protein)
        
        # create smile graph
        global_feature_drug = {}
        smile_graph = {}
        for i in tqdm(range(len(drug_smiles))):
            smile = drug_smiles[i]
            g_d = smile_to_graph(smile)
            smile_graph[smile] = g_d
            global_feature_drug[smile] = extract_drug_features(smile)

        global_feature_drug = scaler_global_drug_features(global_feature_drug)


        train_drugs = np.asarray([x[1] for x in drug_target_list])
        train_prot_keys = np.asarray([x[2] for x in drug_target_list])
        train_Y = np.asarray([x[4] if len(x) > 4 else 1 for x in drug_target_list])
        dataset = DTADataset(root='data', dataset="", xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y.astype(float), smile_graph=smile_graph, target_graph=target_graph, global_feature_drug=global_feature_drug, global_feature_protein = global_feature_protein)

        return dataset
