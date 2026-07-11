import os
import json, pickle
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator,MACCSkeys,Descriptors, rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import requests
import time


def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    fp = gen.GetFingerprint(mol)
    
    fp_np = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_np)
    
    return torch.tensor(fp_np, dtype=torch.float)

# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None, global_feature_drug= None,global_feature_protein=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug = xd
        self.target = target_key
        self.y = y
        self.smile_graph = smile_graph
        self.target_graph = target_graph
        self.global_feature_drug = global_feature_drug
        self.global_feature_protein = global_feature_protein
        self.process(xd, target_key, y, smile_graph, target_graph)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must have the same length!'
        data_list_mol = []
        data_list_pro = []
        data_len = len(xd)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):
            smiles = xd[i]
            tar_key = target_key[i]
            labels = y[i]
            # print(labels,type(labels))
            # convert SMILES to molecular representation using rdkit
            mol_size, mol_features, mol_edge_index, mol_edges_weights = smile_graph[smiles]
            # c_size, features, edge_index = smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_key]
            # target_size, target_features, target_edge_index = target_graph[tar_key]
            features_physchem_drug = self.global_feature_drug[smiles][0]
            features_maccs_drug = self.global_feature_drug[smiles][1]
            global_features_target,features_aa_target = self.global_feature_protein[tar_key]
            
            # print(i,target_size,target_features.shape, target_edge_index.shape,target_edge_weight.shape,y[i])
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            # print(fp)
            GCNData_mol = DATA.Data(x=torch.Tensor(mol_features),
                                    edge_index=torch.LongTensor(mol_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(mol_edges_weights),
                                    features_physchem_drug=features_physchem_drug.unsqueeze(0),
                                    features_maccs_drug=features_maccs_drug.unsqueeze(0),
                                    smiles = smiles.decode('utf-8').strip(),
                                    target_key = tar_key,
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(target_edge_weight),
                                    global_features=global_features_target.unsqueeze(0),
                                    features_aa = features_aa_target.unsqueeze(0),
                                    smiles = smiles.decode('utf-8').strip(),
                                    target_key = tar_key,
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
            # print(GCNData_pro.x.size(), GCNData_pro.edge_index.size(), GCNData_pro.y.size())
            # print(GCNData_pro.edge_index)
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
        return self.data_mol[idx], self.data_pro[idx]


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn, TRAIN_BATCH_SIZE=512):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        # if batch_idx % LOG_INTERVAL == 0:
        #     print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
        #                                                                    batch_idx * TRAIN_BATCH_SIZE,
        #                                                                    len(train_loader.dataset),
        #                                                                    100. * batch_idx / len(train_loader),
        #                                                                    loss.item()))


# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

import pandas as pd

def predicting_rec(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    
    # Liste per raccogliere i metadati
    xd_list = []
    target_key_list = []
    
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
            
            # Raccolta di xd e target_key per ogni batch
            xd_list.extend(data_mol.smiles)
            target_key_list.extend(data_mol.target_key)
    
    preds = total_preds.numpy().flatten()
    
    # Creazione del DataFrame
    df = pd.DataFrame({
        'drug_structure': xd_list,
        'protein_id': target_key_list,
        'prediction': preds
    })
    
    return df


# prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


def smiles_to_pubchem(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName,Title/JSON"
    
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            props = r.json()["PropertyTable"]["Properties"][0]
            return props.get("Title", props.get("IUPACName"))
        except requests.RequestException:
            time.sleep(2)

def uniprot_to_name(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            gene = data.get("genes", [{}])[0].get("geneName", {}).get("value", "")
            name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
            return gene, name
        except requests.RequestException:
            time.sleep(2)
    return None, None


def complete_dataframe_with_names(df):
    df = df[df["prediction"] >= 0.50].copy()
    print(f"Number of rows after filtering with a prediction threshold of 0.85: {df.shape[0]}")
    drug_cache = {}
    protein_cache = {}

    def get_drug(smiles):
        if smiles not in drug_cache:
            drug_cache[smiles] = smiles_to_pubchem(smiles)
        return drug_cache[smiles]

    def get_protein(pid):
        if pid not in protein_cache:
            protein_cache[pid] = uniprot_to_name(pid)
        return protein_cache[pid]

    print("Fetching drug and protein names...")
    print("Fetching drug names...")
    df["drug_name"] = df["drug_structure"].apply(get_drug)
    print("Fetching drug names completed.")
    print("Fetching protein names...")
    df[["gene_name", "protein_name"]] = (
        df["protein_id"]
        .apply(get_protein)
        .apply(pd.Series)
    )
    print("Fetching protein names completed.")

    return df
