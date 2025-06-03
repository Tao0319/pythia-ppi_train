import torch
import pickle
import random
import pandas as pd
import os
from tqdm import tqdm
from Bio.SeqUtils import seq1

from torch.utils.data import  DataLoader
from utils.pythia.model import * 
from utils.pythia.pdb_utils import *

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
non_standard_residue_substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR', 
    'MLY':'LYS', 'M3L':'LYS', 'CMT':'CYS'
}
class Datasets(torch.utils.data.Dataset):
    def __init__(self, csv_path, pdb_dir, val_fold, split, feature_path):
        self.df = pd.read_csv(csv_path)
        self.entries = {
            "train":[],
            "val": []
        }
        if os.path.exists(feature_path):
            print('Loading features from {}'.format(feature_path))
            with open(feature_path, 'rb') as file:
                self.feature = pickle.load(file)
        else:
            print('Generating features to {}'.format(feature_path))
            self.feature = {}
            for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting features"):
                pdb_id = row['pdb_id']
                mt_chain = row['mutation'][1]
                ddG = row['ddG']
                pdbid_identify = row['pdb_id'] + '_' + row['mutation']
                mutation = row['mutation'][0] + row['mutation'][2:]
                pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdb_id))

                wt_id, mt_id, node_in, edge_in = preprocessing(pdb_path, mt_chain, mutation)

                self.feature[pdbid_identify] = {
                    'pdbid_identify': pdbid_identify,
                    'ddG': ddG,
                    'wt_id': wt_id,
                    'mt_id': mt_id,
                    'node_in': node_in,
                    'edge_in': edge_in
                    }
            with open(feature_path, 'wb') as file: 
                pickle.dump(self.feature, file)
        for _,row in self.df.iterrows():
            pdbid_identify = row['pdb_id'] + '_' + row['mutation']
            if row['protein_level_group'] == val_fold:
                self.entries['val'].append(pdbid_identify)
            else: 
                self.entries['train'].append(pdbid_identify)
        self.dataset = self.entries[split]
        if split == 'train': random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index):

        pdbid_identify = self.dataset[index]
        sample_feature = self.feature[pdbid_identify]

        return sample_feature
    
def extract_feature(pdb_id, pdb_dir, wt, mt, mt_position, device='cuda'):
    pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdb_id))
    protbb, chain_dict = read_pdb_to_protbb(pdb_path, return_chain_dict=True)
    seq_index = protbb.seq.detach().numpy()

    # print('________')
    # seq = ''
    # for x in seq_index:
    #     seq += alphabet[int(x[0])]
    # print(pdb_id, wt, mt_position, mt)
    # print(len(seq_index))
    # print(seq)
    # print(seq_index[int(mt_position)][0])
    # print(alphabet.index(wt))

    assert alphabet.index(wt) == int(seq_index[int(mt_position)][0])

    node, edge, seq, indices = get_neighbor(protbb, noise_level=0, mask=False)
    protbb.seq[int(mt_position)] = alphabet.index(mt)
    node_mt, edge, _, indices = get_neighbor(protbb, noise_level=0, mask=False)

    node = torch.stack([node[:,int(mt_position),:], node_mt[:,int(mt_position),:]],dim=1).unsqueeze(0)
    edge = torch.stack([edge[:,int(mt_position),:], edge[:,int(mt_position),:]],dim=1).unsqueeze(0)
    
    wt_index = alphabet.index(wt)
    mt_index = alphabet.index(mt)

    wt_id = torch.nn.functional.one_hot(torch.tensor(wt_index).long(), num_classes=21)
    mt_id = torch.nn.functional.one_hot(torch.tensor(mt_index).long(), num_classes=21)

    # node: [1, 32, 2, 28]
    # batch_size = node.shape[0]
    node_in = torch.cat([node[:,:,0,:], node[:,:,1,:]], dim=0).transpose(0,1).to(device)
    edge_in = torch.cat([edge[:,:,0,:], edge[:,:,1,:]], dim=0).transpose(0,1).to(device)
    
    return wt_id, mt_id, node_in, edge_in


def dataset_dataloader(batch_size, csv_path, pdb_dir, feature_path, val_fold, split, shuffle):
    dataset = Datasets(csv_path, pdb_dir, val_fold, split, feature_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def preprocessing(pdb_path, mt_chain, mutation):
    chain_position_residue_dict = {}
    position_residue_dict = {}
    i = 0
    wt_pdb_sequence = ''
    with open(pdb_path, 'r') as file: 
        for line in file: 
            lines = line.rstrip()
            if lines[:6] == "HETATM" and lines[17:17 + 3] == "MSE":
                lines = lines.replace("HETATM", "ATOM  ")
                lines = lines.replace("MSE", "MET")

            if lines[:4] == 'ATOM':
                if lines[17:17+3] in non_standard_residue_substitutions:
                    residue = non_standard_residue_substitutions[lines[17:17+3]]
                else:
                    residue = lines[17:17+3]

                residue = seq1(residue)
                residue_position = lines[22:22+5].rstrip()
                chain = lines[21:22]

                if i == 0: chain_letter=chain

                if chain_letter != chain :
                    chain_position_residue_dict[chain_letter] = position_residue_dict
                    chain_letter = chain
                    position_residue_dict = {}
                else:
                    i = 1
                
                position_residue_dict[residue_position] = residue
    
    chain_position_residue_dict[chain] = position_residue_dict
    pdb_mutation_position = int(mutation[1:-1])

    length_chain = 0
    j = 0
    for chain in chain_position_residue_dict:
        g = 0
        h = 0
        init = 0
        record = 0
        for position in chain_position_residue_dict[chain]:
            if g == 0: 
                g = int(position) -1
                init = int(position)
            if int(position) - g == 1:
                wt_pdb_sequence += chain_position_residue_dict[chain][position]
            else:
                a = int(position) - g - 1
                wt_pdb_sequence += "" * a
                wt_pdb_sequence += chain_position_residue_dict[chain][position]
                h += 1*a
                if pdb_mutation_position >= int(position):
                    record = h      
            g = int(position)

        if chain == mt_chain and j == 0: 
            pdb_position = length_chain + pdb_mutation_position - init - record
            j += 1

        length_chain += len(chain_position_residue_dict[chain]) 
    assert wt_pdb_sequence[pdb_position] == mutation[0]
    
    wt = mutation[0]
    mt = mutation[-1]
    k = pdb_path.rfind('/')
    pdb_dir = pdb_path[:k+1]
    pdb_id = pdb_path[k+1:].split('.')[0]

    return extract_feature(pdb_id, pdb_dir, wt, mt, pdb_position)


