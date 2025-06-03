from easydict import EasyDict
import yaml
import os
import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, save_path, val_fold, patience=10, verbose=False, delta=0):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.local_optimum_pearson_train  = None
        self.local_optimum_spearman_train = None
        self.local_optimum_pearson_val  = None
        self.local_optimum_spearman_val = None
        self.val_fold = val_fold

    def __call__(self, val_loss, model, pearson_train, spearman_train, pearson_val, spearman_val):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.local_optimum_pearson_train = pearson_train
            self.local_optimum_spearman_train = spearman_train
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.early_stop = False
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.local_optimum_pearson_train = pearson_train
            self.local_optimum_spearman_train = spearman_train
            self.local_optimum_pearson_val = pearson_val
            self.local_optimum_spearman_val = spearman_val

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, '{}_fold.pth'.format(self.val_fold))
        torch.save(model.state_dict(), path)	

def consume_random_state(steps=2024):
    for _ in range(steps):
        np.random.rand()
        torch.rand(1)
        torch.tensor(1.0, device='cuda').normal_()

import pandas as pd
import random

def random_group(csv_path, num_cvfolds, seed):
    df = pd.read_csv(csv_path)
    df = df.copy()
    random.seed(seed)
    df['random_group'] = [random.randint(0, num_cvfolds - 1) for _ in range(len(df))]
    return df

def protein_level_group(csv_path, num_cvfolds, seed):
    df = pd.read_csv(csv_path)
    df = df.copy()
    pdb_id= list(df['pdb_id'].unique())
    df['protein_level_group'] = 100
    clust = {}
    for index, row in df.iterrows():
        id = row['pdb_id']
        if id not in clust:
            clust[id] = 1
        else:
            clust[id] += 1
    random.seed(seed)
    total_size = len(df)
    used = []
    cv_folds = [1/num_cvfolds for i  in range(num_cvfolds)]
    cv_folds[-1] = -1
    print('Total size:\t', total_size)
    number_group = 0

    for fold in cv_folds:
        fold_size = 0
        if fold == -1:
            df.loc[df['protein_level_group'] == 100, ['protein_level_group']] = number_group
            fold_size = len(df[df['protein_level_group'] == 2])
        else:
            while fold_size < total_size * fold:
                pick = random.randint(0, len(pdb_id)-1) 
                if pick not in used:
                    used.append(pick)
                    cluster_picked = pdb_id[pick]
                else:
                    continue
                fold_size += clust[cluster_picked]
                df.loc[df['pdb_id'] == cluster_picked, ['protein_level_group']] = number_group
        
        print('Fold Size:\t', fold_size)
        print('=' * 50)
        number_group += 1
    return df
