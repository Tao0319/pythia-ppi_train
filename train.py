import argparse
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr
from utils.misc import set_seed, EarlyStopping, protein_level_group, random_group
from utils.dataset import dataset_dataloader 
from utils.model import Pythia_PPI, get_torch_model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('csv_path', type=str)
    argparser.add_argument('pdb_dir', type=str)
    argparser.add_argument('--feature_path', type=str, default='./feature.pkl')
    argparser.add_argument('--pth_save_dir', type=str, default='./pth_save/')
    argparser.add_argument('--corr_path', type=str, default='./corr.txt')
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--seed', type=int, default=2024)
    argparser.add_argument('--epoch', type=int, default=100)
    argparser.add_argument('--learning_rate', type=int, default=1e-4)
    argparser.add_argument('--num_cvfolds', type=int, default=5)
    args = argparser.parse_args()
     
    
    # Split the dataset into training and validation groups based on the protein_level_group 
    protein_level_group(args.csv_path, args.num_cvfolds,args.seed).to_csv(args.csv_path, index=False)

    # Split the dataset into training and validation groups based on the protein_level_group labels
    # random_group(args.csv_path, args.num_cvfolds,args.seed).to_csv(args.csv_path, index=False)

    if not os.path.exists(args.pth_save_dir):
        os.makedirs(args.pth_save_dir)

    val_folds = [i for i in range(args.num_cvfolds)]

    file = open(args.corr_path, 'a')
    for val_fold in val_folds:
        set_seed(args.seed)
        print(f'Train: {val_fold} fold')
        print('Loading datasets...')
        train_dataloader = dataset_dataloader(
            args.batch_size,
            args.csv_path,
            args.pdb_dir,
            args.feature_path,
            val_fold,
            split='train', 
            shuffle=True)
        val_dataloader = dataset_dataloader(
            args.batch_size,
            args.csv_path,
            args.pdb_dir,
            args.feature_path,
            val_fold,
            split='val', 
            shuffle=False) 
        # Model,Optimizer &Scheduler
        torch_model_p = get_torch_model("./utils/pythia/pythia-p.pt", args.device)
        pythia_ppi = Pythia_PPI(torch_model_p)
        pythia_ppi.to(args.device)
        early_stopping = EarlyStopping(args.pth_save_dir, val_fold)
        cal_loss = torch.nn.L1Loss()
        opt = torch.optim.Adam(pythia_ppi.parameters(), lr=args.learning_rate)
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 
            mode='min', 
            factor=0.1, 
            patience=5, 
            verbose=True, 
            threshold=0.0001, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=0, 
            eps=1e-08
            )
        # Train & Validation
        for epoch in tqdm(range(args.epoch), desc='training'):
            train_loss_list = []
            valid_loss_list = []

            train_ddG_true_list = []
            train_ddG_pred_list = []

            skempi_train_ddG_true_list = []
            skempi_train_ddG_pred_list = []
            fireprot_train_ddG_true_list = []
            fireprot_train_ddG_pred_list = []

            val_ddG_true_list = []
            val_ddG_pred_list = []

            train_pdbid_identify_list = []
            val_pdbid_identify_list = []

            for feature_dict in train_dataloader:
                
                pdbid_identify = feature_dict['pdbid_identify']
                wt_id = feature_dict['wt_id']
                mt_id = feature_dict['mt_id']
                node_in = feature_dict['node_in']
                edge_in = feature_dict['edge_in']

                ddG_pred, logits = pythia_ppi(
                    wt_id.to(args.device), 
                    mt_id.to(args.device),
                    node_in.to(args.device),
                    edge_in.to(args.device)
                    )

                train_ddG_true_list.extend(feature_dict['ddG'].detach().cpu().numpy().tolist())
                train_ddG_pred_list.extend(ddG_pred.detach().cpu().numpy().tolist())
                train_pdbid_identify_list.extend(pdbid_identify)
                ddG_true = feature_dict['ddG'].to(args.device)
                
                opt.zero_grad() 
                train_loss = cal_loss(ddG_pred, ddG_true)
                train_loss.backward()
                opt.step() 
                train_loss_list.append(train_loss.item())
            
            for feature_dict in val_dataloader:
                pdbid_identify = feature_dict['pdbid_identify']
                wt_id = feature_dict['wt_id']
                mt_id = feature_dict['mt_id']
                node_in = feature_dict['node_in']
                edge_in = feature_dict['edge_in']

                with torch.no_grad():
                    ddG_pred, logits = pythia_ppi(
                        wt_id.to(args.device), 
                        mt_id.to(args.device),
                        node_in.to(args.device),
                        edge_in.to(args.device)
                        )
                
                val_ddG_true_list.extend(feature_dict['ddG'].detach().cpu().numpy().tolist())
                val_ddG_pred_list.extend(ddG_pred.detach().cpu().numpy().tolist())
                val_pdbid_identify_list.extend(pdbid_identify)
                ddG_true = feature_dict['ddG']

                val_loss = cal_loss(ddG_pred, ddG_true.to(args.device)) 
                valid_loss_list.append(val_loss.item())
         
            train_pearson = pearsonr(train_ddG_true_list, train_ddG_pred_list)[0]
            train_spearman = spearmanr(train_ddG_true_list, train_ddG_pred_list)[0]

            val_pearson = pearsonr(val_ddG_true_list, val_ddG_pred_list)[0]
            val_spearman = spearmanr(val_ddG_true_list, val_ddG_pred_list)[0]
            print(f'train\t{np.mean(train_loss_list)}\tpearson\t{train_pearson}\tspearman\t{train_spearman}')
            print(f'val\t{np.mean(valid_loss_list)}\tpearson\t{val_pearson}\tspearman\t{val_spearman}')
            
            early_stopping(np.mean(valid_loss_list), pythia_ppi, train_pearson, train_spearman, val_pearson, val_spearman)

            if early_stopping.early_stop:
                print('Early stopping')
                break
            schedule.step(np.mean(valid_loss_list))
        print('Finish!')
        file.write(f'{val_fold} fold validation \n')
        file.write(f'train___{early_stopping.local_optimum_pearson_train}___{early_stopping.local_optimum_spearman_train}')
        file.write('\n')
        file.write(f'val___{early_stopping.local_optimum_pearson_val}____{early_stopping.local_optimum_spearman_val}')
        file.write('\n')
