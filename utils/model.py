import torch
import torch.nn as nn
from utils.pythia.model import AMPNN

def get_torch_model(ckpt_path, device='cuda'):
    model = AMPNN(
        embed_dim = 128,
        edge_dim = 27,
        node_dim = 28,
        dropout = 0.2,
        layer_nums = 3,
        token_num = 21,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    model.eval()
    model.to(device)
    return model

class Pythia_PPI(nn.Module):
    def __init__(self, encoders):
        super().__init__()
        self.encoders = encoders
        self.mlp = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.logits_trans = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        self.out1 = nn.Linear(128, 1)
        self.out2 = nn.Linear(128, 1)
        
    def forward(self, wt_id, mt_id, node_in, edge_in, batch_size=1):
        node_in = node_in.transpose(0,1) 
        node_in_wt = node_in[:,:,0,:]
        node_in_mt = node_in[:,:,1,:]
        node_in = torch.cat([node_in_wt, node_in_mt], dim=1)
        
        edge_in = edge_in.transpose(0,1)
        edge_in_wt = edge_in[:,:,0,:]
        edge_in_mt = edge_in[:,:,1,:]
        edge_in = torch.cat([edge_in_wt, edge_in_mt], dim=1)

        output = self.encoders(node_in, edge_in)
        h = output['hidden_states'] 
        shape = int(output['logits'].shape[0]/2)

        wt_logits = output['logits'][:shape,:]
        mt_logits = output['logits'][shape:,:]
        
        hiddens = torch.cat((h[0].reshape(shape*2,batch_size,128), h[1].reshape(shape*2,batch_size,128), h[2].reshape(shape*2,batch_size,128)), -1)
        hiddens_out = self.mlp(hiddens)
        hiddens_out = self.out1(hiddens_out)
        hiddens = (hiddens_out[shape:,0,0] - hiddens_out[:shape,0,0])

        logits = (wt_logits.squeeze(1)*wt_id).sum(dim=1) - (mt_logits.squeeze(1)*mt_id).sum(dim=1) 
        logits = self.logits_trans(logits.unsqueeze(-1))
        logits = self.out2(logits).squeeze(-1)
        
        out = hiddens + logits
        
        return out, logits