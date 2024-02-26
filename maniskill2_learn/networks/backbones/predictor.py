from maniskill2_learn.networks.modules.norm import need_bias
import torch.nn as nn, torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from einops.layers.torch import Rearrange
import torch
from torch import distributions as torchd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.utils.torch import load_checkpoint, ExtendedModule
import torch.optim as optim
from ..builder import BACKBONES


@BACKBONES.register_module()
class Predictor(ExtendedModule):
    def __init__(self, feat_size, lr, action_shape):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(feat_size + action_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, feat_size)

        self.optim= optim.Adam(self.parameters(), lr)
        self.loss = nn.MSELoss()


    def forward(self, nn_input):
        nn_input = F.relu(self.fc1(nn_input))
        nn_input = F.relu(self.fc2(nn_input))
        nn_input = F.relu(self.fc3(nn_input))
        return nn_input
    
    def update_parameters(self, obs, feat, next_feat):  
        ret = {}
        state = torch.tensor(obs['actions']).to(self.device)
        inp = torch.cat((feat,state), dim=1)
        inp = self.forward(inp)
        output = self.loss(inp, next_feat)
        output.backward()

        ret['loss'] = output.item()
        
        self.optim.step()
        self.optim.zero_grad()
        return ret
    
    def uncertainty(self, cur_emb, next_emb, actions):
        state = torch.tensor(actions).to(self.device)
        inp = torch.cat((cur_emb,state), dim=1)
        inp = self.forward(inp)
        return torch.norm(inp - next_emb,1,dim=1).detach().cpu().numpy()
