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
class Embedder(ExtendedModule):
    def __init__(self, feat_size, lr, action_shape, actor_state_length, img_size):
        super(Embedder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 5)
        self.fc_size = 200 * 2 + actor_state_length
        self.fc1 = nn.Linear(self.fc_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, feat_size)
        self.img_size = img_size
        


        self.out_size = self.img_size * self.img_size * 6 + actor_state_length # 
        self.actor_state_length = actor_state_length
        self.fc4 = nn.Linear(feat_size, 84)
        self.fc5 = nn.Linear(84, 128)
        self.fc6 = nn.Linear(128, self.out_size)

        self.optim= optim.Adam(self.parameters(), lr)
        self.loss = nn.MSELoss()

    def conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x
    

    def embed(self, obs, get_inp=False):
        images, state = self.preprocess_obs(obs)
        emb = []
        for image in images:
            emb.append(self.conv(image))
        emb += [state]
        assert len(emb) == 3
        nn_input = torch.concatenate(emb, dim=1)
        nn_input = F.relu(self.fc1(nn_input))
        nn_input = F.relu(self.fc2(nn_input))
        nn_input = self.fc3(nn_input)

        if get_inp:
            return nn_input, state, images
        return nn_input

    def update_parameters(self, obs):

        orig_emb, orig_state, orig_images = self.embed(obs['obs'], get_inp=True)
        recon_images, recon_state = self.unembed(orig_emb)
        # also embed next_obs

        # convert loss
        ret = {}

        img_loss = self.loss(recon_images, orig_images)
        ret["img_loss"] = img_loss.item()

        state_loss = self.loss(orig_state, recon_state)
        ret['state_loss'] = state_loss.item()


        loss = img_loss + state_loss
        loss.backward(retain_graph=True)
        self.optim.step()
        self.optim.zero_grad()
        ret['loss'] = loss.item()

        return ret, orig_emb
    

    def preprocess_obs(self, obs):
        image = obs['rgb_image']
        state = obs['state']
        if len(image.shape) == 3:
            image = image[None]
            state = state[None]

        images = (torch.tensor(image).float().to(self.device) - 128) / 128
        images = images.reshape(images.shape[0], -1, 3, self.img_size,self.img_size)     
        images = torch.transpose(images, 0,1)
        return images, torch.tensor(state).to(self.device)
        
    def unembed(self, x):
        nn_input = F.relu(self.fc4(x))
        nn_input = F.relu(self.fc5(nn_input))
        nn_input = self.fc6(nn_input)
        # reshape image
        batch_size = nn_input.shape[0]
        *images, state = torch.split(nn_input, [3*self.img_size**2,3*self.img_size**2,self.actor_state_length], dim=1)
        out_img = []
        for image in images:
            out_img.append(image.view(batch_size, -1, self.img_size, self.img_size))

        return torch.stack(out_img), state