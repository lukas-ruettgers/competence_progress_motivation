"""
Soft Actor-Critic Algorithms and Applications:
    https://arxiv.org/abs/1812.05905
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor:
   https://arxiv.org/abs/1801.01290
"""
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from maniskill2_learn.env import build_replay, build_rollout
from maniskill2_learn.networks import build_model, build_actor_critic
from maniskill2_learn.utils.torch import build_optimizer, no_grad, BaseAgent, hard_update, soft_update
from maniskill2_learn.utils.data import DictArray, GDict, to_np, to_torch
from ..builder import MFRL
from copy import deepcopy
from time import time
from collections import defaultdict, OrderedDict


@MFRL.register_module()
class DreamerAgent(BaseAgent):
    def __init__(
        self,
        dreamer_cfg,
        env_params,
        dreamer_updates=1
    ):
        super(DreamerAgent, self).__init__()
        dreamer_cfg = deepcopy(dreamer_cfg)
        dreamer_cfg['env_params'] = deepcopy(env_params)
        self.dreamer_agent = build_model(dreamer_cfg)
        
        self.dreamer_updates = dreamer_updates
        # for evaluation
        self.actor = self.dreamer_agent
        # for test
        self.PRINT_DETAILED_TIME = False
    @no_grad
    def forward(self, obs, **kwargs):
        obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs) if self._in_test else self.obs_rms.add(obs)
        if self.obs_processor is not None:
            obs = self.obs_processor({"obs": obs})["obs"]

        return self.dreamer_agent(obs)

    def update_parameters(self, memory, updates, rollout, **kwargs):

        self.measure_time("start", reset=True)
        recent_replay = kwargs["recent_traj_replay"] 
        recent_memory = recent_replay.get_all()

        start_indices, end_indices = self.get_start_end_indices(recent_memory)
        self.measure_time("get_start_end")
        trajectories = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            trajectory_obs = self.slice_dictionary_copy(recent_memory, start_idx, end_idx)
            trajectories.append(trajectory_obs)
            self.measure_time("slice_trajectory")
        
        for _ in range(self.dreamer_updates):
            for trajectory in trajectories:
                ret_ = self.dreamer_agent._train(trajectory)
                self.join_key_returns(ret_, None)            
                self.measure_time("dreamer")
        


        self.print_time()
        return self.return_ret_dict()

    def train_dreamer(self, trajectory):
        # TODO (Lukas): Implement the dreamer function with rollout
        return self.dreamer_agent._train(trajectory)
    

    def get_start_end_indices(self, recent_memory):
        # TODO: Wollen wir die letzte truncated Trajectory garantiert einbinden, auch wenn sie erfolglos war?
        done_trajectory = recent_memory["episode_dones"] + recent_memory["is_truncated"]
        done_trajectory_indices = np.where(done_trajectory == True)[0]

        # find the start and end indices of each trajectory
        start_indices = (0,) + tuple(done_trajectory_indices + 1)
        end_indices = tuple(done_trajectory_indices + 1)

        if end_indices[-1] != len(recent_memory):
            end_indices += (len(recent_memory),)
        else:
            start_indices = start_indices[:-1]
        return start_indices, end_indices

    def slice_dictionary_copy(self, input_dict, start, end, keys=[]):
        """slices the input dictionary along all key in keys on the first axis from start to end with copy"""
        output_dict = {}
        keys = input_dict.keys() if len(keys) == 0 else keys

        for key in keys:
            if isinstance(input_dict[key], dict):
                output_dict[key] = self.slice_dictionary_copy(input_dict[key], start, end)
            else:
                output_dict[key] = np.copy(input_dict[key][start:end])
        return output_dict
