"""
Proximal Policy Optimization Algorithms (PPO):
    https://arxiv.org/pdf/1707.06347.pdf

Related Tricks(May not be useful):
    Mastering Complex Control in MOBA Games with Deep Reinforcement Learning (Dual Clip)
        https://arxiv.org/pdf/1912.09729.pdf
    A Closer Look at Deep Policy Gradients (Value clip, Reward normalizer)
        https://openreview.net/pdf?id=ryxdEkHtPS
    Revisiting Design Choices in Proximal Policy Optimization
        https://arxiv.org/pdf/2009.10897.pdf

Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations (DAPG):
        https://arxiv.org/pdf/1709.10087.pdf
"""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from time import time
from maniskill2_learn.env import build_replay, build_rollout
from maniskill2_learn.networks import build_actor_critic, build_model
from maniskill2_learn.utils.torch import build_optimizer, no_grad
from maniskill2_learn.utils.data import DictArray, GDict, to_np, to_torch
from maniskill2_learn.utils.meta import get_logger, get_world_rank, get_world_size
from maniskill2_learn.methods.builder import build_agent
from maniskill2_learn.utils.torch import (
    BaseAgent,
    RunningMeanStdTorch,
    RunningSecondMomentumTorch,
    barrier,
    get_flat_grads,
    get_flat_params,
    set_flat_grads,
)

from ..builder import MFRL

GENERATOR = 0
PLAYER = 1


@MFRL.register_module()
class LS(BaseAgent):
    def __init__(
        self,
        challenger_cfg,
        player_run_rounds,
        challenger_run_rounds,
        n_steps,
        player_cfg,
        env_params,
        env_cfg,
        num_mini_batch=-1,
        run_length=100,
        **kwargs,
    ):
        super(LS, self).__init__()
        ########################
        ## General Parameters ##
        ########################
        self.env_cfg = env_cfg
        self.num_mini_batch = num_mini_batch
        self.run_length = run_length

        self.challenger_cfg = deepcopy(challenger_cfg)
        self.player_cfg = deepcopy(player_cfg)
        ###########################
        ## Challenger Parameters ##
        ###########################
        challenger_agent_cfg = self.challenger_cfg["agent_cfg"]
        challenger_agent_cfg["env_params"] = env_params
        self.challenger_agent = build_agent(challenger_agent_cfg)
        challenger_value_cfg = self.challenger_cfg["value_cfg"]
        self.challenger_value = build_model(challenger_value_cfg)

        self.run_skill_length = self.challenger_cfg.run_skill_length
        self.run_length = run_length
        #######################
        ## Player Parameters ##
        #######################

        self.max_steps_per_skill = self.player_cfg.max_steps_per_skill
        self.num_skills = self.player_cfg.num_skills

        skill_cfg = self.player_cfg.skill_cfg
        skill_cfg.env_params = env_params

        self.comp_iter = self.player_cfg.comp_iter
        self.comp_total_run = self.comp_iter * self.run_length

        ###################
        ## Further Setup ##
        ###################
        self.active_generator = 1000

        self.previous_goal = ""
        self.current_goal = ""

        self.skill = build_agent(skill_cfg)

        self.n_steps = n_steps

        self.num_player_runs = player_run_rounds
        self.num_challenger_runs = challenger_run_rounds
        # Initialize the player and generator switch variables
        self.active_agent = self.skill
        self.run_iter = 0

        self.goals = []
        self.prior_competence_buffer = []
        self.post_competence_buffer = []
        self.memory_to_competence_inc = []

        # HERE INITILIAZE DREAMER (for both challenger and player )with build_(....)

    def generate_challenge_set(self, skill):
        eps_actor = self._build_eps_actor
        self.challenger_rollout.reset()
        trajectory = self.comp_rollout.forward_with_policy(eps_actor, self.run_skill_length)

        return 0

    def get_active_agent_name(self):
        return "player" if self.active_agent == self.skill else "challenger"

    def estimate_skill_competence(self, obs, goal):
        """Estimates how good a policy is at achieving a challenge as the sum of the reward given a short trajectory on it."""
        total_reward = self.challenge_reward(obs, goal)
        return total_reward / self.comp_iter

    def challenge_reward(self, obs, goal):
        """Given an observation, uses the embedding to calculate how close the observation is to the goal"""
        return 0

    def _build_eps_actor(self, obs, alpha=0.2):
        """Does not actually build anything, but just returns a function, that can be called with obs to return an action."""

        pass

    def embed(self, obs):
        return np.array([0] * len(obs["xyz"].shape[0]))

    def modify_memory_reward(self, memory, goal):
        """Modifies the reward in memory to measure how close to reward. Scaled by some value."""
        memory_embedding = self.embed(memory.get_all()["obs"])
        goal_embedding = self.embed(self.goals[-1])
        reward = np.linalg.norm(memory_embedding, goal_embedding, ord=1)
        memory["rewards"] = reward

    def update_parameters(self, memory, updates, with_v=False, rollout=None):
        world_size = get_world_size()
        logger = get_logger()
        ret = defaultdict(list)
        current_goal = self.goals[-1]

        if self.get_active_agent_name() == "player":
            if self.run_iter + 1 < self.num_player_runs:
                self.prior_competence_buffer.append(self.estimate_skill_competence(memory, current_goal))

            if self.run_iter > 0:
                self.post_competence_buffer.append(self.estimate_skill_competence(memory, current_goal))

            # in last round calculate the final progress on challenge
            if self.run_iter == self.num_player_runs - 1:
                progress = 0
                for idx in range(1, self.num_player_runs + 1):
                    # TODO: get scaling factor of length of embedding
                    progress += self.post_competence_buffer[-idx] - self.prior_competence_buffer[-idx]
                    #
                # calculate progress with -log(sum())

                # TODO:

                # add progress to
                self.memory_to_competence_inc.append((current_goal, progress))

            # train skill network

            # change reward for self.goals[-1]
            self.modify_memory_reward(memory, self.goals[-1])
            self.skill.update_parameters(memory)

            pass

        if self.get_active_agent_name() == "challenger":
            # self.challenger_agent
            # train challenger
            pass

        self.check_and_update_active_agent()

        return None

    def check_and_update_active_agent(self):
        """Checks if the active player needs to be changed and then resets self.run_iter"""
        self.run_iter += 1
        if self.get_active_agent_name() == "player":
            if self.run_iter >= self.num_player_runs:
                self.active_agent = self.challenger_agent
                self.run_iter = 0
        if self.get_active_agent_name() == "challenger":
            if self.run_iter >= self.num_challenger_runs:
                self.active_agent = self.skill
                self.run_iter = 0

    @no_grad
    def forward(self, obs, **kwargs):
        obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs) if self._in_test else self.obs_rms.add(obs)
        if self.obs_processor is not None:
            obs = self.obs_processor({"obs": obs})["obs"]
        return self.active_agent(obs)
