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
from maniskill2_learn.networks import build_model, build_actor_critic, build_backbone
from maniskill2_learn.methods.builder import build_agent
from maniskill2_learn.utils.torch import build_optimizer, no_grad, BaseAgent, hard_update, soft_update
from maniskill2_learn.utils.data import DictArray, GDict, to_np, to_torch
from ..builder import MFRL
from copy import deepcopy
from time import time
from collections import defaultdict, OrderedDict
from pprint import pprint
_to_np = lambda x: x.detach().cpu().numpy()
PRINT_DETAILED_TIME = False

@MFRL.register_module()
class HER_DREAMER_SIMPLE(BaseAgent):
    def __init__(
        self,
        orig_agent_cfg,
        explorer_agent_cfg,
        embedder_cfg,
        predictor_cfg,
        env_params,
        imagined_replay_cfg,
        n_procs_rollout=1,
        goal_eval_explor_factor=1,
        goal_eval_extr_factor=1,
        new_reward_dist_factor=1,
        new_reward_extr_factor=1,
        orig_buffer_updates=10,
        imagine_buffer_updates=10,
        pe_updates=3,
        batch_size_pe=200,
        explorer_updates=3,
        expl_prob_decay = 0.9995,
        expl_min_prob=0.1,
    ):
        super(HER_DREAMER_SIMPLE, self).__init__()
        ################################################
        ## OWN PARAMETERS (NOT SAC)
        ################################################
        self.num_episodes = 0
        self.n_procs_rollout = n_procs_rollout
        self.imagined_replay = build_replay(imagined_replay_cfg)
        self.goal_eval_explor_factor = goal_eval_explor_factor
        self.goal_eval_extr_factor = goal_eval_extr_factor
        self.new_reward_dist_factor = new_reward_dist_factor
        self.new_reward_extr_factor = new_reward_extr_factor
        self.orig_buffer_updates = orig_buffer_updates
        self.imagine_buffer_updates = imagine_buffer_updates

        self.batch_size_pe = batch_size_pe
        self.pe_updates = pe_updates
        self.pe_warm_up_updates=40
        self.explorer_warm_up_updates=40
        # Keys of env_params dict: obs_shape, action_shape, action_space, is_discrete
        #### DREAMER V2
        # To lukas: Write here the init code for dreamer.
        # -> All functions that need to be implemented using some functionality from dreamer
        #    are labeled with "TODO"
        embedder_cfg = deepcopy(embedder_cfg)
        predictor_cfg = deepcopy(predictor_cfg)
        predictor_cfg["feat_size"] = embedder_cfg.feat_size
        self.embedder = build_backbone(embedder_cfg)
        self.predictor = build_backbone(predictor_cfg)
        
        orig_agent_cfg = deepcopy(orig_agent_cfg)
        explorer_agent_cfg = deepcopy(explorer_agent_cfg)
        orig_agent_cfg['env_params'] = deepcopy(env_params)
        explorer_agent_cfg['env_params'] = deepcopy(env_params)
        self.orig_agent = build_agent(orig_agent_cfg)
        self.explorer_agent = build_agent(explorer_agent_cfg)
        self.explorer_updates = explorer_updates
        self.time_measures = OrderedDict()
        self.c_time = time()

        self.actor = self.orig_agent
        self.expl_prob_decay = expl_prob_decay
        self.expl_min_prob=expl_min_prob
        self.expl_prob = 1

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


    def update_parameters(self, memory, updates, rollout=None, *args, **kwargs):
        # return dictionary with information
        warm_up_training = kwargs['warm_up_training'] if "warm_up_training" in kwargs else False

        self.measure_time("start", reset=True)
        # 1. Train Embedder and Predictor with memory
        for i in range(self.pe_updates):
            sampled_batch = memory.sample(self.batch_size_pe).to_torch(device=self.device, non_blocking=True)
            ret1_, cur_emb  = self.embedder.update_parameters(sampled_batch)
            next_emb = self.get_embedding(sampled_batch['next_obs'])

            ret2_ = self.predictor.update_parameters(sampled_batch, cur_emb, next_emb)
            # store loss outputs
            self.join_key_returns(ret1_, "embed")
            self.join_key_returns(ret2_, "pred")
            self.measure_time("update pi")
        
        self.measure_time("update pe finished")
        # 2. Slice recent observation into trajectories

        if not warm_up_training:
            recent_replay = kwargs["recent_traj_replay"] 
            recent_memory = recent_replay.get_all()

            start_indices, end_indices = self.get_start_end_indices(recent_memory)
            self.measure_time("get_start_end")

            trajectories = []
            for start_idx, end_idx in zip(start_indices, end_indices):
                trajectory_obs = self.slice_dictionary_copy(recent_memory, start_idx, end_idx)
                trajectories.append(trajectory_obs)
                self.measure_time("slice_trajectory")

            # 3. evaluate trajectory goals and decide for new goal positions
            goal_eval = self.evaluate_rollout_for_goals(recent_memory)
            self.measure_time("goal_eval")
            new_goals = []
            for start_idx, end_idx, trajectory in zip(start_indices, end_indices, trajectories):
                max_idx = np.argmax(goal_eval[start_idx:end_idx])
                if self.is_goal_good_enough(goal_eval[max_idx], goal_eval[start_idx:end_idx]):
                    new_goals.append((start_idx, end_idx, max_idx, trajectory))
            self.measure_time("goal_eppend")
            # 4. update the rewards using the new found goals
            for start_idx, end_idx, max_idx, trajectory in new_goals:
                goal_embedding = self.get_embedding(recent_memory.slice(max_idx)['obs'])
                distance = self.get_distance_in_embedding(recent_memory, start_idx, end_idx, goal_embedding)

                new_rewards = (
                    np.expand_dims(-self.new_reward_dist_factor * _to_np(distance), 1) + self.new_reward_extr_factor * trajectory_obs["rewards"]
                )
                # 5. generate new trajectory with rewards and push to buffer
                trajectory["rewards"] = new_rewards
                self.imagined_replay.push_batch(trajectory_obs)
                recent_replay['rewards'][start_idx:end_idx] = new_rewards
                self.measure_time("push new rewards")



        if not warm_up_training:
            # 6. Update explorer
            for _ in range(self.explorer_updates):
                ret_ = self.explorer_agent.update_parameters(recent_replay, updates=updates)
                self.join_key_returns(ret_, "explor")            
                self.measure_time("explor")
            # 7.learn SAC with both original buffer and imagine_buffer
            for _ in range(self.orig_buffer_updates):
                # update parameters
                ret_ = self.orig_agent.update_parameters(memory, updates=updates)
                # put information into return dictionary
                self.join_key_returns(ret_, "orig", also_normal=True)
                self.measure_time("orig")
            for _ in range(self.imagine_buffer_updates):
                ret_ = self.orig_agent.update_parameters(self.imagined_replay, updates=updates)
                self.join_key_returns(ret_, "imag", also_normal=True)
                self.measure_time("imagine")
                # put information into return dictionary
            self.update_explorer_percentage()
        # average all values in ret:

        self.measure_time("finished")
        self.print_time()
        return self.return_ret_dict()



    @no_grad
    def forward(self, obs, **kwargs):
        obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs) if self._in_test else self.obs_rms.add(obs)
        if self.obs_processor is not None:
            obs = self.obs_processor({"obs": obs})["obs"]

        # choose actor and dreamer at random
        random_value = np.random.rand()
        
        # did episode reset last round?
        if random_value < self.expl_prob:
            out = self.explorer_agent(obs)
        else:
            # maybe not neccessrary?
            rgb_image = obs.pop("rgb_image")  # <-
            depth = obs.pop("depth")  # <-
            out = self.orig_agent(obs)
            obs["rgb_image"] = rgb_image
            obs["depth"] = depth



        return out

    def update_explorer_percentage(self):
        """Returns a value between 0 and 1. The value corresponds to the percentage of actions that should
        be generated with dreamer. Should depend on num_episodes
        """
        self.expl_prob *= self.expl_prob_decay
        self.expl_prob = max(self.expl_prob, self.expl_min_prob)


        

    def evaluate_uncertainty(self, memory):
        """This function should return how unknown a position is for dreamer given obs array"""
        # TODO (Lukas): Both on batches shape = (B,X) as well as just shape = (X)
        cur_emb = self.get_embedding(memory['obs'])
        next_emb = self.get_embedding(memory['next_obs'])
        return self.predictor.uncertainty(cur_emb, next_emb, memory['actions'])


    def get_embedding(self, obs):
        """This function should output an embedding from the obs space into the hidden state."""
        # TODO (Lukas): Both on batches shape = (B,X) as well as just shape = (X)
        return self.embedder.embed(obs)

    def evaluate_rollout_for_goals(self, replay):
        """Loops over all states and measures their effectivness as goals for their trajectory.
        Returns an array of those values with an entry for each obs in replay
        """
        exploration = self.evaluate_uncertainty(replay)
        goal_evaluation = self.goal_eval_extr_factor * np.copy(replay["rewards"])[:,0]  # adjust rewards
        goal_evaluation = goal_evaluation + self.goal_eval_explor_factor * exploration  # add uncertainty
        return goal_evaluation

    def get_distance_in_embedding(self, obs, start_idx, end_ids, emb):
        """Calculates the distance of the embedding of obs to the existing embedding"""
        # TODO (Lukas): Maybe it is better to further invoke the dynamics model to obtain the latent state distribution.
        # self.dreamer_agent._wm.dynamics.obs_step(...)
        embedding = self.get_embedding(obs['obs'])
        distance = torch.linalg.vector_norm(embedding - emb, 1, dim=1)  # use l1 norm as distance
        return distance

    def is_goal_good_enough(self, goal_value, trajectory):
        """Function that decides if a goal is good enough :) Needs to be improved"""
        return goal_value.item() > 0

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
