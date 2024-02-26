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
from maniskill2_learn.networks import build_model, build_actor_critic, utils_dreamer
from maniskill2_learn.utils.torch import build_optimizer, no_grad, BaseAgent, hard_update, soft_update
from maniskill2_learn.utils.data import DictArray, GDict, to_np, to_torch
from ..builder import MFRL
from copy import deepcopy
from time import time
from collections import defaultdict, OrderedDict
from pprint import pprint

_to_np = lambda x: x.detach().cpu().numpy()

@MFRL.register_module()
class HER_DREAMER(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        dreamer_cfg,
        env_params,
        imagined_replay_cfg,
        n_updates,
        batch_size=128,
        gamma=0.99,
        update_coeff=0.005,
        alpha=0.2,
        ignore_dones=True,
        target_update_interval=1,
        automatic_alpha_tuning=True,
        target_smooth=0.90,  # For discrete SAC
        alpha_optim_cfg=None,
        target_entropy=None,
        shared_backbone=False,
        detach_actor_feature=False,
        n_procs_rollout=1,
        goal_eval_explor_factor=1.0,
        goal_eval_extr_factor=1.0,
        goal_eval_extr_scale_init = 1.0,
        goal_eval_extr_scale_target = 4.0,
        goal_eval_extr_scale_adjustment = 1.5,

        # Coefficients for new reward calculation
        # -- Coefficient for entire reward term to make it comparable to actual reward term. 
        new_reward_scale_init=1.0, 
        new_reward_scale_target = 1.0, 
        new_reward_scale_adjustment = 1.25,

        # -- The uncertainty term has no coefficient and the other coefficients are dynamically adjusted to meet the target ratios specified below.
        # -- Coefficient for the extrinisic task-specific reward.
        new_reward_extr_scale_init=1.0, 
        new_reward_extr_scale_target = 4.0, 
        new_reward_extr_scale_adjustment = 1.5,

        # -- Coefficient for the distance to the imaginary goal in the observation embedding of the Dreamer World Model.
        new_reward_dist_emb_scale_init=1.0,  
        new_reward_dist_emb_scale_target=0.3, # NOTE (Lukas): Influence ratio of embedding distance over exploration rewards on the new goal rewards.
        new_reward_dist_emb_scale_adjustment=1.5,
        
        # -- Coefficient for the absolute step distance to the imaginary goal.
        new_reward_dist_step_scale_init=1.0,  
        new_reward_dist_step_scale_target=0.3, # NOTE (Lukas): Influence ratio of step distance over exploration rewards on the new goal rewards.
        new_reward_dist_step_scale_adjustment=1.5,
        orig_buffer_updates=10,
        imagine_buffer_updates=10,
        expl_decay = 0.9995, # TODO (Lukas): Implement 1/x with linear increasing x idea.
        min_expl = 0.05,
        init_expl = 0.95,
        expl_decay_steps = 1000,
        reward_memory_decay = 0.99,
        goal_quality_extr_bound = 32.0, # NOTE (Lukas): How much worse than target extrinsic reward may goal trajectory be?
        goal_quality_expl_bound = 8.0, # NOTE (Lukas): How much worse than target exploration reward may goal trajectory be?
        goal_quality_rel_bound = 1.15, # NOTE (Lukas): How much smaller must the mean reward of the trajectory be compard to the goal state reward?
        goal_quality_bound_adjustment = 0.999,
        goal_eval_offset = 10,
    ):
        super(HER_DREAMER, self).__init__()
        ################################################
        ## OWN PARAMETERS (NOT SAC)
        ################################################
        self.num_episodes = 0
        self.n_procs_rollout = n_procs_rollout
        self.imagined_replay = build_replay(imagined_replay_cfg)
        self.n_updates = n_updates

        # Goal evaluation
        self.goal_eval_explor_factor = goal_eval_explor_factor
        self.goal_eval_extr_factor = goal_eval_extr_factor
        self.goal_eval_extr_scale = goal_eval_extr_scale_init
        self.goal_eval_extr_scale_target = goal_eval_extr_scale_target
        self.goal_eval_extr_scale_adjustment = goal_eval_extr_scale_adjustment
        # NOTE (Lukas): Initial state is purely 0 in latent state.
        # Predictor ensemble disagreement will always be high in these states. Ignore them.
        self.goal_eval_offset = goal_eval_offset
        
        # Goal quality constraints
        self.goal_quality_extr_bound = goal_quality_extr_bound
        self.goal_quality_expl_bound = goal_quality_expl_bound
        self.goal_quality_rel_bound = goal_quality_rel_bound
        self.goal_quality_bound_adjustment = goal_quality_bound_adjustment

        # Reward calculation for new goals
        self.new_reward_scale = new_reward_scale_init
        self.new_reward_scale_target = new_reward_scale_target
        self.new_reward_scale_adjustment = new_reward_scale_adjustment

        self.new_reward_dist_emb_scale = new_reward_dist_emb_scale_init
        self.new_reward_dist_emb_scale_target = new_reward_dist_emb_scale_target
        self.new_reward_dist_emb_scale_adjustment = new_reward_dist_emb_scale_adjustment

        self.new_reward_dist_step_scale = new_reward_dist_step_scale_init
        self.new_reward_dist_step_scale_target = new_reward_dist_step_scale_target
        self.new_reward_dist_step_scale_adjustment = new_reward_dist_step_scale_adjustment

        self.new_reward_extr_scale = new_reward_extr_scale_init
        self.new_reward_extr_scale_target = new_reward_extr_scale_target
        self.new_reward_extr_scale_adjustment = new_reward_extr_scale_adjustment

        self.orig_buffer_updates = orig_buffer_updates
        self.imagine_buffer_updates = imagine_buffer_updates
        
        # Exploration rate
        self.init_expl = init_expl
        self.expl = init_expl
        self.expl_decay = expl_decay
        self.min_expl = min_expl
        self.expl_decay_steps = expl_decay_steps
        self.step_count = 1.0 # Incremented by 1.0/expl_decay_steps in each update_parameter call.
        
        # Reward memory
        # NOTE (Lukas): Memorizes the mean reward with time decay.
        self.reward_memory_decay = reward_memory_decay
        self.mean_reward_memory = 0.0 


        # NOTE (Lukas): To avoid recomputations of the same values, we will track the n_updates iteration
        self.update_clock = 0

        # Keys of env_params dict: obs_shape, action_shape, action_space, is_discrete
        #### DREAMER V2
        # To lukas: Write here the init code for dreamer.
        # -> All functions that need to be implemented using some functionality from dreamer
        #    are labeled with "TODO"
        dreamer_cfg = deepcopy(dreamer_cfg)
        dreamer_cfg['env_params'] = env_params
        self.dreamer_agent = build_model(dreamer_cfg)
        
        self.PRINT_DETAILED_TIME = False

        ################################################
        ## SAC CODE COPIED FROM HERE ON IN INIT
        ################################################
        actor_cfg = deepcopy(actor_cfg)
        critic_cfg = deepcopy(critic_cfg)

        actor_optim_cfg = actor_cfg.pop("optim_cfg")
        critic_optim_cfg = critic_cfg.pop("optim_cfg")
        action_shape = env_params["action_shape"]
        self.is_discrete = env_params["is_discrete"]

        self.gamma = gamma
        self.update_coeff = update_coeff
        self.alpha = alpha
        self.ignore_dones = ignore_dones
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.automatic_alpha_tuning = automatic_alpha_tuning

        actor_cfg.update(env_params)
        critic_cfg.update(env_params)

        self.actor, self.critic = build_actor_critic(actor_cfg, critic_cfg, shared_backbone)
        self.shared_backbone = shared_backbone
        self.detach_actor_feature = detach_actor_feature

        self.actor_optim = build_optimizer(self.actor, actor_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, critic_optim_cfg)

        self.target_critic = build_model(critic_cfg)
        hard_update(self.target_critic, self.critic)

        self.log_alpha = nn.Parameter(torch.ones(1, requires_grad=True) * np.log(alpha))
        if target_entropy is None:
            if env_params["is_discrete"]:
                # Use label smoothing to get the target entropy.
                n = np.prod(action_shape)
                explore_rate = (1 - target_smooth) / (n - 1)
                self.target_entropy = -(
                    target_smooth * np.log(target_smooth) + (n - 1) * explore_rate * np.log(explore_rate)
                )
                self.log_alpha = nn.Parameter(torch.tensor(np.log(0.1), requires_grad=True))
                # self.target_entropy = np.log(action_shape) * target_smooth
            else:
                self.target_entropy = -np.prod(action_shape)
        else:
            self.target_entropy = target_entropy
        if self.automatic_alpha_tuning:
            self.alpha = self.log_alpha.exp().item()

        self.alpha_optim = build_optimizer(self.log_alpha, alpha_optim_cfg)

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
        # Get used observations
        recent_memory = kwargs["recent_traj_replay"].get_all()

        start_indices, end_indices = self.get_start_end_indices(recent_memory)

        mean_reward = np.mean(recent_memory['rewards'])
        self.mean_reward_memory = self.reward_memory_decay * self.mean_reward_memory + (1-self.reward_memory_decay) * mean_reward

        # The following operations have only to be computed for the entire update loop (self.n_update steps).    
        if self.update_clock == 0:
            # Alternative: Exponential decay exploration (might be too strong in the long-run): 
            # self.expl = max(self.expl*self.expl_decay, self.min_expl) 
            # TODO (Lukas): Finetune exploration decay between exponential and 1/x decay.
            # NOTE (Lukas): Step count should be invariant of updates call, so I put it into the conition too.
            self.step_count += 1/(self.expl_decay_steps)
            self.expl = max(self.init_expl/self.step_count, self.min_expl) 

            # 1. Find start and end indices of trajectories in recent_obs
            self.trajectories = []
            for start_idx, end_idx in zip(start_indices, end_indices):
                trajectory = self.slice_dictionary_copy(recent_memory, start_idx, end_idx)
            self.trajectories.append(trajectory)

            # 2. Compute new imaginary goals.
            accepted_imag_goals = 0
            for trajectory in self.trajectories:
                # 3. Evaluate goal quality of trajectory.
                goal_eval = self.evaluate_rollout_for_goals(trajectory)
                # NOTE (Lukas): Remove second dimension and truncate at the beginning.
                goal_eval = np.squeeze(goal_eval)[self.goal_eval_offset:]
                trajectory_trunc = self.slice_dictionary_copy(trajectory, self.goal_eval_offset, len(trajectory['rewards']))

                # 4. Find the best goal state based on the goal quality evaluation.
                max_idx = np.argmax(goal_eval, axis=0)
                self.join_key_returns(dict(max_idx=max_idx), 'goal_eval')
                if not self.is_goal_good_enough(goal_eval, max_idx, trajectory_trunc):
                    continue
                accepted_imag_goals += 1
                trajectory_trunc = self.slice_dictionary_copy(trajectory_trunc, 0, max_idx + 1)
                
                # 5. Calculate the reward for the new imaginary goal trajectories.
                # Compute the new rewards as a linear combination of embedding distance, step distance, uncertainty, and extrinsic reward.

                # Extract the exploration rewards from the goal quality score.
                expl_rewards = np.expand_dims(goal_eval[:max_idx+1],1) - self.goal_eval_extr_scale * trajectory_trunc['rewards']
                expl_rewards_mean = np.mean(expl_rewards)
                extr_rewards_mean = np.mean(trajectory_trunc['rewards'])
                # Dynamically adjust the extrinsic reward scale if it deviates too strongly from the desired target interval.
                self.new_reward_extr_scale = utils_dreamer.adjust_scale(
                    self.new_reward_extr_scale,
                    self.new_reward_extr_scale_target,
                    extr_rewards_mean,
                    expl_rewards_mean,
                    self.new_reward_extr_scale_adjustment,
                    low_tol=0.25,
                    high_tol=4.0
                )

                # For each state in the trajectory, compute the improvement in the embedding distance to the goal state. 
                goal_embedding = self.get_image_embedding(trajectory_trunc['obs']['rgb_image'][max_idx])
                emb_distance = self.get_distance_in_embedding(trajectory_trunc['obs']['rgb_image'], goal_embedding)
                emb_distance[1:] -= emb_distance[:-1].clone()
                emb_distance[0] = emb_distance[1]
                # So far, we have negative values for improvement, inverse them. 
                emb_distance.mul_(-1.0)
                # emb_distance is yet torch Tensor, convert to np now.
                # Add a second dimension to fit the shape of the trajectories
                emb_distance = np.expand_dims(_to_np(emb_distance),1)
                emb_distance_mean = np.mean(emb_distance)

                # Dynamically adjust the embedding distance scale if it deviates too strongly from the desired target interval.
                self.new_reward_dist_emb_scale = utils_dreamer.adjust_scale(
                    self.new_reward_dist_emb_scale,
                    self.new_reward_dist_emb_scale_target,
                    emb_distance_mean,
                    expl_rewards_mean,
                    self.new_reward_dist_emb_scale_adjustment,
                    low_tol=0.125,
                    high_tol=8.0
                )

                # Further add a reward that reflects the absolute step distance to the goal.
                # Add a second dimension to fit the shape of the trajectories
                step_distance = np.expand_dims(np.linspace(0.0,1.0,max_idx+1), 1)
                step_distance_mean = np.mean(step_distance) # Around 0.5
                # Dynamically adjust the step distance scale if it deviates too strongly from the desired target interval.
                self.new_reward_dist_step_scale = utils_dreamer.adjust_scale(
                    self.new_reward_dist_step_scale,
                    self.new_reward_dist_step_scale_target,
                    step_distance_mean,
                    expl_rewards_mean,
                    self.new_reward_dist_step_scale_adjustment,
                    low_tol=0.125,
                    high_tol=8.0
                )
                
                new_rewards = expl_rewards
                new_rewards += self.new_reward_dist_emb_scale * emb_distance
                new_rewards += self.new_reward_dist_step_scale * step_distance
                new_rewards += self.new_reward_extr_scale * trajectory_trunc['rewards']

                # NOTE (Lukas): Formulate the ratios relative to the extr rewards, no the expl rewards. 
                # This simplifies finetuning the relative ratio of the imag rewards to the true external rewards.
                # Finally, ensure that the rewards for the new goals are coarsely of similar magnitude as the actual rewards.
                new_rewards /= self.new_reward_extr_scale
                
                new_rewards_mean = np.mean(new_rewards)
                self.new_reward_scale = utils_dreamer.adjust_scale(
                    self.new_reward_scale,
                    self.new_reward_scale_target,
                    new_rewards_mean,
                    extr_rewards_mean,
                    self.new_reward_scale_adjustment,
                    low_tol=0.125,
                    high_tol=8.0
                )
                # TODO (Lukas): Investigate how rewards develop.
                # Targeting self.new_reward_scale might throttle the benefit of non-external rewards,
                # because we punish the new rewards more the lower the external reward is.
                # I am uncertain whether this is useful, so I first replace the scale by a constant 2.0.
                # If the above scale targets are met, then the imag reward should be 2.6 times larger than the orig reward.
                # We hence bias slightly towards the artificial rewards to lore the agent to these challenges.
                trajectory_trunc["rewards"] = 0.5 * new_rewards 
                # trajectory["rewards"] = self.new_reward_scale * new_rewards 

                new_rewards_mean = np.mean(trajectory_trunc['rewards'])
                self.imagined_replay.push_batch(trajectory_trunc)

                new_rewards_mean_ret = dict(
                    expl = expl_rewards_mean,
                    dist_emb = emb_distance_mean,
                    dist_step = step_distance_mean,
                    extr = extr_rewards_mean,
                    total_scale = self.new_reward_scale,
                    new_rewards_mean=new_rewards_mean,
                )
                self.join_key_returns(new_rewards_mean_ret, 'new_rewards_mean')

                
            coeffs_ret = dict(
                new_reward_dist_emb_scale=self.new_reward_dist_emb_scale,
                new_reward_dist_step_scale=self.new_reward_dist_step_scale,
                new_reward_extr_scale=self.new_reward_extr_scale,
                new_reward_scale=self.new_reward_scale,
                goal_eval_extr_scale=self.goal_eval_extr_scale,
                mean_reward_memory=self.mean_reward_memory,
                expl_ratio=self.expl,
                goal_quality_extr_bound = self.goal_quality_extr_bound,
                goal_quality_expl_bound = self.goal_quality_expl_bound,
                goal_quality_rel_bound = self.goal_quality_rel_bound,
            )
            self.join_key_returns(coeffs_ret, 'coeffs')
                
            self.join_key_returns(dict(accepted_imag_goals=accepted_imag_goals), 'replay_imag')
        
        # 6. Train Dreamer with off-policy memory.
        for trajectory in self.trajectories:
            ret_ = self.train_dreamer(trajectory)
            self.join_key_returns(ret_, None)

        # 7. Train SAC with both original buffer and imagine_buffer.
        # NOTE (Lukas): Randomly alternating the order of orig and imag buffer updates works.
        total_updates = self.orig_buffer_updates + self.imagine_buffer_updates
        imag_ratio = self.imagine_buffer_updates / total_updates

        # NOTE (Lukas): Edge case: All imag goals rejected, empty imag buffer.
        if len(self.imagined_replay.sampling) == 0:
            imag_ratio = 0.0
            total_updates = self.orig_buffer_updates

        for _ in range(total_updates):
            random_value = np.random.rand()
            if random_value < imag_ratio:
                ret_ = self._update_parameters(self.imagined_replay, updates=updates)
                # put information into return dictionary
                self.join_key_returns(ret_, 'imag')
            else:
                ret_ = self._update_parameters(memory, updates=updates)
                # put information into return dictionary
                self.join_key_returns(ret_, 'orig')
            # self.measure_time('update')

        self.update_clock = (self.update_clock + 1) % self.n_updates

        # self.print_time()
        return self.return_ret_dict()

    def _update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size).to_torch(device=self.device, non_blocking=True)
        sampled_batch = self.process_obs(sampled_batch)
        with torch.no_grad():
            if self.is_discrete:
                _, next_action_prob, _, _, dist_next = self.actor(sampled_batch["next_obs"], mode="all_discrete")
                q_next_target = self.target_critic(sampled_batch["next_obs"], actions_prob=next_action_prob)
                min_q_next_target = (
                    torch.min(q_next_target, dim=-1, keepdim=True).values + self.alpha * dist_next.entropy()[..., None]
                )
            else:
                next_action, next_log_prob = self.actor(sampled_batch["next_obs"], mode="all")[:2]
                q_next_target = self.target_critic(sampled_batch["next_obs"], next_action)
                min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            if self.ignore_dones:
                q_target = sampled_batch["rewards"] + self.gamma * min_q_next_target
            else:
                q_target = (
                    sampled_batch["rewards"] + (1 - sampled_batch["dones"].float()) * self.gamma * min_q_next_target
                )
            q_target = q_target.repeat(1, q_next_target.shape[-1])
        q = self.critic(sampled_batch["obs"], sampled_batch["actions"])

        critic_loss = F.mse_loss(q, q_target) * q.shape[-1]
        with torch.no_grad():
            abs_critic_error = torch.abs(q - q_target).max().item()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        with torch.no_grad():
            critic_grad = self.critic.grad_norm
        if self.shared_backbone:
            self.critic_optim.zero_grad()

        if self.is_discrete:
            _, pi, _, _, dist = self.actor(
                sampled_batch["obs"],
                mode="all_discrete",
                save_feature=self.shared_backbone,
                detach_visual=self.detach_actor_feature,
            )
            entropy_term = dist.entropy().mean()
        else:
            pi, log_pi = self.actor(
                sampled_batch["obs"],
                mode="all",
                save_feature=self.shared_backbone,
                detach_visual=self.detach_actor_feature,
            )[:2]
            entropy_term = -log_pi.mean()

        visual_feature = self.actor.backbone.pop_attr("saved_visual_feature")
        if visual_feature is not None:
            visual_feature = visual_feature.detach()

        if self.is_discrete:
            q = self.critic(sampled_batch["obs"], visual_feature=visual_feature, detach_value=True).min(-2).values
            q_pi = (q * pi).sum(-1)
            with torch.no_grad():
                q_match_rate = (pi.argmax(-1) == q.argmax(-1)).float().mean().item()
        else:
            q_pi = self.critic(sampled_batch["obs"], pi, visual_feature=visual_feature)
            q_pi = torch.min(q_pi, dim=-1, keepdim=True).values
        actor_loss = -(q_pi.mean() + self.alpha * entropy_term)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        with torch.no_grad():
            actor_grad = self.actor.grad_norm

        if self.automatic_alpha_tuning:
            alpha_loss = self.log_alpha.exp() * (entropy_term - self.target_entropy).detach()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
        if updates % self.target_update_interval == 0:
            soft_update(self.target_critic, self.critic, self.update_coeff)

        ret = {
            "sac/critic_loss": critic_loss.item(),
            "sac/max_critic_abs_err": abs_critic_error,
            "sac/actor_loss": actor_loss.item(),
            "sac/alpha": self.alpha,
            "sac/alpha_loss": alpha_loss.item(),
            "sac/q": torch.min(q, dim=-1).values.mean().item(),
            "sac/q_target": torch.mean(q_target).item(),
            "sac/entropy": entropy_term.item(),
            "sac/target_entropy": self.target_entropy,
            "sac/critic_grad": critic_grad,
            "sac/actor_grad": actor_grad,
        }
        if self.is_discrete:
            ret["sac/q_match_rate"] = q_match_rate

        return ret

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
        done = kwargs['dones']
        done = np.squeeze(done)

        # NOTE (Lukas): We should only use dreamer in training, not in evaluation.
        if self.training and random_value < self.get_explorer_percentage():
            # TODO (Lukas): Adjust to real function head syntax
            out = self.dreamer_agent(obs)
        else:
            # maybe not neccessrary?
            rgb_image = obs.pop("rgb_image")  # <-
            depth = obs.pop("depth")  # <-
            out = self.actor(obs)
            obs["rgb_image"] = rgb_image
            obs["depth"] = depth

            # because dreamer was not run update its state nonehtheless if dreamer needs to be run in the next rund
            self.dreamer_agent.update_state(obs, out, done)

        return out

    def eval_act(self, obs):
        obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
        return self.actor(obs)

    def get_explorer_percentage(self):
        """Returns a value between 0 and 1. The value corresponds to the percentage of actions that should
        be generated with dreamer. Should depend on num_episodes
        """
        # TODO (Lukas): Start with high value and apply a decay of 0.9995 (worked well in other tasks).
        return self.expl  

    def train_dreamer(self, trajectory):
        # TODO (Lukas): Implement the dreamer function with rollout
        return self.dreamer_agent._train(trajectory)

    def evaluate_uncertainty(self, trajectory) -> torch.Tensor:
        """This function should return how unknown a position is for dreamer given obs array"""
        # TODO (Lukas): Both on batches shape = (B,X) as well as just shape = (X)
        return self.dreamer_agent.evaluate_uncertainty(trajectory)

    def get_image_embedding(self, rgb_image):
        """This function should output an embedding from the obs space into the hidden state."""
        # TODO (Lukas): Both on batches shape = (B,X) as well as just shape = (X)
        return self.dreamer_agent.get_embedding(rgb_image)
    
    def evaluate_rollout_for_goals(self, trajectory):
        """Loops over all states and measures their effectiveness as goals for their trajectory.
        Returns an array of those values with an entry for each observation in the trajectory.
        """
        expl_rewards = self.evaluate_uncertainty(trajectory)
        expl_rewards_mean = expl_rewards.mean()
        extr_rewards = np.copy(trajectory["rewards"])
        extr_rewards_mean = np.mean(extr_rewards)

        # NOTE (Lukas): The extrinsic rewards impact on the goal evaluation should be goal_eval_extr_scale_target times larger
        # than the impact of the exploration uncertainty. 
        # We allow deviations up to certain factors and dynamically adjust the scale if the deviation is too strong.  
        self.goal_eval_extr_scale = utils_dreamer.adjust_scale(
            self.goal_eval_extr_scale,
            self.goal_eval_extr_scale_target,
            extr_rewards_mean,
            expl_rewards_mean,
            self.goal_eval_extr_scale_adjustment,
            low_tol=0.125,
            high_tol=8.0
            )
        
        goal_evaluation = _to_np(expl_rewards) + self.goal_eval_extr_scale * extr_rewards
        # NOTE (Lukas): Old static scaling version.
        # goal_evaluation = self.goal_eval_extr_factor * np.copy(trajectory["rewards"])  # adjust rewards
        # goal_evaluation = goal_evaluation + self.goal_eval_explor_factor * _to_np(exploration)  # add uncertainty
        return goal_evaluation

    def get_distance_in_embedding(self, rgb_image, emb) -> torch.Tensor:
        """Calculates the distance of the embedding of obs to the existing embedding"""
        # TODO (Lukas): Maybe it is better to further invoke the dynamics model to obtain the latent state distribution.
        # self.dreamer_agent._wm.dynamics.obs_step(...)
        embedding = self.get_image_embedding(rgb_image)
        distance = torch.linalg.vector_norm(embedding - emb, 1, dim=1)  # use l1 norm as distance
        return distance

    def is_goal_good_enough(self, goal_eval, max_idx, trajectory) -> bool:
        """Function that decides if a goal is good enough."""
        goal_value = goal_eval[max_idx]
        goal_value_mean = np.mean(goal_eval)

        extr_rewards = np.copy(trajectory['rewards'])
        # TODO (Lukas): Ensure same dimensions of goal_eval
        extr_rewards_mean = np.mean(extr_rewards)

        # Extract the exploration rewards.
        expl_rewards = np.expand_dims(goal_eval,1) - self.goal_eval_extr_scale * extr_rewards
        expl_rewards_mean = np.mean(expl_rewards)

        # NOTE (Lukas): We adjust all goal quality constraints in the following intuitive fashion:
        # The more goals you accept, the stricter the contraints get.
        # The more goals you recept, the looser the constraints get.
        
        if goal_value.item() < 0.0:
            # NOTE (Lukas): If this happens, then the extrinsic rewards most be negative throughout. 
            # The uncertainty scores that form the exploration reward should always be positive.
            return False
        
        # 1. Ensure that we actually have at least one transition.
        if max_idx == 0:
            return False

        accept = True
        accept_score = 0
        # 2. Ensure absolute exploration benefit.
        # If exploration benefit is neglibigly small compared to extrinsic rewards, reject.
        if self.goal_quality_expl_bound * self.goal_eval_extr_scale_target * expl_rewards_mean < self.goal_eval_extr_scale * extr_rewards_mean:
            self.goal_quality_expl_bound /= self.goal_quality_bound_adjustment
            accept = False
            accept_score -= 1
            # NOTE (Lukas): Experienced this the first time on train iteration 185, bound was at 7.5
        else:
            self.goal_quality_expl_bound *= self.goal_quality_bound_adjustment
            accept_score += 1
        
        # 3. Ensure absolute extrinsic benefit
        # If extrinsic reward is neglibigly small compared to the time-decayed memorized mean reward, reject.
        if self.goal_quality_extr_bound * extr_rewards_mean < self.mean_reward_memory:
            self.goal_quality_extr_bound /= self.goal_quality_bound_adjustment
            accept = False
            accept_score -= 1
        else:
            self.goal_quality_extr_bound *= self.goal_quality_bound_adjustment
            accept_score += 1

        # 4. Ensure specific benefit of goal 
        # If goal state has no substantial gain in rewards, reject.
        if goal_value < self.goal_quality_rel_bound * goal_value_mean:
            self.goal_quality_rel_bound *= self.goal_quality_bound_adjustment
            # NOTE (Lukas): In one run, the first goal was rejected after 44, 47, 49 iterations.
            accept = False
            accept_score -= 1
        else:
            self.goal_quality_rel_bound /= self.goal_quality_bound_adjustment
            accept_score += 1

        return accept_score > 0

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
