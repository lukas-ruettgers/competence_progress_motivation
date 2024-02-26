import pathlib
import sys

from copy import deepcopy
from collections import defaultdict
sys.path.append(str(pathlib.Path(__file__).parent))

from .exploration import Random, Plan2Explore
from .dynamics_models import WorldModel, ImagBehavior
from maniskill2_learn.networks import utils_dreamer
from maniskill2_learn.utils.torch.module_utils import ExtendedModule

from ..builder import BACKBONES
import torch
from torch import distributions as torchd

to_np = lambda x: x.detach().cpu().numpy()


@BACKBONES.register_module()
class Dreamer(ExtendedModule):
    def __init__(
        self,
        nn_cfg,
        behaviour_cfg,
        expl_cfg,
        train_cfg,
        env_cfg,
        env_params,
    ):
        # Set new value: rollout_cfg["env_cfg"] = ...
        # Set existing value: rollout_cfg.env_cfg = ...
        # Get value: rollout_cfg.env_cfg
        super(Dreamer, self).__init__()

        self._nn_cfg = deepcopy(nn_cfg)
        self._behaviour_cfg = deepcopy(behaviour_cfg)
        self._expl_cfg = deepcopy(expl_cfg)
        self._train_cfg = deepcopy(train_cfg)
        self._env_cfg = deepcopy(env_cfg)
        self._env_cfg.num_actions = env_params.action_shape
        self._env_cfg.size = env_params.obs_shape.rgb_image[1:] # rgb_image=[C,H,W] 
        self._env_cfg['channels'] = env_params.obs_shape.rgb_image[0]
        self._env_params = env_params

        # NOTE (Lukas): In the calculation of uncertainty,
        # apply a dynamic scale to the KL value to satisfy the target scale up to a certain error.
        self.uncertainty_kl_scale_target = self._expl_cfg.uncertainty_kl_scale_target
        self.uncertainty_kl_scale_adjustment = self._expl_cfg.uncertainty_kl_scale_adjustment
        self.uncertainty_kl_scale = 1.0

        self._metrics = defaultdict(list)
        self._step = count_steps()

        # Schedules.
        self._behaviour_cfg.actor_entropy = lambda x=self._behaviour_cfg.actor_entropy: utils_dreamer.schedule(
            x, self._step
        )
        self._behaviour_cfg.actor_state_entropy = (
            lambda x=self._behaviour_cfg.actor_state_entropy: utils_dreamer.schedule(x, self._step)
        )
        self._behaviour_cfg.imag_gradient_mix = lambda x=self._behaviour_cfg.imag_gradient_mix: utils_dreamer.schedule(
            x, self._step
        )

        self._wm = WorldModel(self._step, self._nn_cfg, self._behaviour_cfg, self._train_cfg, self._env_cfg)
        self._task_behavior = ImagBehavior(
            self._nn_cfg,
            self._behaviour_cfg,
            self._train_cfg,
            self._env_cfg,
            self._wm,
            self._behaviour_cfg.behavior_stop_grad,
        )
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: Random(self._behaviour_cfg, self._env_cfg),
            plan2explore=lambda: Plan2Explore(
                self._nn_cfg,
                self._expl_cfg,
                self._behaviour_cfg,
                self._train_cfg,
                self._env_cfg,
                self._wm,
                reward,
            ),
        )[self._expl_cfg.expl_behavior]()

        self.reset_state()
        self.training = True

    def __call__(self, obs):
        if "rgb_image" in obs:
            obs = self.preprocess_for_dreamer(obs)

        policy_output = self._policy(obs)

        return policy_output

    def reset_state(self):
        # if the environment reset last round, the state does not make sense to keep => reset to None
        self.state = None

    def preprocess_for_dreamer(self, trajectory):
        # Convert to dict['action', 'image', 'reward']
        # When this method is called from _train(), trajectory has elements actions, rewards, obs
        obs = {}
        if "actions" in trajectory:
            obs["action"] = trajectory["actions"]

        if "rgb_image" in trajectory:
            # Preprocess is called from __call__ or update_state()
            obs["image"] = trajectory["rgb_image"]
        else:
            # Preprocess is called from _train()
            obs["image"] = trajectory["obs"]["rgb_image"]
        
        if "rewards" in trajectory:
            # Preprocess is called from _train()
            obs["reward"] = trajectory["rewards"]
        return obs

    def update_state(self, obs, action, done):
        if done:
            self.reset_state()
        # self.state is updated if the other actor is called
        if self.state is None:
            latent = self._wm.dynamics.initial(1)
            action = torch.zeros(self._env_cfg.num_actions).to(self.device) 
        else:
            latent, action = self.state
        # Observation --Encoder--> Observation Encoding
        if "rgb_image" in obs:
            obs = self.preprocess_for_dreamer(obs)
        # TODO: Test correct preprocessing
        embed = self._wm.encoder(self._wm.preprocess(obs))
        # Obtain the posterior latent state from the observation encoding
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, self._behaviour_cfg.collect_dyn_sample)
        self.state = (latent, action)
    
    def _policy(self, obs):
        """Make the next action based on the current observation and the previous latent state."""
        # Obtain current hidden state and action
        if self.state is None:
            latent = self._wm.dynamics.initial(1)
            action = torch.zeros(self._env_cfg.num_actions).to(self.device)  # to device ?

            # NOTE: action has shape (8), but latent elements have shape (1,50). 
            # Therefore, we add a first dimension to action for consistency (else we have an error).
            # embed also has shape (1, 1024), so I guess it makes most sense to adjust actions to it.
            action = action[None, :]
        else:
            latent, action = self.state

        # Observation --Encoder--> Observation Encoding
        embed = self._wm.encoder(self._wm.preprocess(obs))

        # Obtain the posterior latent state from the observation encoding
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, self._behaviour_cfg.collect_dyn_sample)
        if self._behaviour_cfg.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        
        # Decide between task-specific exploitation or pure exploration
        # NOTE: An actor corresponds to the action Head of the ImagBehaviour class
        if not self.training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        else:
            random_value = torch.rand(1)
            # NOTE (Lukas): Explore with probability ex.
            if random_value < self._expl_cfg.expl_ratio:
                actor = self._expl_behavior.actor(feat)
            else:
                actor = self._task_behavior.actor(feat)
            action = actor.sample()

        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        if self._behaviour_cfg.actor_dist == "onehot_gumble":
            action = torch.one_hot(torch.argmax(action, dim=-1), self._env_cfg.num_actions)
        # Extend deterministic action by distribution to account for exploration
        action = self._exploration(action)
        policy_output = {"action": action, "logprob": logprob}
        self.state = (latent, action)
        return action
    
    def _exploration(self, action):
        """Sample a random action around the deterministic action.

        Keyword Arguments:
          action -- deterministic action
          training -- Whether the model is in training or evaluation phase
        """
        amount = self._behaviour_cfg.expl_amount if self.training else self._env_cfg.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._behaviour_cfg.actor_dist:
            probs = amount / self._env_cfg.num_actions + (1 - amount) * action
            return utils_dreamer.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)

    def _train(self, data, preprocess=True):
        """Train Dreamer with a trajectory.

        Keyword arguments:
          data (dict['action', 'image', 'reward', 'discount']) -- Trajectory of environment observations with their preceding action
        """
        if preprocess:
            data = self.preprocess_for_dreamer(data)

        metrics = defaultdict(list)
        # Firstly, train the RSSM world model.
        # This includes the forward dynamics model, encoder, and the decoder/reward/discount heads.
        post, context, mets = self._wm._train(data)
        
        metrics.update({"wm/" + key: value for key, value in mets.items()})
        start = post

        # Secondly, train the task-specific policy-value network.
        if self._nn_cfg.pred_discount:  # Last step could be terminal.
            start = {k: v[:, :-1] for k, v in start.items()}
            context = {k: v[:, :-1] for k, v in context.items()}
        reward = lambda f, s, a: self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()
        mets = self._task_behavior._train(start, objective=reward)[-1]
        metrics.update({"behaviour/" + key : value for key, value in mets.items()})

        # Finally, train the exploration policy-value network.
        if self._expl_cfg.expl_behavior != "greedy":
            if self._nn_cfg.pred_discount:
                data = {k: v[:, :-1] for k, v in data.items()}
            mets = self._expl_behavior._train(start, context, data)[-1]
            metrics.update({"explor/" + key: value for key, value in mets.items()})
            
        for name, value in metrics.items():
            self._metrics[name].append(value)

        return metrics

    def wm_reward_prediction(self, f, s, a):
        feat = self._wm.dynamics.get_feat(s)
        return self._wm.heads['reward'](feat).mode()
    
    def get_embedding(self, obs):
        processed_obs = {}
        if "rgb_image" in obs:
            processed_obs["image"] = obs["rgb_image"]
        else:
            processed_obs["image"] = obs
        embed = self._wm.encoder(self._wm.preprocess(processed_obs))
        return embed

    def evaluate_uncertainty(self, trajectory):
        data = self.preprocess_for_dreamer(trajectory)
        feat, post, kl_value = self._wm.get_uncertainty_metrics(data)
        disag_reward = self._expl_behavior._intrinsic_reward(feat, post, torch.tensor(data['action']).to(self.device))
        
        # NOTE (Lukas): Dynamically update scale for kl_value
        # NOTE (Lukas): The kl values impact on the uncertainty should be uncertainty_kl_scale_target times larger
        # than the impact of the ensemble predictor disagreement. 
        # Note that uncertainty_kl_scale_target < 1.0.
        # We allow deviations up to certain factors and dynamically adjust the scale to it.  
        
        kl_value_mean = kl_value.mean()
        disag_reward_mean = disag_reward.mean()
        # TODO (Lukas): Check if the update scale of 2.0 is too sensitive.
        # TODO (Lukas): Since I do not know how well the KL values indicate uncertainty, I give the model more freedom to lower them.
        
        self.new_reward_dist_step_scale = utils_dreamer.adjust_scale(
            self.uncertainty_kl_scale,
            self.uncertainty_kl_scale_target,
            kl_value_mean,
            disag_reward_mean,
            self.uncertainty_kl_scale_adjustment,
            low_tol=0.5,
            high_tol=6.0
        )
        
        return disag_reward + self.uncertainty_kl_scale * kl_value.unsqueeze(-1)

# NOTE (Lukas): It won't harm if we just leave it this way.
def count_steps():
    return 0
