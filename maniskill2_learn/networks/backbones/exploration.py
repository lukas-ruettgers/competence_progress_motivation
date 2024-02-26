import torch
from torch import nn
from torch import distributions as torchd
# import tensorflow as tf
from maniskill2_learn.utils.meta import Registry, build_from_cfg

from . import dynamics_models
from .mlp import DenseHead
from maniskill2_learn.networks import utils_dreamer
from maniskill2_learn.utils.torch.module_utils import ExtendedModule

_to_np = lambda x: x.detach().cpu().numpy()

EXPLORATION = Registry("exploration")


@EXPLORATION.register_module()
class Random(ExtendedModule):
    def __init__(self, behaviour_cfg, env_cfg):
        super(Random, self).__init__()
        self._behaviour_cfg = behaviour_cfg
        self._env_cfg = env_cfg

    def actor(self, feat):
        shape = feat.shape[:-1] + [self._env_cfg.num_actions]
        if self._behaviour_cfg.actor_dist == "onehot":
            return utils_dreamer.OneHotDist(torch.zeros(shape))
        else:
            ones = torch.ones(shape)
            return utils_dreamer.ContDist(torchd.uniform.Uniform(-ones, ones))

    def _train(self, start, context):
        return None, {}


@EXPLORATION.register_module()
class Plan2Explore(ExtendedModule):
    def __init__(self, nn_cfg, expl_cfg, behaviour_cfg, train_cfg, env_cfg, world_model, reward=None):
        super(Plan2Explore, self).__init__()
        self._nn_cfg = nn_cfg
        self._expl_cfg = expl_cfg
        self._train_cfg = train_cfg
        self._env_cfg = env_cfg
        self._reward = reward
        self._behavior = dynamics_models.ImagBehavior(nn_cfg, behaviour_cfg, train_cfg, env_cfg, world_model)
        self.actor = self._behavior.actor
        stoch_size = self._nn_cfg.dyn_stoch
        self.disag_pre_exp_scale = self._expl_cfg.disag_pre_exp_scale
        if self._nn_cfg.dyn_discrete:
            stoch_size *= self._nn_cfg.dyn_discrete
        size = {
            "embed": 32 * self._nn_cfg.cnn_depth,
            "stoch": stoch_size,
            "deter": self._nn_cfg.dyn_deter,
            "feat": self._nn_cfg.dyn_stoch + self._nn_cfg.dyn_deter,
        }[self._expl_cfg.disag_target]
        # NOTE (Lukas): Defining embed as target makes it harder than latent (450 vs. 100 values), but embed may be closer to reality.
        # NOTE: It does not make sense to define the input dim as dyn_stoch, but later provide dyn_stoch+dyn_deter.
        kw = dict(
            inp_dim=self._nn_cfg.dyn_stoch + self._nn_cfg.dyn_deter,  # pytorch version
            # NOTE (Lukas): I extended inp_dim by the deterministic part of the hidden state to follow the paper.
            shape=size,
            layers=self._expl_cfg.disag_layers,
            units=self._expl_cfg.disag_units,
            act_cfg=self._nn_cfg.act,
            norm_cfg=self._nn_cfg.norm_mlp,
            bias=self._nn_cfg.bias,
        )
        # NOTE (Lukas): The author refrained from including the action into the predictor input.
        # The action is essential to predict the next state, so I added it again.
        if self._expl_cfg.disag_action_cond:
            kw['inp_dim'] += self._env_cfg.num_actions
        self._networks = nn.ModuleList([DenseHead(**kw) for _ in range(self._expl_cfg.disag_models)])
        
        # NOTE (Lukas): optimizer needs to be capital, changed to Optimizer.
        self._opt = utils_dreamer.Optimizer(
            self._train_cfg.opt,
            self.parameters(),
            self._train_cfg.model_lr,
            eps=self._train_cfg.opt_eps,
            clip=self._train_cfg.grad_clip,
            wd=self._nn_cfg.weight_decay,
        )

        # self._opt = utils_dreamer.Optimizer(
        #    'ensemble', self._train_cfg.model_lr, self._train_cfg.opt_eps, self._train_cfg.grad_clip,
        #    self._nn_cfg.weight_decay, opt=self._train_cfg.opt)

    def _preprocess(self, data):
        """Only data['action'] is required here."""
        data = {'action':torch.tensor(data['action'].copy()).to(self.device)}
        return data

    def _train(self, start, context, data):
        """Train the exploration policy on a trajectory along with the latent states provided by Dreamer.
        Parameters:
            start: posterior latent states
            context (dict['embed', 'feat', 'kl', 'postent']) --
                embed: feature embeddings of each observation, 
                feat: concatenated stochastic and deterministic part of the posterior latent state of each step,
                kl: KL values for each step,
                postent: entropy of posterior latent state for each step 
            data dict['action', 'image', 'reward']: preprocessed trajectory data 
        """
        data = self._preprocess(data)
        metrics = {}
        stoch = start["stoch"]
        # if self._nn_cfg.dyn_discrete:
            # stoch = tf.reshape(stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
        target = {
            "embed": context["embed"],
            "stoch": stoch,
            "deter": start["deter"],
            "feat": context["feat"],
        }[self._expl_cfg.disag_target]
        inputs = context["feat"]
        if self._expl_cfg.disag_action_cond:
            inputs = torch.cat([inputs, data["action"]], dim=-1)
        mets = self._train_ensemble(inputs, target)
        metrics.update({"ensemble/" + k:v for k,v in mets.items()})
        
        metrics.update(self._behavior._train(start, self._intrinsic_reward)[-1])
        metrics.update({"ensemble/disag_pre_exp_scale" :  self.disag_pre_exp_scale})
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        inputs = feat
        if self._expl_cfg.disag_action_cond:
            inputs = torch.cat([inputs, action], dim=-1) # TORCH version
        preds = torch.cat([head(inputs).mode().unsqueeze(0) for head in self._networks], dim=0)
        # Unsqueeze seems necessary for torch to not concatenate on the trajectory dimension
        # preds.shape = (disag_models=10, traj_len=15, feat_size)
        # NOTE (Lukas): preds values are in 1e-1 domain, not quite large.
        disag = torch.mean(torch.std(preds, 0), -1)
        # NOTE (Lukas): disag values are in 1e-1 domain too.
        # disag.shape = (traj_len)
        # TODO (Lukas): Finetune these scale updates.
        if torch.mean(self.disag_pre_exp_scale * disag) < 0.4 and torch.max(self.disag_pre_exp_scale * disag) < 0.6:
            self.disag_pre_exp_scale *= 1.5
        elif torch.mean(self.disag_pre_exp_scale * disag) > 2.0:
            self.disag_pre_exp_scale /= 1.5
        if self._expl_cfg.disag_log:
            # TODO (Lukas): Tune disag_pre_log_scale
            disag = torch.log(self._expl_cfg.disag_pre_log_scale * disag)
        elif self._expl_cfg.disag_exp:
            # NOTE (Lukas): disag > 0, hence the exp term will be larger than 1.0 and the final term never negative.
            disag = torch.exp(self.disag_pre_exp_scale * disag) - 1.0
        reward = self._expl_cfg.expl_intr_scale * disag
        # NOTE (Lukas): The following is never called.
        if self._expl_cfg.expl_extr_scale:
            # TODO (Lukas): Check if cast to torch.float32 is necessary
            reward += self._expl_cfg.expl_extr_scale * self._reward(feat, state, action)
        
        # NOTE (Lukas): We need one more dim on the right side
        reward = reward.unsqueeze(-1)
        return reward

    def _train_ensemble(self, inputs, targets):
        # NOTE: disag_offset describes the offset in steps the predictors shall predict.
        # If disag_offset=1, then the predictors shall predict the state in the next time step t+1 from time step t.
        if self._expl_cfg.disag_offset:
            # NOTE: The dimensions seem twisted here, the offset should be applied to the first, not the second!
            targets = torch.tensor(targets[self._expl_cfg.disag_offset :, :],requires_grad=True)
            inputs = torch.tensor(inputs[: -self._expl_cfg.disag_offset, :],requires_grad=True)
            # NOTE: Now, inputs.shape=[traj_len-disag_offset, stoch+deter]

        # Only use the following line when disag input is only stoch
        # inputs = inputs[:,:-200]
        with utils_dreamer.RequiresGrad(self):
            preds = [head(inputs) for head in self._networks]
            # NOTE (Lukas): .view(1) necessary because the log_probs are dimension-less scalars.
            # However, .cat() requires at least one dimension.
            likes = [torch.mean(pred.log_prob(targets)).view(1) for pred in preds]
            likes = torch.cat(likes, dim=0)
            loss = -torch.sum(likes)
            # TODO: Check if self.parameters() works or if we need to provide self._networks.parameters() in another shape.
            metrics = self._opt(loss, self.parameters())
            return metrics
    
    """ 
    TF version
        def _intrinsic_reward(self, feat, state, action):
            inputs = feat
            if self._expl_cfg.disag_action_cond:
                # inputs = tf.concat([inputs, action], -1) 
            preds = [head(inputs, tf.float32).mean() for head in self._networks]
            disag = tf.reduce_mean(tf.math.reduce_std(preds, 0), -1)
            torch.std_mean()
            if self._expl_cfg.disag_log:
                disag = tf.math.log(disag)
            reward = self._expl_cfg.expl_intr_scale * disag
            if self._expl_cfg.expl_extr_scale:
                reward += tf.cast(self._expl_cfg.expl_extr_scale * self._reward(feat, state, action), tf.float32)
            return reward

    def _train_ensemble(self, inputs, targets):
        # NOTE: disag_offset describes the offset in steps the predictors shall predict.
        # If disag_offset=1, then the predictors shall predict the state in the next time step t+1 from time step t.
        
        if self._expl_cfg.disag_offset:
            # NOTE: The dimensions seem twisted here, the offset should be applied to the first, not the second!
            # targets = targets[:, self._expl_cfg.disag_offset :]
            # inputs = inputs[:, : -self._expl_cfg.disag_offset]
            targets = targets[self._expl_cfg.disag_offset :, :]
            inputs = inputs[: -self._expl_cfg.disag_offset, :]
            # NOTE: inputs.shape=[traj_len-disag_offset, stoch+deter]
        
        targets = tf.stop_gradient(targets)
        inputs = tf.stop_gradient(inputs)

        # Only use the following line when disag input is only stoch
        # inputs = inputs[:,:-200]
        with tf.GradientTape() as tape:
            preds = [head(inputs) for head in self._networks]
            likes = [tf.reduce_mean(pred.log_prob(targets)) for pred in preds]
            loss = -tf.cast(tf.reduce_sum(likes), tf.float32)
        metrics = self._opt(tape, loss, self._networks)
        return metrics
    """


def build_exploration_module(cfg, default_args=None):
    return build_from_cfg(cfg, EXPLORATION, default_args)
