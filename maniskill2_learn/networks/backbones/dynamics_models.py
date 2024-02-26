import torch, torch.nn as nn
from ..backbones import mlp, rssm, rl_cnn
from maniskill2_learn.networks import utils_dreamer
from maniskill2_learn.utils.torch.module_utils import ExtendedModule

to_np = lambda x: x.detach().cpu().numpy()


class WorldModel(ExtendedModule):
    def __init__(self, step, nn_cfg, behaviour_cfg, train_cfg, env_cfg):
        super(WorldModel, self).__init__()
        self._step = step
        self._nn_cfg = nn_cfg
        self._behaviour_cfg = behaviour_cfg
        self._train_cfg = train_cfg
        self._env_cfg = env_cfg
        self._use_amp = False
        self.encoder = rl_cnn.ConvEncoder(
            self._env_cfg.channels,
            depth=self._nn_cfg.cnn_depth,
            act_cfg=self._nn_cfg.act,  # NOTE: Converted activation function string to config dict
            norm_cfg=self._nn_cfg.norm_cnn,
            kernels=self._nn_cfg.encoder_kernels,
            stride=self._nn_cfg.encoder_stride,
        )
        if self._env_cfg.size[0] == 64 and self._env_cfg.size[1] == 64:
            embed_size = 2 ** (len(self._nn_cfg.encoder_kernels) - 1) * self._nn_cfg.cnn_depth
            embed_size *= 2 * 2
        elif self._env_cfg.size[0] == 32 and self._env_cfg.size[1] == 32:
            embed_size = 2 ** (len(self._nn_cfg.encoder_kernels) - 1) * self._nn_cfg.cnn_depth
            embed_size *= 2 * 2 # TODO: Current embed size: 1024. Check in RSSM if still compatible despite smaller image
        else:
            raise NotImplemented(f"{self._env_cfg.size} is not applicable now")
        
        self.dynamics = rssm.RSSM(
            self._nn_cfg.dyn_stoch,
            self._nn_cfg.dyn_deter,
            self._nn_cfg.dyn_hidden,
            self._nn_cfg.dyn_input_layers,
            self._nn_cfg.dyn_output_layers,
            self._nn_cfg.dyn_rec_depth,
            self._nn_cfg.dyn_shared,
            self._nn_cfg.dyn_discrete,
            self._nn_cfg.act,
            self._nn_cfg.dyn_mean_act,
            self._nn_cfg.dyn_std_act,
            self._nn_cfg.dyn_temp_post,
            self._nn_cfg.dyn_min_std,
            self._nn_cfg.dyn_cell,
            self._env_cfg.num_actions,
            embed_size,
        )
        self.heads = nn.ModuleDict()
        channels = self._env_cfg.channels
        # NOTE: It seems tuple + list concatenation is not possible.
        shape = (channels, self._env_cfg.size[0], self._env_cfg.size[1])
        if self._nn_cfg.dyn_discrete:
            feat_size = self._nn_cfg.dyn_stoch * self._nn_cfg.dyn_discrete + self._nn_cfg.dyn_deter
        else:
            feat_size = self._nn_cfg.dyn_stoch + self._nn_cfg.dyn_deter
        self.heads["image"] = rl_cnn.ConvDecoder(
            feat_size,  # pytorch version
            depth=self._nn_cfg.cnn_depth,
            act_cfg=self._nn_cfg.act,
            norm_cfg=self._nn_cfg.norm_cnn,
            shape=shape,
            kernels=self._nn_cfg.decoder_kernels,
            stride=self._nn_cfg.decoder_stride,
            thin=self._nn_cfg.decoder_thin,
        )
        self.heads["reward"] = mlp.DenseHead(
            feat_size, 
            [], 
            self._nn_cfg.reward_layers, 
            self._nn_cfg.units, 
            act_cfg=self._nn_cfg.act, # pytorch version
            norm_cfg=self._nn_cfg.norm_mlp,
            bias=self._nn_cfg.bias,
        )
        if self._nn_cfg.pred_discount:
            self.heads["discount"] = mlp.DenseHead(
                feat_size,  # pytorch version
                [],
                self._nn_cfg.discount_layers,
                self._nn_cfg.units,
                self._nn_cfg.act,
                norm_cfg=self._nn_cfg.norm_mlp,
                bias=self._nn_cfg.bias,
                dist="binary",
            )
        for name in self._nn_cfg.grad_heads:
            assert name in self.heads, name
        self._model_opt = utils_dreamer.Optimizer(
            "model",
            self.parameters(),
            self._train_cfg.model_lr,
            self._train_cfg.opt_eps,
            self._train_cfg.grad_clip,
            self._nn_cfg.weight_decay,
            opt=self._train_cfg.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=self._nn_cfg.reward_scale, discount=self._nn_cfg.discount_scale, image=self._nn_cfg.image_scale)

    def _train(self, data):
        """Train the dynamics model based on the prediction accuracy of the observations along a trajectory.
        Firstly, train the forward dynamics model in the latent space.
        Secondly, train the (encoder-)decoder architecture to correctly reconstruct the image from its encoding.
        Thirdly, train the reward and the discount head.
        The encoder architecture is indirectly trained from the losses of the aforementioned modules.

        Keyword arguments:
          data (dict['action', 'image', 'reward', 'discount']) -- Trajectory of environment observations with their preceding action

        Return:
          post -- posterior latent states
          context (dict['embed', 'feat', 'kl', 'postent']) --
            embed: feature embeddings of each observation, 
            feat: concatenated stochastic and deterministic part of the posterior latent state of each step,
            kl: KL values for each step,
            postent: entropy of posterior latent state for each step 
          metrics -- Training metrics
        """
        self.train()
        data = self.preprocess(data)

        with utils_dreamer.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Embed the image with the CNN
                embed = self.encoder(data)
                # Compute the prior and posterior latent state for each transition in the action trajectory.
                # The prior predicts the next latent state merely from the prior latent state and the action.
                # The posterior is also based on the observation's encoding.
                post, prior = self.dynamics.observe(embed, data["action"])

                # For each transition in the trajectory, obtain the KL loss between the prior and posterior distribution.
                kl_balance = utils_dreamer.schedule(self._nn_cfg.kl_balance, self._step)
                kl_free = utils_dreamer.schedule(self._nn_cfg.kl_free, self._step)
                kl_scale = utils_dreamer.schedule(self._nn_cfg.kl_scale, self._step)
                kl_loss, kl_value = self.dynamics.kl_loss(
                    post, prior, self._nn_cfg.kl_forward, kl_balance, kl_free, kl_scale
                )
                losses = {}
                likes = {}
                # Absolute and relative predictor error statistics.
                abs_errors_max = {}
                abs_errors_avg = {}
                abs_errors_std = {}
                rel_errors_max = {}
                rel_errors_avg = {}
                rel_errors_std = {}

                # Predict image, reward and discount and compare with true observation.
                for name, head in self.heads.items():
                    grad_head = name in self._nn_cfg.grad_heads
                    feat = self.dynamics.get_feat(post)
                    if not grad_head:
                        feat = feat.detach()
                    pred = head(feat)
                    like = pred.log_prob(data[name])
                    abs_err = torch.abs(pred.mean - data[name])
                    rel_err = torch.div(abs_err, torch.abs(data[name]))
                    
                    max_abs_err = abs_err.max().item()
                    avg_abs_err = abs_err.mean().item()
                    std_abs_err = abs_err.std().item()
                    max_rel_err = rel_err.max().item()
                    avg_rel_err = rel_err.mean().item()
                    std_rel_err = rel_err.std().item()
                    # like.shape = [traj_len]
                    abs_errors_max[name] = max_abs_err
                    rel_errors_max[name] = max_rel_err
                    abs_errors_avg[name] = avg_abs_err
                    rel_errors_avg[name] = avg_rel_err
                    abs_errors_std[name] = std_abs_err
                    rel_errors_std[name] = std_rel_err
                    
                    likes[name] = like
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                model_loss = sum(losses.values()) + kl_loss
                
            metrics = self._model_opt(model_loss, self.parameters())

        # for name, loss in losses.items():
        #   print(f"{name}_loss: {loss}")
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics.update({f"{name}_abs_err_max": err for name, err in abs_errors_max.items()})
        metrics.update({f"{name}_rel_err_max": err for name, err in rel_errors_max.items()})
        metrics.update({f"{name}_abs_err_avg": err for name, err in abs_errors_avg.items()})
        metrics.update({f"{name}_rel_err_avg": err for name, err in rel_errors_avg.items()})
        metrics.update({f"{name}_abs_err_std": err for name, err in abs_errors_std.items()})
        metrics.update({f"{name}_rel_err_std": err for name, err in rel_errors_std.items()})
        metrics["kl_balance"] = kl_balance
        metrics["kl_free"] = kl_free
        metrics['kl_loss'] = to_np(kl_loss)
        metrics["kl_scale"] = kl_scale
        metrics['loss'] = to_np(sum(losses.values()))
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
            metrics["post_ent"] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def get_uncertainty_metrics(self, trajectory):
        self.eval()
        trajectory = self.preprocess(trajectory)
        embed = self.encoder(trajectory)
        post, prior = self.dynamics.observe(embed, trajectory["action"])
        kl_balance = utils_dreamer.schedule(self._nn_cfg.kl_balance, self._step)
        kl_free = utils_dreamer.schedule(self._nn_cfg.kl_free, self._step)
        kl_scale = utils_dreamer.schedule(self._nn_cfg.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._nn_cfg.kl_forward, kl_balance, kl_free, kl_scale
        )
        feat = self.dynamics.get_feat(post)
        
        return feat, post, kl_value

    def preprocess(self, obs):
        """Normalize the pixel values, clip rewards and format the input."""
        obs = obs.copy()
        obs["image"] = torch.tensor(obs["image"]).to(self.device) / 255.0 - 0.5
        if "reward" in obs:
            if self._env_cfg.clip_rewards == "tanh":
                obs["reward"] = torch.tanh(torch.tensor(obs["reward"], device=self.device))
            elif self._env_cfg.clip_rewards == "identity":
                obs["reward"] = torch.tensor(obs["reward"]).to(self.device)
            else:
                raise NotImplemented(f"{self._env_cfg.clip_rewards} is not implemented")
        if "discount" in obs:
            obs["discount"] *= self._behaviour_cfg.discount
            obs["discount"] = torch.tensor(obs["discount"]).to(self.device).unsqueeze(-1)
        obs = {k: torch.tensor(v).to(self.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        truth = data["image"][:6] + 0.5
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(embed[:6, :5], data["action"][:6, :5])
        recon = self.heads["image"](self.dynamics.get_feat(states)).mode()[:6]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        openl = self.heads["image"](self.dynamics.get_feat(prior)).mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2

        return torch.cat([truth, model, error], 2)


class ImagBehavior(ExtendedModule):
    def __init__(self, nn_cfg, behaviour_cfg, train_cfg, env_cfg, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._nn_cfg = nn_cfg
        self._behaviour_cfg = behaviour_cfg
        self._train_cfg = train_cfg
        self._env_cfg = env_cfg

        # NOTE: Torch's AMP (Automatic Mixed Precision) is definitely useful for efficiency
        # self._use_amp = True if self._log_cfg.precision==16 else False
        self._use_amp = True
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward

        # Architecture
        if self._nn_cfg.dyn_discrete:
            feat_size = self._nn_cfg.dyn_stoch * self._nn_cfg.dyn_discrete + self._nn_cfg.dyn_deter
        else:
            feat_size = self._nn_cfg.dyn_stoch + self._nn_cfg.dyn_deter
        self.actor = mlp.ActionHead(
            feat_size,  # pytorch version
            self._env_cfg.num_actions,
            self._nn_cfg.actor_layers,
            self._nn_cfg.units,
            act_cfg=self._nn_cfg.act,
            norm_cfg=None,
            bias=True,
            dist=self._behaviour_cfg.actor_dist,
            init_std=self._behaviour_cfg.actor_init_std,
            min_std=self._behaviour_cfg.actor_min_std,
            action_disc=self._behaviour_cfg.actor_dist,
            temp=self._behaviour_cfg.actor_temp,
            outscale=self._behaviour_cfg.actor_outscale,
        )
        self.value = mlp.DenseHead(
            feat_size,  # pytorch version
            [],
            self._nn_cfg.value_layers,
            self._nn_cfg.units,
            act_cfg=self._nn_cfg.act,
            norm_cfg=self._nn_cfg.norm_mlp,
            bias=self._nn_cfg.bias,
            dist=self._nn_cfg.value_head,
        )
        if self._train_cfg.slow_value_target or self._train_cfg.slow_actor_target:
            self._slow_value = mlp.DenseHead(
                feat_size, # pytorch version
                [], 
                self._nn_cfg.value_layers, 
                self._nn_cfg.units, 
                act_cfg=self._nn_cfg.act,
                norm_cfg=self._nn_cfg.norm_mlp,
                bias=self._nn_cfg.bias,
            )
            self._updates = 0
        kw = dict(wd=self._nn_cfg.weight_decay, opt=self._train_cfg.opt, use_amp=self._use_amp)
        self._actor_opt = utils_dreamer.Optimizer(
            "actor",
            self.actor.parameters(),
            self._train_cfg.actor_lr,
            self._train_cfg.opt_eps,
            self._train_cfg.actor_grad_clip,
            **kw,
        )
        self._value_opt = utils_dreamer.Optimizer(
            "value",
            self.value.parameters(),
            self._train_cfg.value_lr,
            self._train_cfg.opt_eps,
            self._train_cfg.value_grad_clip,
            **kw,
        )

    def _train(self, start, objective=None, reward=None, repeats=None):
        """Train the imagination ability of Dreamer.
        Parameters:
            start: posterior latent states
            objective: objective/reward function
            reward: reward placeholder variable, actually not used
        """
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}
        # NOTE: As in WorldModel _train(), start is not only ONE, but MULTIPLE latent states. Take only first.
        start = {key:value[0,:].unsqueeze(0) for key, value in start.items()}
        # TODO (Lukas): The above operation completely removes the first dimension of each dict array. 
        # Check if this causes any inconsistencies in the subsequent code. 
        # DONE: It indeed causes inconsistencies, therefore I added dimension again (unsqueeze(0)).
        # TODO (Lukas): Again, check if ADDING the first dimension causes any inconsistencies in _imagine.
        # DONE: It seems to work this way.

        # Firstly, train the policy network.
        with utils_dreamer.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # Imagine executing an trajectory and obtain the latent state and its feature representation
                # TODO (Lukas): Currently, self._behaviour_cfg.imag_horizon=15. Test if adaption is adequate.
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._behaviour_cfg.imag_horizon, repeats
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

                # Compute the state values in a Forward View TD(lambda) fashion.
                target, weights = self._compute_target(
                    imag_feat, imag_state, imag_action, reward, actor_ent, state_ent, self._train_cfg.slow_actor_target
                )

                # Assess the actor with these state values.
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat, imag_state, imag_action, target, actor_ent, state_ent, weights
                )
                # print(f"actor_loss : {actor_loss}")
                metrics.update(mets)
                if self._train_cfg.slow_value_target != self._train_cfg.slow_actor_target:
                    target, weights = self._compute_target(
                        imag_feat,
                        imag_state,
                        imag_action,
                        reward,
                        actor_ent,
                        state_ent,
                        self._train_cfg.slow_value_target,
                    )
                value_input = imag_feat

        # Secondly, train the value network.
        with utils_dreamer.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                # NOTE: Value network output is not a scalar, but a distribution
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                value_loss = -value.log_prob(target.detach())
                # NOTE (Lukas): Will not be executed
                if self._behaviour_cfg.value_decay:
                    value_loss += self._behaviour_cfg.value_decay * value.mode()
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
                # print(f"value_loss : {value_loss}")

        metrics["reward_mean"] = to_np(torch.mean(reward))
        metrics["reward_std"] = to_np(torch.std(reward))
        metrics["actor_ent"] = to_np(torch.mean(actor_ent))
        with utils_dreamer.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        """Compute trajectory of sampled actions from policy

        Arguments:
          start -- initial latent state
          policy -- policy to sample actions from
          horizon -- length of desired trajectory

        Returns:
          feats -- feature representations of each latent state on the trajectory
          states [dict('deter','stoch','mean','std')] -- trajectory state informations 
          actions -- trajectory's actions
        """
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        # NOTE (Lukas): We should not(!) flatten here, we need a batch dim.
        # start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            """Sample action and imagine its outcome when executed at the current latent state."""
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            # Sample action from the policy's action distribution
            action = policy(inp).sample()
            # Imagine the successor state 
            succ = dynamics.img_step(state, action, sample=self._behaviour_cfg.imag_sample)
            return succ, feat, action

        feat = 0 * dynamics.get_feat(start)
        action = policy(feat).mode()
        succ, feats, actions = utils_dreamer.static_scan(step, [torch.arange(horizon)], (start, feat, action))
        # NOTE (Lukas): succ.keys() contains 'deter' and 'stoch'.

        # Concat the start latent state with the rest of the trajectory
        # Since I did not flatten the array here, we do not need to add another dimension above.
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        # NOTE (Lukas): Remove second dimension
        states = {k:v.squeeze(1) for k,v in states.items()}
        actions = actions.squeeze(1)
        feats = feats.squeeze(1)

        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent, slow):
        """Compute target value V for state in the Forward View TD(lambda) fashion.
        
        Parameters:
            imag_feat: features of the latent trajectory, 
            imag_state [dict('deter','stoch','mean','std')]: latent trajectory state informations,
            imag_action: actions that created the latent trajectory, 
            reward: imaginary state rewards generated by Dreamer's reward head, 
            actor_ent: entropy of the actor policy that generated this latent trajectory, 
            state_ent: entropy of the stochastic part of the latent states along this latent trajectory, 
            slow: Whether to use the slow value network or the normal value network to compute the state values

        """
        if "discount" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._world_model.heads["discount"](inp).mean
        else:
            discount = self._behaviour_cfg.discount * torch.ones_like(reward)

        # Increase reward by entropy of policy distribution
        if self._behaviour_cfg.future_entropy and self._behaviour_cfg.actor_entropy() > 0:
            reward += self._behaviour_cfg.actor_entropy() * actor_ent
        if self._behaviour_cfg.future_entropy and self._behaviour_cfg.actor_state_entropy() > 0:
            reward += self._behaviour_cfg.actor_state_entropy() * state_ent
        if slow:
            value = self._slow_value(imag_feat).mode()
        else:
            value = self.value(imag_feat).mode()

        # Compute value of initial state in the Forward View TD(lambda) fashion.
        target = utils_dreamer.lambda_return(
            reward[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self._behaviour_cfg.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
        return target, weights

    def _compute_actor_loss(self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent, weights):
        """Compute the actor loss as the mean reward (target*weights) of the trajectory he just executed.
        If activated, the actor is further assessed based on the entropy of its policy and the latent states.

        Parameters:
            imag_feat: features of the latent trajectory, 
            imag_state [dict('deter','stoch','mean','std')]: latent trajectory state informations,
            imag_action: actions that created the latent trajectory, 
            target: unweighted(!) cumulative TD(lambda) state values generated from the value network on this trajectory, 
            actor_ent: entropy of the actor policy that generated this latent trajectory, 
            state_ent: entropy of the stochastic part of the latent states along this latent trajectory, 
            weights: cumulative discount weights to multiply the target values with

        """
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        target = torch.stack(target, dim=1)
        if self._behaviour_cfg.imag_gradient == "dynamics":
            actor_target = target
        elif self._behaviour_cfg.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._behaviour_cfg.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._behaviour_cfg.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._behaviour_cfg.imag_gradient)
        if not self._behaviour_cfg.future_entropy and (self._behaviour_cfg.actor_entropy() > 0):
            actor_ent = actor_ent.unsqueeze(-1)
            actor_target += self._behaviour_cfg.actor_entropy() * actor_ent[:-1][:, :, None]
        if not self._behaviour_cfg.future_entropy and (self._behaviour_cfg.actor_state_entropy() > 0):
            # NOTE (Lukas): It could be possible that state_ent needs to be extended by one dimension aswell. 
            # But this code seems not to occur, remove if you're sure.
            actor_target += self._behaviour_cfg.actor_state_entropy() * state_ent[:-1]
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._train_cfg.slow_value_target or self._train_cfg.slow_actor_target:
            if self._updates % self._train_cfg.slow_target_update == 0:
                mix = self._train_cfg.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
