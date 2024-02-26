import torch
from torch import nn
from torch import distributions as torchd

from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.utils.torch import load_checkpoint, ExtendedModule
from ..modules import build_norm_layer, need_bias, build_activation_layer

from ..builder import BACKBONES
from ...networks import utils_dreamer
from maniskill2_learn.utils.torch.module_utils import ExtendedModule

# TODO: Implement the tf.nest functions by yourself.
# TODO: Implement GRUCell by yourself to enable normalization
# TODO: Determine input size parameter.


@BACKBONES.register_module()
class RSSM(ExtendedModule):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act_cfg=dict(type="ELU"),
        mean_act="none",
        std_act="Softplus",
        temp_post=True,
        min_std=0.1,
        cell="gru",
        num_actions=None,
        embed=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        self._act_cfg = act_cfg
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._embed = embed

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden))
            inp_layers.append(build_activation_layer(self._act_cfg))
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)

        if cell == "gru":
            self._cell = GRUCell(self._hidden, self._deter)
        elif cell == "gru_layer_norm":
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
        else:
            raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden))
            img_out_layers.append(build_activation_layer(self._act_cfg))
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden))
            obs_out_layers.append(build_activation_layer(self._act_cfg))
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)

        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self.device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self.device),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self.device),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self.device),
                std=torch.zeros([batch_size, self._stoch]).to(self.device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self.device),
                deter=deter,
            )
        return state

    def observe(self, embed, action, state=None, sample_latent_state=True):
        """Process an entire trajectory of observations.
        Compute the prior and posterior latent state for each transition in this trajectory.

        Keyword arguments:
          embed (traj_len, embed_size) -- encoder's embeddings of the trajectory's observation
          action (traj_len, action_shape) -- trajectory's actions
          state -- initial latent state
          sample_latent_state -- whether the returned posterior should be a sample or the mode of the distribution

        Returns:
          Posterior and prior latent state for each transition in the trajectory.
        """
        # NOTE: Swapping was done because the trajectory dim was originally the second, not the first.
        # swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            # state = self.initial(action.shape[0])
            # It does not make sense to generate zeros for the entire trajectory.
            # We need only ONE latent state, not traj_len many. I hence write the following:
            state = self.initial(1)
            # This initial(1) call was also done by the authors in _policy(), I don't know why they used action.shape here.
            # First dimension is redundant, remove it.
            state = {key: value.squeeze(0) for key, value in state.items()}

        # embed, action = swap(embed), swap(action)
        post, prior = utils_dreamer.static_scan(
            lambda prev_state, prev_act, embed: self.obs_step(prev_state[0], prev_act, embed, sample=sample_latent_state),
            (action, embed),
            (state, state),
        )
        # post = {k: swap(v) for k, v in post.items()}
        # prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        """Predict an entire trajectory of latent states from a trajectory of actions.

        Keyword arguments:
          action -- trajectory's actions
          state -- initial latent state

        Returns:
          Prior latent states for each state in the trajectory.
        """
        # swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(1)
            state = {key: value.squeeze(0) for key, value in state.items()}
        assert isinstance(state, dict), state
        # action = swap(action)
        prior = utils_dreamer.static_scan(self.img_step, [action], state)
        prior = prior[0]
        # prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        """Divide hidden state into deterministic and stochastic part and concatenate them."""
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        """Return a torch distribution object corresponding to the distribution parametrized by state."""
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(utils_dreamer.OneHotDist(logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            dist = utils_dreamer.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        return dist

    def obs_step(self, prev_state, prev_action, embed, sample=True):
        """Process an observation and compute the prior and posterior latent state.

        Keyword arguments:
          prev_state array[stoch, deter, mean, std] -- previous latent state
          prev_action -- previous action
          embed -- encoder's embedding of the current observation
          sample -- whether to sample or only choose the mode from the stochastic hidden state

        Returns:
          Posterior and prior latent state.
        """
        # Prior has same format and shape as prev_state
        
        # NOTE: action has shape (action_shape), but latent elements have shape (1,50). 
        # Therefore, we add a first dimension to action for consistency (else we have an error).
        # embed also has shape (1, 1024) (but NOT ALWAYS!), so I guess it makes most sense to adjust actions to it.
        # Since embed seems to have shape (1024) sometimes aswell, I check embed for correct dim aswell.
        
        if len(prev_action.shape) < len(prev_state["deter"].shape):
            prev_action = prev_action[None, :]
        
        if len(embed.shape) < len(prev_state["deter"].shape):
            embed = embed[None, :] 

        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                # Concatenate deterministic hidden state with encoder embedding
                x = torch.cat([prior["deter"], embed], -1)
            else:
                x = embed
            # Process embedding vector with Sequential NN to obtain hidden state.
            x = self._obs_out_layers(x)
            # x now has shape (batch_size, self._hidden)

            # Obtain the distribution parameters for the stochastic part of the hidden state.
            stats = self._suff_stats_layer("obs", x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        """Predict the next latent state from the current latent state and the subsequent action.

        Keyword arguments:
          prev_state -- previous latent state
          prev_action -- previous action
          embed -- placebo for the encoder's observation embedding
          sample -- whether to sample or only choose the mode from the stochastic hidden state

        Returns:
          The (prior) latent state.
        """
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                # NOTE: self._embed is None by default, so just ignore it.
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros(shape)
            x = torch.cat([prev_stoch, prev_action, embed], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
            # x.shape=(batch_size, self._stoch + action_shape)
        
        # Process the stochastic prior through a linear NN.
        x = self._inp_layers(x)
        # x.shape=(batch_size, self._hidden)

        # Feed both the stochastic prior (x) and the deterministic prior through the GRU cells.
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
            # x and deter keep their shape
        
        # Postprocess the stochastic part of the GRU output state through another linear NN.
        x = self._img_out_layers(x)
        # x.shape=(batch_size, self._hidden)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def _suff_stats_layer(self, name, x):
        """Process embedding through linear NN to obtain distribution."""
        if name == "ims":
            x = self._ims_stat_layer(x)
        elif name == "obs":
            x = self._obs_stat_layer(x)
        else:
            raise NotImplementedError
        if self._discrete:
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, forward, balance, free, scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = kld(
                dist(lhs) if self._discrete else dist(lhs)._dist, dist(rhs) if self._discrete else dist(rhs)._dist
            )
            loss = torch.mean(torch.maximum(value, free))
        else:
            value_lhs = value = kld(
                dist(lhs) if self._discrete else dist(lhs)._dist,
                dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist,
            )
            value_rhs = kld(
                dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
                dist(rhs) if self._discrete else dist(rhs)._dist,
            )
            loss_lhs = torch.maximum(torch.mean(value_lhs), torch.tensor([free])[0])
            loss_rhs = torch.maximum(torch.mean(value_rhs), torch.tensor([free])[0])
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        loss *= scale
        return loss, value


class GRUCell(ExtendedModule):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=norm is not None)
        if norm:
            self._norm = nn.LayerNorm(3 * size)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]
