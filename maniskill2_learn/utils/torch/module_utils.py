from torch.nn import Module, ModuleList, Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import parameters_to_vector
from contextlib import contextmanager
import torch, numpy as np
import math
from collections import OrderedDict, defaultdict
from maniskill2_learn.utils.data import GDict, DictArray, to_torch
from maniskill2_learn.utils.meta import get_logger
from .misc import no_grad, mini_batch, run_with_mini_batch

from time import time

class ExtendedModuleBase(Module):
    def __init__(self, *args, **kwargs):
        super(ExtendedModuleBase, self).__init__(*args, **kwargs)
        self._in_test = False  # For RL test mode ( do not update obs_rms and rew_rms )

    def set_mode(self, mode="train"):
        self._in_test = mode == "test"
        for module in self.children():
            if isinstance(module, ExtendedModuleBase):
                module.set_mode(mode)
        return self

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def trainable_parameters(self):
        return [_ for _ in self.parameters() if _.requires_grad]

    @property
    def size_trainable_parameters(self):
        return GDict([_ for _ in self.parameters() if _.requires_grad]).nbytes_all

    @property
    def num_trainable_parameters(self):
        return sum([_.numel() for _ in self.parameters() if _.requires_grad])

    @property
    @no_grad
    def grad_norm(self, ord=2):
        grads = [torch.norm(_.grad.detach(), ord) for _ in self.parameters() if _.requires_grad and _.grad is not None]
        ret = torch.norm(torch.stack(grads), ord).item() if len(grads) > 0 else 0.0
        return ret

    @no_grad
    def vector_parameters(self):
        return parameters_to_vector(self.parameters())

    def pop_attr(self, name):
        if hasattr(self, name):
            ret = getattr(self, name)
            setattr(self, name, None)
            return ret
        else:
            return None


class ExtendedModule(ExtendedModuleBase):
    # DDP has attribute device!!!!
    @property
    def device(self):
        return next(self.parameters()).device

    @contextmanager
    def no_sync(self):
        yield


class ExtendedModuleList(ModuleList, ExtendedModule):
    @property
    def device(self):
        return next(self.parameters()).device

    @contextmanager
    def no_sync(self):
        yield


class ExtendedSequential(Sequential, ExtendedModule):
    def append(self, module):
        index = len(self)
        self.add_module(str(index), module)

    def append_list(self, modules):
        assert isinstance(modules, Sequential) or isinstance(modules, (list, tuple))
        for module in modules:
            self.append(module)


class ExtendedDDP(DDP, ExtendedModuleBase):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    # def __init__(self, *args, **kwargs):
    #     super(ExtendedDDP, self).__init__(*args, **kwargs)
    #     self._in_test = False  # For RL test mode which is different from torch eval mode

    # def set_mode(self, mode='train'):
    #     self._in_test = mode == 'test'
    #     for module in self.children():
    #         if isinstance(module, ExtendedModule):
    #             module.set_mode(mode)
    #     return self

    # def __getattr__(self, name):
    #     try:
    #         return super().__getattr__(name)
    #     except AttributeError:
    #         return getattr(self.module, name)

    # """
    # DDP has device
    # @property
    # def device(self):
    #     return next(self.parameters()).device
    # """

    # def trainable_parameters(self):
    #     return [_ for _ in self.parameters() if _.requires_grad]

    # def max_grad_norm(self, ord=None):
    #     ret = [torch.linalg.norm(_.grad, ord) for _ in self.parameters() if _.requires_grad and _.grad is not None]
    #     return max(ret)


class BaseAgent(ExtendedModule):
    def __init__(self, *args, **kwargs):
        super(BaseAgent, self).__init__(*args, **kwargs)
        self._device_ids = None
        self._be_data_parallel = False
        self._tmp_attrs = {}

        self.obs_processor = None
        self.obs_rms = None
        self.rew_rms = None
        self.batch_size = None
        self.PRINT_DETAILED_TIME = False
        self.time_measures = OrderedDict()
        # init ret dictionary
        self.ret = defaultdict(list)
        self.first_time_print_time = True
        self.print_time_interval_average = 1
    def reset(self, *args, **kwargs):
        pass

    @property
    def has_obs_process(self):
        return self.obs_rms is not None or self.obs_processor is not None

    def join_key_returns(self, ret_prime, keyword, also_normal=False):
        if keyword is None:
            keyword = ""
            also_normal = False
        else:
            keyword = keyword + "/" if keyword is not None else ""
    
        for key in ret_prime:
            n_key = keyword + key
            self.ret[n_key].append(ret_prime[key])
            if also_normal:
                self.ret[key].append(ret_prime[key])

    def return_ret_dict(self):
        ret_dict = {}
        for key in self.ret:
            ret_dict[key] = sum(self.ret[key]) / len(self.ret[key])
        self.ret = defaultdict(list)
        return ret_dict

    @no_grad
    def process_obs(self, data, **kwargs):
        for key in ["obs", "next_obs"]:
            if key in data:
                if self.obs_rms is not None:
                    data[key] = run_with_mini_batch(
                        self.obs_rms.normalize,
                        data[key],
                        **kwargs,
                        device=self.device,
                        wrapper=False,
                    )
                if self.obs_processor is not None:
                    data[key] = run_with_mini_batch(self.obs_processor, {"obs": data[key]}, **kwargs)["obs"]
        return data

    def print_time(self):
        if self.PRINT_DETAILED_TIME:
            d = 2
            last_time = 0
            keys = ["checkpoint" , "total", "relat"]
            max_len = [max([len(key) for key in self.time_measures]), 5, 5]
            act_max_len = [max(len(key), l) + d for key,l in zip(keys, max_len)]
            # print header
            for key, act_len in zip(keys, act_max_len):
                print(key, end="") 
                print(max(0,act_len - len(key)) * " ", end="")
            print()
            for key, item in self.time_measures.items():
                end_time = item
                if isinstance(item, list):
                    end_time = item[-1]
        
                print(key + (act_max_len[0] - len(key)) * " ", end="")
                print(float_to_str(end_time, 5) + max(act_max_len[1] - 5, 0) * " ", end="")

                if not isinstance(item, list):
                    item = [item]
                else:
                    rel_time = end_time - last_time
                    print(float_to_str(rel_time, 5) + max( act_max_len[2] - 5, 0) * " ", end="")
                    
                for i in item:
                    print(float_to_str(i - last_time, 5) + "  ", end="")
                    last_time = i
                print()

            if self.first_time_print_time:
                self.first_time_print_time = False
                print("Info: Disable time printing by setting self.PRINT_DETAILED_TIME to False.")


    def measure_time(self, key, reset=False):
        if self.PRINT_DETAILED_TIME:
            if reset:
                self.time_measures = OrderedDict()
                self.ctime = time()
            if key in self.time_measures:
                if isinstance(self.time_measures[key], list):
                    self.time_measures[key].append(time() - self.ctime)
                else:
                    v = self.time_measures[key]
                    self.time_measures[key] = [v]
                    self.time_measures[key].append(time() - self.ctime)
            else:
                self.time_measures[key] = time() - self.ctime

    @no_grad
    def forward(self, obs, **kwargs):
        obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs) if self._in_test else self.obs_rms.add(obs)
        if self.obs_processor is not None:
            obs = self.obs_processor({"obs": obs})["obs"]
        return self.actor(obs, **kwargs)

    def get_dist_with_logp(self, obs, actions=None, **kwargs):
        @mini_batch(False)
        def run(obs, actions, **kwargs):
            ret = self.actor(obs, mode="dist", **kwargs)
            if actions is not None:
                if isinstance(ret, (list, tuple)):
                    ret = [ret[0], ret[0].log_prob(actions)], ret[1]
                else:
                    ret = [ret, ret.log_prob(actions)]
            return ret

        return run(obs=obs, actions=actions, **kwargs, device=self.device)

    def get_values(self, obs, actions=None, **kwargs):
        return run_with_mini_batch(self.critic, obs=obs, actions=actions, **kwargs, device=self.device)

    @no_grad
    def compute_gae(
        self,
        obs,
        next_obs,
        rewards,
        dones,
        episode_dones,
        ignore_dones=True,
        update_rms=True,
        batch_size=None,
    ):
        """
        High-Dimensional Continuous Control Using Generalized Advantage Estimation
            https://arxiv.org/abs/1506.02438
        """
        rewards = to_torch(rewards, device=self.device, non_blocking=True)
        dones = to_torch(dones, device=self.device, non_blocking=True)
        episode_dones = to_torch(episode_dones, device=self.device, non_blocking=True)
        episode_masks = 1.0 - episode_dones.float()
        # get_logger().info(type(episode_dones))
        with self.no_sync(mode="critic"):
            values = self.get_values(
                obs=obs,
                batch_size=batch_size,
                ret_device=self.device,
                wrapper=False,
                episode_dones=episode_dones,
            )

            next_values = self.get_values(
                obs=next_obs,
                batch_size=batch_size,
                ret_device=self.device,
                wrapper=False,
            )

        if self.rew_rms is not None:
            std = self.rew_rms.std
            values = values * std
            next_values = next_values * std
        if not ignore_dones:
            next_values = next_values * (1.0 - dones.float())

        delta = rewards + next_values * self.gamma - values

        coeff = episode_masks * self.gamma * self.lmbda
        advantages = torch.zeros(len(rewards), 1, device=self.device, dtype=torch.float32)
        # print(advantages.shape, coeff.shape, delta.shape)
        # exit(0)

        gae = 0
        for i in range(len(rewards) - 1, -1, -1):
            gae = delta[i] + coeff[i] * gae
            advantages[i] = gae
        returns = advantages + values

        ret = {
            "old_values": values,
            "old_next_values": next_values,
            "original_returns": returns,
            "returns": returns,
            "advantages": advantages,
            # 'v_states': v_states,
            # 'v_next_states': v_next_states,
        }
        if self.rew_rms is not None:
            if update_rms:
                assert self.rew_rms.training
                self.rew_rms.add(ret["returns"])
                self.rew_rms.sync()
            std = self.rew_rms.std

            ret["old_values"] = ret["old_values"] / std
            ret["old_next_values"] = ret["old_next_values"] / std
            ret["returns"] = ret["returns"] / std
        ret = GDict(ret).to_numpy()

        # get_logger().info(f"Values shape {values.shape}")
        # get_logger().info(f"Values {values.mean().item():.2f}, Next Values {next_values.mean().item():.2f} Delta {delta.mean().item():.2f}, Adv {advantages.mean().item():.2f}, Normalized Rew {ret['returns'].mean().item():.2f}, Original Rew {returns.mean().item():.2f}")

        torch.cuda.empty_cache()
        return ret

    def actor_grad(self, with_shared=True):
        ret = {}
        if getattr(self, "actor", None) is None:
            return ret
        ret[f"grad/actor_grad_norm"] = self.actor.grad_norm
        if with_shared:
            assert self.shared_
            if getattr(self.actor_grad.backbone, "visual_nn", None) is not None:
                ret["grad/visual_grad"] = self.actor.backbone.visual_nn.grad_norm

            if self.actor.final_mlp is not None:
                ret["grad/mlp_grad"] = self.actor.final_mlp.grad_norm

    def critic_grad(self, with_shared=True):
        ret = {}
        if getattr(self, "critic", None) is None:
            return ret
        ret[f"grad/critic_grad_norm"] = self.critic.grad_norm
        if with_shared:
            assert self.shared_
            if getattr(self.actor_grad.backbone, "visual_nn", None) is not None:
                ret["grad/visual_grad"] = self.actor.backbone.visual_nn.grad_norm

            if self.actor.final_mlp is not None:
                ret["grad/mlp_grad"] = self.actor.final_mlp.grad_norm

    def to_ddp(self, device_ids=None, find_unused_parameters=True):
        self._device_ids = device_ids
        self.recover_ddp(find_unused_parameters=find_unused_parameters)

    def to_normal(self):
        if self._be_data_parallel and self._device_ids is not None:
            self._be_data_parallel = False
            for module_name in dir(self):
                item = getattr(self, module_name)
                if isinstance(item, DDP):
                    setattr(self, module_name, item.module)

    def recover_ddp(self, find_unused_parameters=True):
        if self._device_ids is None:
            return
        self._be_data_parallel = True
        for module_name in dir(self):
            item = getattr(self, module_name)
            if isinstance(item, ExtendedModule) and len(item.trainable_parameters) > 0:
                if module_name not in self._tmp_attrs:
                    self._tmp_attrs[module_name] = ExtendedDDP(
                        item,
                        device_ids=self._device_ids,
                        find_unused_parameters=find_unused_parameters,
                    )
                setattr(self, module_name, self._tmp_attrs[module_name])

    def is_data_parallel(self):
        return self._be_data_parallel

    def no_sync(self, mode="actor"):
        return getattr(self, mode).no_sync()


def async_no_grad_pi(pi):
    import torch

    def run(*args, **kwargs):
        with pi.no_sync():
            with torch.no_grad():
                return pi(*args, **kwargs)

    return run

def float_to_str(value, length = 3):
    value = round(value + 1e-9, length - 2)
    s = str(value)
    if len(s) < length:
        s += (length - len(s)) * "0"
    if len(s) > length:
        s = s[:length]
    return s

if __name__ == "__main__":
    # Test print
    b = {'start':0,
         'get_start_end':0.332432423432432,
         'slice_trajectory':1.32112213,
         'dreamer':[1.834242432234, 2.0486035346984863, 2.508052110671997, 2.9639298915863037]}
    d = BaseAgent()
    d.time_measures = b
    d.PRINT_DETAILED_TIME = True
    d.print_time()