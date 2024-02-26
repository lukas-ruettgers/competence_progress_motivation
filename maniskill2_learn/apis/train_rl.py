import itertools
import os
import os.path as osp
import time
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np
from maniskill2_learn.env import ReplayMemory, save_eval_statistics
from maniskill2_learn.utils.data import GDict, dict_to_str, is_not_null, num_to_str, to_float
from maniskill2_learn.utils.math import EveryNSteps
from maniskill2_learn.utils.meta import get_logger, get_total_memory, get_world_rank, td_format
from maniskill2_learn.utils.torch import TensorboardLogger, save_checkpoint


class EpisodicStatistics:
    def __init__(self, num_procs, info_keys_mode={}):
        self.num_procs = num_procs
        self.info_keys_mode = {
            "rewards": [True, "sum", "all"],
            "max_single_R": [True, "max", "all"],
            "lens": [True, "sum", "all"],
            "avg_rewards": [True, "mean" , "all"]
        }  # if used in print, stats over episode, if print all infos
        self.info_keys_mode.update(info_keys_mode)
        for key, item in self.info_keys_mode.items():
            assert item[1] in ["mean", "min", "max", "sum"]
            assert item[2] in ["mean", "all"]
        self.num_workers = 1
        self.reset_current()
        self.reset_history()

    def push(self, trajs):
        rewards, dones, index, infos = (
            trajs["rewards"],
            trajs["episode_dones"],
            trajs.get("worker_indices", None),
            trajs.get("infos", None),
        )
        dones = dones.reshape(-1)
        self.expand_buffer(index)
        for j in range(len(rewards)):
            i = 0 if index is None else int(index[j])
            self.current[i]["lens"] += 1
            self.current[i]["rewards"] += to_float(rewards[j])
            if "max_single_R" not in self.current[i]:
                self.current[i]["max_single_R"] = -np.inf
            self.current[i]["max_single_R"] = max(self.current[i]["max_single_R"], to_float(rewards[j]))

            self.current[i]['avg_rewards'] += to_float(rewards[j])

            if infos is not None:
                for key, manner in self.info_keys_mode.items():
                    if key != "rewards" and key in infos:
                        if manner[1] in ["sum", "mean"]:
                            self.current[i][key] += to_float(infos[key][j])
                        elif manner[1] == "min":
                            if key not in self.current[i]:
                                self.current[i][key] = np.inf
                            self.current[i][key] = np.minimum(self.current[i][key], to_float(infos[key][j]))
                        elif manner[1] == "max":
                            if key not in self.current[i]:
                                self.current[i][key] = -np.inf
                            self.current[i][key] = np.maximum(self.current[i][key], to_float(infos[key][j]))

            if dones[j]:
                # print('Done', i, self.current[i]["lens"])
                for key, value in self.current[i].items():
                    if key not in ["rewards", "max_single_R", "lens"] and self.info_keys_mode[key][1] == "mean" or key == 'avg_rewards':
                        value /= self.current[i]["lens"]
                    self.history[key].append(value)
                self.current[i] = defaultdict(float)

    def expand_buffer(self, index):
        max_index = np.max(index) + 1
        for i in range(max(max_index - self.num_workers, 0)):
            self.current.append(defaultdict(float))

    def reset_history(self):
        self.history = defaultdict(list)

    def reset_current(self):
        self.current = [
            defaultdict(float),
        ]

    def get_sync_stats(self):
        num_ep = GDict(len(self.history["rewards"])).allreduce(op="SUM", wrapper=False)

        history_min, history_max, history_sum = {}, {}, {}
        for key, value in self.history.items():
            value = np.stack(value, axis=0)
            history_min[key] = value.min(0)
            history_max[key] = value.max(0)
            history_sum[key] = value.sum(0)

        history_min = GDict(history_min).allreduce(op="MIN", wrapper=False)
        history_max = GDict(history_max).allreduce(op="MAX", wrapper=False)
        history_sum = GDict(history_sum).allreduce(op="SUM", wrapper=False)
        history_mean = {key: item / num_ep for key, item in history_sum.items()}
        return history_min, history_max, history_mean

    def get_stats_str(self):
        history_min, history_max, history_mean = self.get_sync_stats()
        ret = ""
        for key, item in self.info_keys_mode.items():
            if not (key in history_mean and item[0]) or (
                isinstance(history_mean[key], np.ndarray) and history_mean[key].size > 1
            ):
                continue
            if len(ret) > 0:
                ret += ", "
            if key == "lens":
                precision = 0
            elif key == "rewards":
                precision = 1
            else:
                precision = 2
            mean_i = num_to_str(history_mean[key], precision=precision)
            min_i = num_to_str(history_min[key], precision=precision)
            max_i = num_to_str(history_max[key], precision=precision)

            ret += f"{key}:{mean_i}"
            if item[2] == "all":
                ret += f"[{min_i}, {max_i}]"
        return ret

    def get_stats(self):
        history_min, history_max, history_mean = self.get_sync_stats()
        ret = {}
        for key in self.info_keys_mode:
            if key in history_mean:
                out_key = key if "/" in key else f"env/{key}"
                ret[f"{out_key}_mean"] = history_mean[key]
                ret[f"{out_key}_min"] = history_min[key]
                ret[f"{out_key}_max"] = history_max[key]
        return ret


def train_rl(
    config_dict,
    agent,
    rollout,
    evaluator,
    replay,
    on_policy,
    work_dir,
    expert_replay=None,
    recent_traj_replay=None,
    total_steps=1000000,
    warm_steps=0,
    resume_steps=0,
    use_policy_to_warm_up=False,
    log_episodes=0,
    print_episodes=100,
    n_steps=1,
    n_updates=1,
    n_checkpoint=None,
    n_eval=None,
    eval_cfg=None,
    ep_stats_cfg={},
    warm_up_training=False,
    warm_up_training_steps=1,
    warm_up_train_q_only=-1,
    logging_offline=False,
):
    import wandb

    wandb_logger = wandb.init(
        project="her-dreamer",
        mode="offline" if logging_offline or log_episodes == 0 else "online",
        notes=f"{os.getpid()}",
        tags=[config_dict["agent_cfg"]["type"].lower(), config_dict['env_cfg']['env_name']],
        config=config_dict,
    )

    logger = get_logger()
    agent.set_mode(mode="train")

    import torch
    from maniskill2_learn.utils.torch import get_cuda_info

    checkpoint_dir = osp.join(work_dir, "models")
    checkpoint_dir = osp.join(checkpoint_dir, str(wandb_logger.id))
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rollout is not None:
        obs = rollout.reset()
        agent.reset()
        episode_statistics = EpisodicStatistics(rollout.num_envs, **ep_stats_cfg)
        total_episodes = 0

    check_eval = EveryNSteps(n_eval)
    check_checkpoint = EveryNSteps(n_checkpoint)
    check_log = EveryNSteps(log_episodes)
    check_print = EveryNSteps(print_episodes)

    total_updates = 0

    if warm_steps > 0:
        logger.info(
            f"Begin {warm_steps} warm-up steps with {'initial policy' if use_policy_to_warm_up else 'random policy'}!"
        )
        # Randomly warm up replay buffer for model-free RL and learned model for mode-based RL
        assert not on_policy
        assert rollout is not None
        trajectories = rollout.forward_with_policy(agent if use_policy_to_warm_up else None, warm_steps)

        replay.push_batch(trajectories)


        rollout.reset()
        agent.reset()
        episode_statistics.reset_current()

        logger.info(f"Finish {warm_steps} warm-up steps!")
        if warm_up_train_q_only > 0:
            for i in range(warm_up_train_q_only):
                total_updates += 1
                training_infos = agent.update_parameters(replay, updates=total_updates, q_only=True, rollout=rollout)
                logger.info(f"Warmup pretrain q network: {i}/{warm_up_train_q_only} {dict_to_str(training_infos)}")
        if warm_up_training:
            for i in range(warm_up_training_steps):
                total_updates += 1
                training_infos = agent.update_parameters(replay, updates=total_updates, rollout=rollout, warm_up_training=True)
                ret = {"n_updates": n_updates, "total_updates":total_updates}
                training_infos.update(ret)
                wandb_logger.log(training_infos)
                
    num_steps = warm_steps + resume_steps
    total_steps += resume_steps
    begin_steps = num_steps

    begin_time = datetime.now()
    max_ETA_len = None
    logger.info("Begin training!")
    num_episodes = 0

    collect_sample_time = 0
    update_time = 0
    time_begin_episode = time.time()

    log_values = defaultdict(list)
    print_values = defaultdict(list)

    ep_stats_cfg = {}
    ep_stats_string = ""

    for iteration_id in itertools.count(1):
        ##################################
        #  Collect samples
        ##################################
        start_time = time.time()
        agent.eval()  # For things like batch norm
        trajectories = rollout.forward_with_policy(agent, n_steps, on_policy, replay)

        agent.train()

        extra_args = {}

        if recent_traj_replay is not None:
            recent_traj_replay.push_batch(trajectories)
            extra_args["recent_traj_replay"] = recent_traj_replay
        episode_statistics.push(trajectories)
        n_ep = np.sum(trajectories["episode_dones"].astype(np.int32))
        num_episodes += n_ep
        num_steps += len(trajectories["rewards"])

        collect_sample_time += time.time() - start_time
        ##################################
        # Train agent
        ##################################

        for i in range(n_updates):
            total_updates += 1
            start_time = time.time()
            if expert_replay is not None:
                extra_args["expert_replay"] = expert_replay
            training_infos = agent.update_parameters(replay, updates=total_updates, rollout=rollout, **extra_args)

            for key in training_infos:
                log_values[key].append(training_infos[key])
                # print_values[key].append(training_infos[key])
            update_time += time.time() - start_time

        if recent_traj_replay is not None:
            recent_traj_replay.reset()
        ##################################
        # Logging
        ##################################
        if log_episodes and check_log.check(num_episodes):
            log_values = {key: np.array([np.mean(log_values[key])]) for key in log_values}
            if log_episodes <= print_episodes or not print_episodes:
                ep_stats = episode_statistics.get_stats()
                ep_stats_string = episode_statistics.get_stats_str()

            episode_time = time.time() - time_begin_episode

            log_values.update(dict(episode_time=episode_time, collect_sample_time=collect_sample_time))

            log_values.update(ep_stats)
            log_values.update(dict(num_episodes=num_episodes, total_episodes=total_episodes))

            log_values["update_time"] = update_time
            log_values["total_updates"] = int(total_updates)
            log_values["buffer_size"] = len(replay)

            total_memory = GDict(get_total_memory("G", False)).allreduce(op="SUM", wrapper=False)
            log_values["memory"] = total_memory
            log_values.update(get_cuda_info(device=torch.cuda.current_device(), number_only=True))

            log_values["env/steps"] = num_steps
            log_values["env/episodes"] = num_episodes

            wandb_logger.log(log_values)

            if log_episodes <= print_episodes:
                # For online RL algorithm
                episode_statistics.reset_history()

            if on_policy:
                replay.reset()
                rollout.reset()
                agent.reset()
                episode_statistics.reset_current()

            log_values = defaultdict(list)
            # reset values
            collect_sample_time = 0
            update_time = 0
            time_begin_episode = time.time()

        ##################################
        # Printing
        ##################################

        if print_episodes and check_print.check(num_episodes):
            print_values = {}

            if log_episodes > print_episodes or not log_episodes:
                ep_stats = episode_statistics.get_stats()
                ep_stats_string = episode_statistics.get_stats_str()
            print_values.update(ep_stats)
            print_values.update(dict(num_episodes=num_episodes, total_episodes=total_episodes))

            print_values["update_time"] = update_time
            print_values["total_updates"] = int(total_updates)
            print_values["buffer_size"] = len(replay)

            total_memory = GDict(get_total_memory("G", False)).allreduce(op="SUM", wrapper=False)
            print_values["memory"] = total_memory
            print_values.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))

            print_values["samples_stats"] = ep_stats_string

            print_values = {key.split("/")[-1]: val for key, val in print_values.items()}

            percentage = f"{((num_steps - begin_steps) / (total_steps - begin_steps)) * 100:.0f}%"
            passed_time = td_format(datetime.now() - begin_time)
            ETA = td_format(
                (datetime.now() - begin_time) * max((total_steps - begin_steps) / (num_steps - begin_steps) - 1, 0)
            )
            if max_ETA_len is None:
                max_ETA_len = len(ETA)

            logger.info(
                f"{num_steps}/{total_steps}({percentage}) Passed time:{passed_time} ETA:{ETA} {dict_to_str(print_values)}"
            )
            print_values = {}
            if log_episodes > print_episodes:
                # For online RL algorithm
                episode_statistics.reset_history()
        ##################################
        # Eval agent
        ##################################
        if check_eval.check(num_steps) and is_not_null(evaluator):
            standardized_eval_step = check_eval.standard(num_steps)
            logger.info(
                f"Begin to evaluate at step: {num_steps}. "
                f"The evaluation info will be saved at eval_{standardized_eval_step}"
            )
            eval_dir = osp.join(work_dir, f"eval_{standardized_eval_step}")

            agent.eval()  # For things like batch norm
            agent.set_mode(mode="test")  # For things like obs normalization

            lens, rewards, finishes, success_percentage, success_reached = evaluator.run(agent, **eval_cfg, work_dir=eval_dir)
            # agent.recover_data_parallel()

            torch.cuda.empty_cache()
            save_eval_statistics(eval_dir, lens, rewards, finishes)
            agent.train()
            agent.set_mode(mode="train")

            eval_dict = dict(
                mean_length=np.mean(lens),
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                mean_success=np.mean(success_percentage),
                average_success=sum(success_percentage) / len(success_percentage),
                std_success=np.std(success_percentage),
                average_success_reached=sum(success_reached) / len(success_reached),
            )
            eval_dict["steps"] = num_steps
            eval_dict["episodes"] = num_episodes

            eval_dict = {"eval/" + key: eval_dict[key] for key in eval_dict}
            wandb.log(eval_dict)

        ##################################
        # Generate Checkpoint
        ##################################
        if check_checkpoint.check(num_steps):
            standardized_ckpt_step = check_checkpoint.standard(num_steps)
            model_path = osp.join(checkpoint_dir, f"model_{standardized_ckpt_step}.ckpt")
            logger.info(f"Save model at step: {num_steps}. The model will be saved at {model_path}")
            agent.to_normal()
            save_checkpoint(agent, model_path)
            agent.recover_ddp()
        ##################################
        # Exit Run
        ##################################
        if num_steps >= total_steps:
            break

    if n_checkpoint:
        model_path = osp.join(checkpoint_dir, f"model_final.ckpt")
        logger.info(f"Save checkpoint at final step {total_steps}. The model will be saved at {model_path}.")
        agent.to_normal()
        save_checkpoint(agent, model_path)
        agent.recover_ddp()
