n_procs_rollout = 1
horizon = 200
n_runs_in_episode = 1
# n_updates = n_procs_rollout * horizon * n_runs_in_episode / 32 * 4
n_updates = 6
# TODO (Lukas): The larger, the more older trajectories will remain in the buffer.
n_replay_horizon_orig = 100 
n_replay_horizon_imag = 30 

agent_cfg = dict(
    type="HER_DREAMER",
    n_procs_rollout=n_procs_rollout,
    imagined_replay_cfg=dict(
        type="ReplayMemory",
        # TODO (Lukas): Adjust capacity of replay_buffer.
        capacity=int(n_replay_horizon_imag * n_procs_rollout * horizon * n_runs_in_episode),
    ),
    n_updates = n_updates,
    # NOTE (Lukas): I replaced fixed scales by scales that dynamically adjust to a target scale
    goal_eval_explor_factor=0.5,
    goal_eval_extr_factor=0.5,
    goal_eval_extr_scale_init = 1.0,
    goal_eval_extr_scale_target = 0.75, # NOTE (Lukas): Influence ratio of extrinsic rewards over exploration rewards on the goal quality score. 
    goal_eval_extr_scale_adjustment = 1.5,

    new_reward_scale_init=1.0, # NOTE (Lukas): This is the overall scale on the entire new reward term.
    new_reward_scale_target = 1.0, 
    new_reward_scale_adjustment = 1.25,

    new_reward_extr_scale_init=1.0, # NOTE (Lukas): Keep the same as goal_eval_extr_scale for the first part.
    new_reward_extr_scale_target = 1.5, 
    new_reward_extr_scale_adjustment = 1.5,
    
    new_reward_dist_emb_scale_init=1.0,  
    new_reward_dist_emb_scale_target=0.3, # NOTE (Lukas): Influence ratio of embedding distance over exploration rewards on the new goal rewards.
    new_reward_dist_emb_scale_adjustment=1.5,
    
    new_reward_dist_step_scale_init=1.0,  
    new_reward_dist_step_scale_target=0.3, # NOTE (Lukas): Influence ratio of step distance over exploration rewards on the new goal rewards.
    new_reward_dist_step_scale_adjustment=1.5,

    
    
    # TODO (Lukas): Finetune decay.
    reward_memory_decay = 0.99, # NOTE (Lukas): Agent memorizes the mean reward it has perceived to better assess current rewards.
    
    # TODO (Lukas): Finetune goal quality constraints.
    goal_quality_extr_bound = 8.0, # NOTE (Lukas): How much worse than target extrinsic reward may goal trajectory be?
    goal_quality_expl_bound = 2.0, # NOTE (Lukas): How much worse than target exploration reward may goal trajectory be?
    goal_quality_rel_bound = 1.25, # NOTE (Lukas): How much better must the goal state reward be than the mean reward of the trajectory?
    goal_quality_bound_adjustment = 0.99,
    goal_eval_offset = 4,
    orig_buffer_updates=2, 
    imagine_buffer_updates=1,
    expl_decay = 0.9995,
    min_expl = 0.05,
    init_expl = 0.99,
    expl_decay_steps = 4000, # TODO (Lukas): Somehow, the decay is slower than expected, we target 1000, but might need 500.
    # TODO (Lukas): Finetune expl decay, 1.0 as initial exploration could be too high.
    ## SAC VARIABLES
    batch_size=256,  # Using multiple gpus leads to larger effective batch size, which can be crucial for SAC training
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=True,
    shared_backbone=True,
    detach_actor_feature=True,
    alpha_optim_cfg=dict(type="Adam", lr=3e-4), # Original: 3e-4
    # device='cuda:0', # TODO: Replace by 'cuda' or dynamic cuda_is_visible() term
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="TanhGaussianHead",
            log_std_bound=[-20, 2],
        ),
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(
                type="PointNet",
                feat_dim="pcd_all_channel",
                mlp_spec=[64, 128, 512],
                feature_transform=[],
            ),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                # NOTE (Lukas): For generality.
                mlp_spec=["512 + agent_shape", 256, 256, "action_shape * 2"],
                # mlp_spec=[542, 256, 256, "action_shape * 2"],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4, param_cfg={"(.*?)visual_nn(.*?)": None}), # Original: 3e-4
        # *Above removes visual_nn from actor optimizer; should only do so if shared_backbone=True and detach_actor_feature=True
        # *If either of the config options is False, then param_cfg={} should be removed, i.e. actor should also update the visual backbone.
        #   In addition, mlp_specs should be modified as well
        # *It is unknown if sharing backbone and detaching feature works well under the 3D setting. It is up to the users to figure this out.
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        num_heads=2,
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=None,
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                # NOTE (Lukas): We need to be dynamic to action_shape for assessing different controllers.
                mlp_spec=["512 + agent_shape + action_shape", 256, 256, 1],
                # mlp_spec=[550, 256, 256, 1],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4), # Original: 3e-4
    ),
    dreamer_cfg=dict(
        type="Dreamer",
        # ARCHITECTURE CONFIG
        nn_cfg=dict(
            dyn_cell="gru", # Forward dynamics model cell type
            # Hidden layers of pre- and post-processing linear NN for stochastic latent state before and after forward dynamics model
            dyn_hidden=150, # Original: 200, Mediocre: 150, Simple: 100
            dyn_deter=120, # Size of deterministic part of latent state, Original: 200, Mediocre: 120, Simple: 70
            dyn_stoch=30, # Size of stochastic part of hidden state, Original: 50, Simple: 30
            dyn_discrete=0,
            dyn_input_layers=1, # Number of layers in the preprocessing linear NN for the stochastic latent state 
            dyn_output_layers=1, # Number of layers in the postprocessing linear NN for the stochastic latent state 
            dyn_rec_depth=1, # Layer depth of forward dynamics model
            dyn_shared=False,
            # NOTE: These two activation functions are not created with build_from_cfg.
            # It is hence correct that they are specified by strings and not by dicts.
            dyn_mean_act="none",
            dyn_std_act="sigmoid2",
            dyn_min_std=0.1,
            dyn_temp_post=True,
            grad_heads=["image", "reward"],
            # Units for all Dense Heads in the world model, i.e. reward predictor, actor, and value head
            units=300, # Original: 400, Mediocre: 300, Simple: 200
            reward_layers=2, # Original: 2
            discount_layers=1, # Original: 3
            value_layers=3, # Original: 3
            actor_layers=4, # Original: 4
            act=dict(type="ELU"),
            norm_mlp=dict(type="BN1d"),
            norm_cnn=dict(type="LN2d"),
            bias="auto", # NOTE (Lukas): bias="auto" automatically checks whether norm layer requires bias.
            cnn_depth=16, # Original: 32 (1/6 compression), Mediocre: 24 (1/9 compression), Simple: 16 (1/12 compression)
            # NOTE: Original image was 4 times larger 
            encoder_kernels=[4, 3, 4, 4], # Old (for 64x64 image): 4 (all), Alternatives (for 32x32): [8,5,3,3],[3,4,4,4] 
            encoder_stride = [1, 2, 2, 2], # Old (for 64x64 image): 2 (all)
            decoder_kernels=[3, 3, 3, 4], # Old (for 64x64 image): [5, 5, 6, 6]
            decoder_stride = [1, 2, 2, 2], # Old (for 64x64 image): 2 (all)
            decoder_thin=True,
            value_head="normal",
            kl_scale="1.0",
            kl_balance="0.8",
            kl_free="1.0",
            kl_forward=False,
            pred_discount=False,
            discount_scale=1.0,
            reward_scale=1.0,
            image_scale=0.001, # NOTE <- Simon choose this value. Because image_loss > 5000, while reward_loss < 1.
            weight_decay=0.0,
        ),
        # BEHAVIOUR POLICY CONFIG
        behaviour_cfg=dict(
            discount=0.99, # NOTE (Lukas): I think 0.99 is a reasonable choice
            discount_lambda=0.95, # NOTE (Lukas): I am unsure about adequate lambda and will trust the author here.
            imag_horizon=15,
            imag_gradient="dynamics",
            imag_gradient_mix="0.1",
            imag_sample=True,
            actor_dist="trunc_normal",
            actor_entropy="1e-4",
            actor_state_entropy=0.0,
            actor_init_std=1.0,
            actor_min_std=0.1,
            actor_disc=5,
            actor_temp=0.1,
            actor_outscale=0.0,
            expl_amount=0.0, # NOTE (Lukas): Because the actions are already sampled from a normal dist, this seems redundant. 
            eval_state_mean=False,
            collect_dyn_sample=True,
            behavior_stop_grad=True,
            value_decay=0.0,
            future_entropy=False,
        ),
        # EXPLORATION CONFIG
        expl_cfg=dict(
            expl_behavior="plan2explore", 
            expl_ratio = 1.0, # NOTE (Lukas): Dreamer should always explore in our case.
            expl_until=0,
            expl_extr_scale=0.0,
            expl_intr_scale=1.0,
            disag_target="feat",
            # NOTE (Lukas): The author set disag_target="stoch". The original paper set disag_target="embed".
            # TODO (Lukas): I first try feat, since the input is feat aswell. 
            # In general, the paper's choice of embed might be more reasonable, because it's more related to the empirical data.
            disag_log=False, # Whether to wrap the mean disagreement into the log function.
            disag_pre_log_scale=10.0,
            disag_exp=True, # TODO (Lukas): Evaluate as an alternative to disag_log.
            disag_pre_exp_scale=3.0,
            disag_models=10, # Original: 10
            disag_offset=1,
            disag_layers=3, # Original: 4, Mediocre: 3, Simple: 2
            disag_units=200, # Units for each linear NN in the predictor ensemble, Original: 400, Mediocre: 200, Simple: 100
            disag_action_cond=True, 
            # NOTE (Lukas): The author set disag_action_cond=False, but action should be provided as input for predictor ensemble too.
            uncertainty_kl_scale_target=0.1, # TODO (Lukas): Finetune.
            uncertainty_kl_scale_adjustment = 1.5,
        ),
        train_cfg=dict(
            # batch_size=50,
            # batch_length=50,
            # train_every=5,
            # train_steps=1,
            # pretrain=100,
            # TODO (Lukas): Lower learning rate for submission.
            model_lr=7e-4, # Original: 3e-4, Mediocre: 7e-4, Fast: 1e-3
            value_lr=2e-4, # Original: 8e-5, Mediocre: 2e-4, Fast: 5e-4
            actor_lr=2e-4, # Original: 8e-5, Mediocre: 2e-4, Fast: 5e-4
            opt_eps=1e-5,
            grad_clip=100,
            value_grad_clip=100,
            actor_grad_clip=100,
            # dataset_size=0,
            # oversample_ends=False,
            # TODO (Lukas): Decide on slow value and actor target.
            slow_value_target=True,
            slow_actor_target=True,
            slow_target_update=100,
            slow_target_fraction=1,
            opt="adam",
        ),
        # ENVIRONMENT TODO (Lukas): (should be provided from Agent)
        env_cfg=dict(
            # task="dmc_walker_walk",
            size=[64, 64],
            envs=1,
            action_repeat=2,
            num_actions="action_shape",
            time_limit=1000,
            grayscale=False, # NOTE: Always False in our case.
            prefill=2500,
            eval_noise=0.0,
            clip_rewards="identity",
        ),
    ),
)
    
train_cfg = dict(
    on_policy=False,
    total_steps=4_000_000,
    warm_steps=8_000,  # TODO (Lukas): Finetune
    n_eval=5_000,
    n_checkpoint=40_000,
    n_steps=n_procs_rollout * horizon * n_runs_in_episode,
    n_updates=n_updates, # Check top of file.
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
    log_episodes=16,
    print_episodes=16,
)

env_cfg = dict(
    type="gym",
    env_name="PickSingleYCB-v0", # "PickCube-v0", "PickSingleEGAD-v0", "PickSingleYCB-v0", "StackCube-v0", "TurnFaucet-v0"
    ignore_dones=True,
    horizon=horizon,
    # obs_mode="pointcloud",
    obs_mode="image",
    n_points=1200,
    n_goal_points=50,
    obs_frame='ee', # NOTE (Lukas): Convert point cloud observation to ee frame. 
    # NOTE (Lukas): From the docu:
    # if the environment observation contains goal position ("goal_pos"), randomly sample 50 points near the goal position 
    # and append to the point cloud; this allows goal info to be visual; 
    # if an environment does not have goal position (e.g. in PegInsertionSide-v0), error will be raised.
    img_size=(32, 32),
    extra_wrappers=[
        dict(type="PointCloudObservationWrapper", pop_image_keys=False),
        dict(type="RGBDObservationWrapper", pop_image_keys=True),
    ],
    control_mode="pd_ee_delta_pose",
    # NOTE (Lukas): Follow suggestions from docu and change from pd_joint_delta_pos to pd_ee_delta_pose.
    # This mode seems to be better for pick and place tasks.
)


replay_cfg = dict(
    type="ReplayMemory",
    # TODO (Lukas): 
    capacity=int(n_replay_horizon_orig * n_procs_rollout * horizon * n_runs_in_episode),
)
recent_traj_replay_cfg = dict(
    type="ReplayMemory",
    # TODO (Lukas): Adjust capacity of replay_buffer.
    capacity=int(n_replay_horizon_orig * n_procs_rollout * horizon * n_runs_in_episode),
)
rollout_cfg = dict(
    type="Rollout",
    # TODO (Lukas): Check possible performance speedup for n_procs_rollout > 1 and multi_thread=True.
    num_procs=n_procs_rollout,
    with_info=True,
    multi_thread=False,
)

eval_cfg = dict(
    type="Evaluation",
    num_procs=1,
    num=20,
    use_hidden_state=False,
    save_traj=False,
    save_video=False, # TODO (Lukas): Activate if model turns out to achieve something.
    log_every_step=False,
    env_cfg=dict(ignore_dones=False),
)

# TODO (Lukas): Remove n_goal_points if env is Cabinet (possibly also for Faucet, Stack).