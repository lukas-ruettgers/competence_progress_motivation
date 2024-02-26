import cv2
from mani_skill2.evaluation.solution import BasePolicy
from maniskill2_learn.utils.meta import Config, get_logger
from maniskill2_learn.methods.builder import build_agent
from maniskill2_learn.utils.torch import load_checkpoint
from maniskill2_learn.networks.utils import (
    get_kwargs_from_shape,
    replace_placeholder_with_args,
)
from maniskill2_learn.env import get_env_info, build_env
from maniskill2_learn.env.observation_process import pcd_uniform_downsample
from maniskill2_learn.utils.data import GDict
from copy import deepcopy
import numpy as np
from gymnasium import spaces
to_np = lambda x: x.detach().cpu().numpy()


class UserPolicy(BasePolicy):
    def __init__(
        self, env_id: str, observation_space: spaces.Space, action_space: spaces.Space
    ) -> None:
        super(UserPolicy, self).__init__(env_id, observation_space, action_space)
        
        # TODO (Lukas): Replace ./ by /root/
        config_file = f'/root/user_configs/cpm/{env_id}/user_config.py'
        cfg = Config.fromfile(config_file)
        model_file = f'/root/models/{env_id}/cpm/model_final.ckpt' # Change this
        self.device = 'cuda:0'

        # Adjust env cfg.
        cfg.env_cfg["env_name"] = env_id
        # cfg.env_cfg["obs_mode"] = self.get_obs_mode(env_id)
        cfg.env_cfg["control_mode"] = self.get_control_mode(env_id)
        if 'Pick' not in env_id and hasattr(cfg.env_cfg,"n_goal_points"):
            del cfg.env_cfg.n_goal_points
        
        env_params = get_env_info(cfg.env_cfg)
        cfg.agent_cfg["env_params"] = env_params
        obs_shape = env_params["obs_shape"]
        action_shape = env_params["action_shape"]
            

        # NOTE (Lukas): Adjust state_shape to the true obs space
        """
        state_shape = 0
        agent_space = observation_space['agent']
        extra_space = observation_space['extra']
        for space in [agent_space, extra_space]:
            for key in space.keys():
                if len(space[key].shape) == 0:
                    state_shape += 1
                else:
                    state_shape += space[key].shape[0]
        
        # NOTE (Lukas): observation.space is false! env_params provides correct values.
        env_params["obs_shape"]["state"] = state_shape
        """
        if obs_shape is not None or action_shape is not None:
            replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
            cfg = replace_placeholder_with_args(cfg, **replaceable_kwargs)

        self.logger = get_logger()
        self.env = build_env(cfg.env_cfg) # NOTE (Lukas): None works here too.
        self.agent = build_agent(cfg.agent_cfg)
        self.agent = self.agent.float().to(self.device)
        
        load_checkpoint(self.agent, model_file, self.device, keys_map=None, logger=self.logger)
        self.agent.eval()
        self.agent.set_mode("test")

        self.ms2_env_name = env_id
        self.obs_mode = self.get_obs_mode(env_id)
        self.control_mode = self.get_control_mode(env_id)
        self.img_size = cfg.env_cfg.img_size
        self.n_points = cfg.env_cfg.n_points
        self.n_goal_points = cfg.env_cfg.n_goal_points if hasattr(cfg.env_cfg, "n_goal_points") else -1
        self.obs_frame = cfg.env_cfg.obs_frame

        self.env_id = env_id

    def act(self, observations):
        """Act based on the observations."""
        return self.action_space.sample()
        # obs_env = self.env.observation(observations)
        obs = self.observation(observations)
        obs = GDict(obs).unsqueeze(0).to_torch(device=self.device, dtype="float32", non_blocking=True, wrapper=False)
        action = self.agent.eval_act(obs)
        action = np.squeeze(to_np(action))
        return action

    def reset(self, observations):
        """Called at the beginning of an episode."""
        super().reset(observations)
    
    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        return "pointcloud"

    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        return "pd_ee_delta_pose"

    def observation(self, observation):
        from mani_skill2.utils.common import flatten_state_dict
        from maniskill2_learn.utils.lib3d.mani_skill2_contrib import (
            apply_pose_to_points,
            apply_pose_to_point,
        )
        from mani_skill2.utils.sapien_utils import vectorize_pose
        from sapien.core import Pose
        
        if "PushChair" in self.ms2_env_name or "MoveBucket" in self.ms2_env_name:
            try:
                right_tcp_pose = observation["extra"].pop("right_tcp_pose")
                left_tcp_pose = observation["extra"].pop("left_tcp_pose")
                observation["extra"]["tcp_pose"] = np.stack(
                    [right_tcp_pose, left_tcp_pose], axis=0
                )  # [nhands, 7], multi-arm envs
            except KeyError as e:
                print(
                    "base_pose and/or tcp_pose not found in MS1 environment observations. Please see ManiSkill2_ObsWrapper in env/wrappers.py for more details.",
                    flush=True,
                )
                raise e
        return_dict = {}

        if "state" in self.obs_mode:
            return_dict.update(observation)

        if "rgbd" in self.obs_mode or "RGBDObservationWrapper" in str(self):
            """
            Example *input* observation keys and their respective shapes ('extra' keys don't necessarily match):
            {'image':
                {'hand_camera':
                    {'rgb': (128, 128, 3), 'depth': (128, 128, 1)},
                 'base_camera':
                    {'rgb': (128, 128, 3), 'depth': (128, 128, 1)}
                },
             'agent':
                {'qpos': 9, 'qvel': 9, 'controller': {'arm': {}, 'gripper': {}}, 'base_pose': 7},
             'camera_param':
                {'base_camera': {'extrinsic_cv': (4, 4), 'cam2world_gl': (4, 4), 'intrinsic_cv': (3, 3)},
                'hand_camera': {'extrinsic_cv': (4, 4), 'cam2world_gl': (4, 4), 'intrinsic_cv': (3, 3)}}
             'extra':
                {'tcp_pose': 7, 'goal_pos': 3}}
            """

            obs = observation
            rgb, depth, segs = [], [], []
            imgs = obs["image"]

            # IMPORTANT: the order of cameras can be different across different maniskill2 versions;
            # thus we have to explicitly list out camera names to ensure that camera orders are consistent
            if "hand_camera" in imgs.keys():
                cam_names = ["hand_camera", "base_camera"]
            elif "overhead_camera_0" in imgs.keys():  # ManiSkill1 environments
                cam_names = ["overhead_camera_0", "overhead_camera_1", "overhead_camera_2"]
            else:
                raise NotImplementedError()

            # Process RGB and Depth images
            for cam_name in cam_names:
                rgb.append(imgs[cam_name]["rgb"])  # each [H, W, 3]
                depth.append(imgs[cam_name]["depth"])  # each [H, W, 1]
                if "Segmentation" in imgs[cam_name].keys():
                    segs.append(
                        imgs[cam_name]["Segmentation"]
                    )  # each [H, W, 4], last dim = [mesh_seg, actor_seg, 0, 0]
            rgb = np.concatenate(rgb, axis=2)
            assert rgb.dtype == np.uint8
            depth = np.concatenate(depth, axis=2)
            depth = depth.astype(np.float32, copy=False)
            if len(segs) > 0:
                segs = np.concatenate(segs, axis=2)
            obs.pop("image")

            # Reshape goal images, if any, for environments that use goal image, e.g. Writer-v0, Pinch-v0
            def process_4d_goal_img_to_3d(goal_img):
                if goal_img.ndim == 4:  # [K, H, W, C]
                    # for Pinch-v0, where there are multiple views of the goal
                    goal_img = np.transpose(goal_img, (1, 2, 0, 3))
                    H, W = goal_img.shape[:2]
                    goal_img = goal_img.reshape([H, W, -1])
                return goal_img

            goal_rgb = obs["extra"].pop("goal", None)
            if goal_rgb is None:
                goal_rgb = obs["extra"].pop("target_rgb", None)
            if goal_rgb is not None:
                assert goal_rgb.dtype == np.uint8
                goal_rgb = process_4d_goal_img_to_3d(goal_rgb)
                goal_rgb = cv2.resize(
                    goal_rgb.astype(np.float32),
                    rgb.shape[:2],
                    interpolation=cv2.INTER_LINEAR,
                )
                goal_rgb = goal_rgb.astype(np.uint8)
                if goal_rgb.ndim == 2:  # [H, W]
                    goal_rgb = goal_rgb[:, :, None]
                rgb = np.concatenate([rgb, goal_rgb], axis=2)
            goal_depth = obs["extra"].pop("target_depth", None)
            if goal_depth is not None:
                goal_depth = process_4d_goal_img_to_3d(goal_depth)
                goal_depth = cv2.resize(
                    goal_depth.astype(np.float32, copy=False),
                    depth.shape[:2],
                    interpolation=cv2.INTER_LINEAR,
                )
                if goal_depth.ndim == 2:
                    goal_depth = goal_depth[:, :, None]
                depth = np.concatenate([depth, goal_depth], axis=2)

            # If goal info is provided, calculate the relative position between the robot fingers' tool-center-point (tcp) and the goal
            if "tcp_pose" in obs["extra"].keys() and "goal_pos" in obs["extra"].keys():
                assert obs["extra"]["tcp_pose"].ndim <= 2
                if obs["extra"]["tcp_pose"].ndim == 2:
                    tcp_pose = obs["extra"]["tcp_pose"][0]  # take the first hand's tcp pose
                else:
                    tcp_pose = obs["extra"]["tcp_pose"]
                obs["extra"]["tcp_to_goal_pos"] = obs["extra"]["goal_pos"] - tcp_pose[:3]
            if "tcp_pose" in obs["extra"].keys():
                obs["extra"]["tcp_pose"] = obs["extra"]["tcp_pose"].reshape(-1)

            obs["extra"].pop("target_points", None)
            obs.pop("camera_param", None)

            state_obs = {}
            state_obs["extra"] = deepcopy(obs["extra"])
            state_obs["agent"] = deepcopy(obs["agent"])
            s = flatten_state_dict(
                state_obs
            )  # Other observation keys should be already ordered and such orders shouldn't change across different maniskill2 versions, so we just flatten them

            # Resize RGB and Depth images
            if self.img_size is not None and self.img_size != (
                rgb.shape[0],
                rgb.shape[1],
            ):
                rgb = cv2.resize(
                    rgb.astype(np.float32),
                    self.img_size,
                    interpolation=cv2.INTER_LINEAR,
                )
                depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_LINEAR)

            # compress rgb & depth for e.g., trajectory saving purposes
            out_dict = {
                "rgb_image": rgb.astype(np.uint8, copy=False).transpose(2, 0, 1),  # [C, H, W]
                "depth": depth.astype(np.float16, copy=False).transpose(2, 0, 1),
                "state": s,
            }
            if len(segs) > 0:
                out_dict["segs"] = segs.transpose(2, 0, 1)

            return_dict.update(out_dict)

        if "pointcloud" in self.obs_mode or "PointCloudObservationWrapper" in str(self):
            """
            Example observation keys and respective shapes ('extra' keys don't necessarily match):
            {'pointcloud':
                {'xyz': (32768, 3), 'rgb': (32768, 3)},
                # 'xyz' can also be 'xyzw' with shape (N, 4),
                # where the last dim indicates whether the point is inside the camera depth range
             'agent':
                {'qpos': 9, 'qvel': 9, 'controller': {'arm': {}, 'gripper': {}}, 'base_pose': 7},
             'extra':
                {'tcp_pose': 7, 'goal_pos': 3}
            }
            """
            # Calculate coordinate transformations that transforms poses in the world to self.obs_frame
            # These "to_origin" coordinate transformations are formally T_{self.obs_frame -> world}^{self.obs_frame}
            if self.obs_frame in ["base", "world"]:
                base_pose = observation["agent"]["base_pose"]
                p, q = base_pose[:3], base_pose[3:]
                to_origin = Pose(p=p, q=q).inv()
            elif self.obs_frame == "ee":
                tcp_poses = observation["extra"]["tcp_pose"]
                assert tcp_poses.ndim <= 2
                if tcp_poses.ndim == 2:
                    tcp_pose = tcp_poses[0]  # use the first robot hand's tcp pose as the end-effector frame
                else:
                    tcp_pose = tcp_poses  # only one robot hand
                p, q = tcp_pose[:3], tcp_pose[3:]
                to_origin = Pose(p=p, q=q).inv()
            else:
                print("Unknown Frame", self.obs_frame)
                exit(0)

            # Unify the xyz and the xyzw point cloud format
            pointcloud = observation["pointcloud"].copy()
            xyzw = pointcloud.pop("xyzw", None)
            if xyzw is not None:
                assert "xyz" not in pointcloud.keys()
                mask = xyzw[:, -1] > 0.5
                xyz = xyzw[:, :-1]
                for k in pointcloud.keys():
                    pointcloud[k] = pointcloud[k][mask]
                pointcloud["xyz"] = xyz[mask]

            # Initialize return dict
            ret = {mode: pointcloud[mode] for mode in ["xyz", "rgb"] if mode in pointcloud}

            # Process observation point cloud segmentations, if given
            if "visual_seg" in pointcloud and "actor_seg" in pointcloud:
                visual_seg = pointcloud["visual_seg"].squeeze()
                actor_seg = pointcloud["actor_seg"].squeeze()
                assert visual_seg.ndim == 1 and actor_seg.ndim == 1
                N = visual_seg.shape[0]
                ret_visual_seg = np.zeros([N, 50])  # hardcoded
                ret_visual_seg[np.arange(N), visual_seg] = 1.0
                ret_actor_seg = np.zeros([N, 50])  # hardcoded
                ret_actor_seg[np.arange(N), actor_seg] = 1.0
                ret["seg"] = np.concatenate([ret_visual_seg, ret_actor_seg], axis=-1)

            # Process observation point cloud rgb, downsample observation point cloud, and transform observation point cloud coordinates to self.obs_frame
            ret["rgb"] = ret["rgb"] / 255.0
            uniform_downsample_kwargs = {"env": self.env, "ground_eps": 1e-4, "num": self.n_points}
            pcd_uniform_downsample(ret, **uniform_downsample_kwargs)
            ret["xyz"] = apply_pose_to_points(ret["xyz"], to_origin)

            # Sample from and append the goal point cloud to the observation point cloud, if the goal point cloud is given
            goal_pcd_xyz = observation.pop("target_points", None)
            if goal_pcd_xyz is not None:
                ret_goal = {}
                ret_goal["xyz"] = goal_pcd_xyz
                for k in ret.keys():
                    if k != "xyz":
                        ret_goal[k] = np.ones_like(ret[k]) * (
                            -1
                        )  # special value to distinguish goal point cloud and observation point cloud
                pcd_uniform_downsample(ret_goal, **uniform_downsample_kwargs)
                ret_goal["xyz"] = apply_pose_to_points(ret_goal["xyz"], to_origin)
                for k in ret.keys():
                    ret[k] = np.concatenate([ret[k], ret_goal[k]], axis=0)

            # Get all kinds of position (pos) and 6D poses (pose) from the observation information
            # These pos & poses are in world frame for now (transformed later)
            obs_extra_keys = observation["extra"].keys()
            tcp_poses = None
            if "tcp_pose" in obs_extra_keys:
                tcp_poses = observation["extra"]["tcp_pose"]
                assert tcp_poses.ndim <= 2
                if tcp_poses.ndim == 1:  # single robot hand
                    tcp_poses = tcp_poses[None, :]
                tcp_poses = [
                    Pose(p=pose[:3], q=pose[3:]) for pose in tcp_poses
                ]  # [nhand] tcp poses, where nhand is the number of robot hands
            goal_pos = None
            goal_pose = None
            if "goal_pos" in obs_extra_keys:
                goal_pos = observation["extra"]["goal_pos"]
            elif "goal_pose" in obs_extra_keys:
                goal_pos = observation["extra"]["goal_pose"][:3]
                goal_pose = observation["extra"]["goal_pose"]
                goal_pose = Pose(p=goal_pose[:3], q=goal_pose[3:])
            tcp_to_goal_pos = None
            if tcp_poses is not None and goal_pos is not None:
                tcp_to_goal_pos = (
                    goal_pos - tcp_poses[0].p
                )  # use the first robot hand's tcp pose to calculate the relative position from tcp to goal

            # Sample green points near the goal and append them to the observation point cloud, which serve as visual goal indicator,
            # if self.n_goal_points is specified and the goal information if given in an environment
            # Also, transform these points to self.obs_frame
            if self.n_goal_points > 0:
                assert (
                    goal_pos is not None
                ), "n_goal_points should only be used if goal_pos(e) is contained in the environment observation"
                goal_pts_xyz = np.random.uniform(low=-1.0, high=1.0, size=(self.n_goal_points, 3)) * 0.01
                goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
                goal_pts_xyz = apply_pose_to_points(goal_pts_xyz, to_origin)
                goal_pts_rgb = np.zeros_like(goal_pts_xyz)
                goal_pts_rgb[:, 1] = 1
                ret["xyz"] = np.concatenate([ret["xyz"], goal_pts_xyz])
                ret["rgb"] = np.concatenate([ret["rgb"], goal_pts_rgb])

            # Transform all kinds of positions to self.obs_frame; these information are dependent on
            # the choice of self.obs_frame, so we name them "frame_related_states"
            frame_related_states = []
            base_info = apply_pose_to_point(observation["agent"]["base_pose"][:3], to_origin)
            frame_related_states.append(base_info)
            if tcp_poses is not None:
                for tcp_pose in tcp_poses:
                    tcp_info = apply_pose_to_point(tcp_pose.p, to_origin)
                    frame_related_states.append(tcp_info)
            if goal_pos is not None:
                goal_info = apply_pose_to_point(goal_pos, to_origin)
                frame_related_states.append(goal_info)
            if tcp_to_goal_pos is not None:
                tcp_to_goal_info = apply_pose_to_point(tcp_to_goal_pos, to_origin)
                frame_related_states.append(tcp_to_goal_info)
            if "gripper_pose" in obs_extra_keys:
                gripper_info = observation["extra"]["gripper_pose"][:3]
                gripper_info = apply_pose_to_point(gripper_info, to_origin)
                frame_related_states.append(gripper_info)
            if "joint_axis" in obs_extra_keys:  # for TurnFaucet
                joint_axis_info = to_origin.to_transformation_matrix()[:3, :3] @ observation["extra"]["joint_axis"]
                frame_related_states.append(joint_axis_info)
            if "link_pos" in obs_extra_keys:  # for TurnFaucet
                link_pos_info = apply_pose_to_point(observation["extra"]["link_pos"], to_origin)
                frame_related_states.append(link_pos_info)
            frame_related_states = np.stack(frame_related_states, axis=0)
            ret["frame_related_states"] = frame_related_states

            # Transform the goal pose and the pose from the end-effector (tool-center point, tcp)
            # to the goal into self.obs_frame; these info are also dependent on the choice of self.obs_frame,
            # so we name them "frame_goal_related_poses"
            frame_goal_related_poses = []
            if goal_pose is not None:
                pose_wrt_origin = to_origin * goal_pose
                frame_goal_related_poses.append(np.hstack([pose_wrt_origin.p, pose_wrt_origin.q]))
                if tcp_poses is not None:
                    for tcp_pose in tcp_poses:
                        pose_wrt_origin = goal_pose * tcp_pose.inv()  # T_{tcp->goal}^{world}
                        pose_wrt_origin = to_origin * pose_wrt_origin
                        frame_goal_related_poses.append(np.hstack([pose_wrt_origin.p, pose_wrt_origin.q]))
            if len(frame_goal_related_poses) > 0:
                frame_goal_related_poses = np.stack(frame_goal_related_poses, axis=0)
                ret["frame_goal_related_poses"] = frame_goal_related_poses

            # ret['to_frames'] returns frame transformation information, which is information that transforms
            # from self.obs_frame to other common frames (e.g. robot base frame, end-effector frame, goal frame)
            # Each transformation is formally T_{target_frame -> self.obs_frame}^{target_frame}
            ret["to_frames"] = []
            base_pose = observation["agent"]["base_pose"]
            base_pose_p, base_pose_q = base_pose[:3], base_pose[3:]
            base_frame = (to_origin * Pose(p=base_pose_p, q=base_pose_q)).inv().to_transformation_matrix()
            ret["to_frames"].append(base_frame)
            if tcp_poses is not None:
                for tcp_pose in tcp_poses:
                    hand_frame = (to_origin * tcp_pose).inv().to_transformation_matrix()
                    ret["to_frames"].append(hand_frame)
            if goal_pose is not None:
                goal_frame = (to_origin * goal_pose).inv().to_transformation_matrix()
                ret["to_frames"].append(goal_frame)
            ret["to_frames"] = np.stack(ret["to_frames"], axis=0)  # [Nframe, 4, 4]

            # Obtain final agent state vector, which contains robot proprioceptive information, frame-related states,
            # and other miscellaneous states (probably important) from the environment
            agent_state = np.concatenate([observation["agent"]["qpos"], observation["agent"]["qvel"]])
            if len(frame_related_states) > 0:
                agent_state = np.concatenate([agent_state, frame_related_states.flatten()])
            if len(frame_goal_related_poses) > 0:
                agent_state = np.concatenate([agent_state, frame_goal_related_poses.flatten()])
            for k in obs_extra_keys:
                if k not in [
                    "tcp_pose",
                    "goal_pos",
                    "goal_pose",
                    "tcp_to_goal_pos",
                    "tcp_to_goal_pose",
                    "joint_axis",
                    "link_pos",
                ]:
                    val = observation["extra"][k]
                    agent_state = np.concatenate(
                        [
                            agent_state,
                            val.flatten() if isinstance(val, np.ndarray) else np.array([val]),
                        ]
                    )

            ret["state"] = agent_state
            return_dict.update(ret)

        if "particles" in self.obs_mode and "particles" in observation.keys():
            obs = observation
            xyz = obs["particles"]["x"]
            vel = obs["particles"]["v"]
            state = flatten_state_dict(obs["agent"])
            ret = {
                "xyz": xyz,
                "state": state,
            }
            return_dict.update(ret)
        if len(return_dict) == 0:
            return_dict.update(observation)

        return return_dict