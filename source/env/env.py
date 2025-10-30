from omni.isaac.lab.scene import InteractiveScene
import omni.isaac.lab.sim as sim_utils
from omni.isaac.kit import SimulationApp
import torch
import kornia
import omni.isaac.lab.utils.math as math_utils
from source.env.sample import (
    sample_coordinate_in_cardisian,
    sample_coordinate_in_sphere,
)
from PIL import Image
from source.env.utils import *
import glob


class env(object):
    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene: InteractiveScene,
    ):
        self.scene = scene
        self.sim = sim
        self.sim_time = 0
        self.sim_step = 0
        self.tool_pose_trans_matrix = (
            torch.eye(4).repeat(self.scene.num_envs, 1, 1).to(self.scene.device)
        )
        self.sample_camera_to_tool_trans_matrix = (
            torch.eye(4).repeat(self.scene.num_envs, 1, 1).to(self.scene.device)
        )
        self.sample_camera_world_pose_trans_matrix = (
            torch.eye(4).repeat(self.scene.num_envs, 1, 1).to(self.scene.device)
        )
        self.tool_default_root_state = (
            torch.tensor([-0.0500, -0.0350, 0.2300, 0.5000, 0.5000, 0.5000, 0.5000])
            .repeat(self.scene.num_envs, 1)
            .to(self.scene.device)
        )

    def reset(self, fixed_tool=False, env_ids=None):
        # self.sim_time = 0
        # self.sim_step = 0

        if env_ids is None:
            env_ids = [i for i in range(self.scene.num_envs)]

        if fixed_tool:
            fixed_tool_pose = self.tool_default_root_state.clone()
            fixed_tool_pose[:, :3] += self.scene.env_origins
            fixed_tool_pose_trans_matrix = torch.eye(4).repeat(
                self.scene.num_envs, 1, 1
            )
            fixed_tool_pose_trans_matrix[:, :3, :3] = math_utils.matrix_from_quat(
                fixed_tool_pose[env_ids, 3:]
            )
            fixed_tool_pose_trans_matrix[:, :3, 3] = fixed_tool_pose[:, :3]
            self.tool_pose_trans_matrix[env_ids, ...] = fixed_tool_pose_trans_matrix.to(
                device=self.scene.device
            )[env_ids, ...]
            self.sample_camera_to_tool(env_ids=env_ids)
        else:
            self.sample_camera_to_tool(env_ids=env_ids)
            self.sample_camera_world_pose(env_ids=env_ids)

        camera_pose = trans_matrix_to_pose(self.sample_camera_world_pose_trans_matrix)
        tool_pose = trans_matrix_to_pose(self.tool_pose_trans_matrix)
        self.scene["screwdriver"].write_root_pose_to_sim(tool_pose)
        self.scene["camera"].set_world_poses(camera_pose[:, :3], camera_pose[:, 3:])
        self.scene.reset(env_ids)

    def sample_camera_to_tool(self, env_ids=None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.num_envs)]
        sample_trans_matrix = sample_coordinate_in_cardisian(
            self.scene.num_envs,
            [-0.005, 0.0, 0.0],
            [0.005, 0.0, 0.02],
            [-torch.pi / 15, -torch.pi / 15, -torch.pi / 15],
            [0.0, torch.pi / 15, torch.pi / 15],
            "XYZ",
        ).to(self.scene.device)
        default_camera_pose = torch.eye(4).repeat(self.scene.num_envs, 1, 1)
        default_camera_pose[:, :3, 3] = torch.tensor([0.0, 0.05, 0.03])
        default_camera_pose[:, :3, :3] = math_utils.matrix_from_euler(
            torch.tensor([torch.pi / 2, 0.0, torch.pi]), "XYZ"
        )
        default_camera_pose = default_camera_pose.to(self.scene.device)
        self.sample_camera_to_tool_trans_matrix[env_ids, ...] = (
            default_camera_pose[env_ids, ...] @ sample_trans_matrix[env_ids, ...]
        )
        self.sample_camera_world_pose_trans_matrix[env_ids, ...] = (
            self.tool_pose_trans_matrix[env_ids, ...]
            @ self.sample_camera_to_tool_trans_matrix[env_ids, ...]
        )

    def sample_camera_world_pose(self, env_ids=None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.num_envs)]
        sample_camera_world_pose_trans_matrix = sample_coordinate_in_sphere(
            self.scene.num_envs,
            radius_range=[0.3, 0.301],
            u_range=[0.0, torch.pi / 12],
            v_range=[29 * torch.pi / 30, 31 * torch.pi / 30],
        ).to(self.scene.device)
        sample_camera_world_pose_trans_matrix[:, :3, 3] += self.scene.env_origins
        self.sample_camera_world_pose_trans_matrix[env_ids, ...] = (
            sample_camera_world_pose_trans_matrix[env_ids, ...]
        )
        self.tool_pose_trans_matrix[env_ids, ...] = (
            self.sample_camera_world_pose_trans_matrix[env_ids, ...]
            @ torch.inverse(self.sample_camera_to_tool_trans_matrix)[env_ids, ...]
        )
    

    def apply_action(self, action=None, env_ids=None):
        if env_ids is None:
            env_ids = [i for i in range(self.scene.num_envs)]
        # 0.0166 1 / 60.0
        sim_dt = self.sim.get_physics_dt()
        if action is not None:

            dT = torch.eye(4).repeat(self.scene.num_envs, 1, 1)
            dT[:, :3, 3] = action[:, :3] * sim_dt
            dT[:, :3, :3] = math_utils.matrix_from_quat(
                math_utils.quat_from_angle_axis(
                    angle=torch.norm(action[:, 3:], dim=-1) * sim_dt, axis=action[:, 3:]
                )
            )
            dT = dT.to(self.scene.device)
            self.sample_camera_world_pose_trans_matrix[env_ids, ...] = (
                self.sample_camera_world_pose_trans_matrix[env_ids, ...]
                @ dT[env_ids, ...]
            )
            self.tool_pose_trans_matrix[env_ids, ...] = (
                self.sample_camera_world_pose_trans_matrix[env_ids, ...]
                @ torch.inverse(self.sample_camera_to_tool_trans_matrix[env_ids, ...])
            )

    def step(self):
        sim_dt = self.sim.get_physics_dt()

        camera_pose = trans_matrix_to_pose(self.sample_camera_world_pose_trans_matrix)
        tool_pose = trans_matrix_to_pose(self.tool_pose_trans_matrix)
        self.scene["screwdriver"].write_root_pose_to_sim(tool_pose)
        self.scene["camera"].set_world_poses(camera_pose[:, :3], camera_pose[:, 3:])

        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.sim.render()
        self.scene.update(self.sim.get_physics_dt())

        self.sim_time += sim_dt
        self.sim_step += 1

    def get_observation(self):
        obs = {}
        state = self.scene.get_state()
        rigid_object_state = state["rigid_object"]
        obs.update(rigid_object_state)

        camera_data = self.scene["camera"].data.output
        camera_info = self.scene["camera"].data.info
        color_img = camera_data["rgb"]
        depth = camera_data["distance_to_image_plane"]
        obs.update({"depth": depth})
        obs.update({"color_img": color_img})
        class_to_color_dict = {
            v["class"]: k
            for k, v in camera_info["semantic_segmentation"]["idToLabels"].items()
        }
        mask = (
            (
                camera_data["semantic_segmentation"]
                == torch.tensor(
                    str_to_color_tuple(class_to_color_dict["screwdriver"]),
                    device=self.scene.device,
                )
            ).all(dim=-1)
            if "screwdriver" in class_to_color_dict
            else torch.full(
                camera_data["semantic_segmentation"].shape[:3],
                False,
                device=self.scene.device,
            )
        )
        switch_mask = (
            (
                camera_data["semantic_segmentation"]
                == torch.tensor(
                    str_to_color_tuple(class_to_color_dict["switch"]),
                    device=self.scene.device,
                )
            ).all(dim=-1)
            if "switch" in class_to_color_dict
            else torch.full(
                camera_data["semantic_segmentation"].shape[:3],
                False,
                device=self.scene.device,
            )
        )
        # mask = color_img * mask.unsqueeze(-1)
        obs.update({"mask": mask})
        obs.update({"switch_mask": switch_mask})
        mask = mask.cpu().numpy()[0]
        import numpy as np

        import cv2

        mask = mask.astype(np.uint8) * 255
        cv2.imwrite("mask.png", mask)
        return obs
