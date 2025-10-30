import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual Servo Env")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()  # launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import sys
from xlib.sam.sam_gui import SAM

sys.path.append(".")
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from source.env.vs_scene import SceneCfg
from source.env.env import env


if __name__ == "__main__":
    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 60.0,
        device=args_cli.device,
        render=sim_utils.RenderCfg(antialiasing_mode="DLAA"),
    )
    sim = sim_utils.SimulationContext()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_render_mode(mode=sim_utils.SimulationContext.RenderMode.FULL_RENDERING)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    scene_cfg = SceneCfg(
        num_envs=args_cli.num_envs, env_spacing=2.0, lazy_sensor_update=False
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    vs_env = env(sim=sim, scene=scene)
    vs_env.reset(fixed_tool=True)
    from xlib.algo.vs.vs_controller.ibvs import IBVS
    from xlib.algo.vs.kp_matcher import RomaMatchAlgo
    from xlib.device.sensor.camera import Camera

    camera = Camera()
    camera.set_param(
        fx=602.8553466796875,
        fy=602.9547119140625,
        cx=328.829833984375,
        cy=242.49514770507812,
        width=640,
        height=480,
    )
    sam = SAM()
    controller_list = [
        IBVS(camera, kp_algo=RomaMatchAlgo) for i in range(args_cli.num_envs)
    ]
    import numpy as np
    import torch
    import cv2

    depth = np.zeros((480, 640))
    depth[:] = 0.2
    while simulation_app.is_running():
        # print(vs_env.sim_step)
        if vs_env.sim_step == 10:
            obs = vs_env.get_observation()
            print(obs.keys())
            reference_img = obs["color_img"].cpu().numpy()
            mask = np.full(reference_img.shape[:3], False)
            for i in range(args_cli.num_envs):
                _, mask_peg, _, _ = sam.segment_img(reference_img[i])
                _, mask_back, _, _ = sam.segment_img(reference_img[i])
                mask[i] = np.logical_or(mask_peg, mask_back)
                controller_list[i].update(tar_img=reference_img[i], tar_depth=depth)
            vs_env.sample_camera_world_pose(env_ids=[0])
        elif vs_env.sim_step > 10:
            obs = vs_env.get_observation()
            action_list = []
            image_list = []
            score_list = []
            current_image = obs["color_img"].cpu().numpy()
            # mask = obs["mask"].cpu().numpy()
            for i in range(args_cli.num_envs):
                controller_list[i].update(cur_img=current_image[i], cur_depth=depth)
                success, action, score, match_img = controller_list[i].calc_vel(
                    mask=mask[i], use_median_depth=True
                )
                if not success:
                    action = np.zeros((6))
                    vs_env.reset(fixed_tool=True, env_ids=[i])

                action_list.append(action)
                image_list.append(match_img)
                score_list.append(score)
            action = torch.tensor(action_list).to(scene.device)
            vs_env.apply_action(action * 3.0, env_ids=[0])
            print(vs_env.scene["camera"].data.intrinsic_matrices)
            if image_list[0] is not None:
                cv2.imshow("match_img", image_list[0])
                cv2.waitKey(1)

        if vs_env.sim_step > 50:
            vs_env.reset(fixed_tool=True, env_ids=[0])

        vs_env.step()
