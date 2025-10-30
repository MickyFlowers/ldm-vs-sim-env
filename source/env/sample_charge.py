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
from source.env.charge_scene import SceneCfg
from source.env.charge_env import env


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
    import numpy as np
    import torch
    import cv2

    count1 = -1
    count2 = 0
    while simulation_app.is_running():
        # print(vs_env.sim_step)
        if vs_env.sim_step >= 10:
            if (vs_env.sim_step - 10) % 30 == 0:
                vs_env.reset()
                vs_env.step()
                count1 += 1
                count2 = 0
            else:
                vs_env.sample_camera_to_tool()
                vs_env.step()
                count2 += 1
            obs = vs_env.get_observation()
            color_image = obs["color_img"].cpu().numpy()[0]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            mask = obs["mask"].cpu().numpy()[0]
            seg_image = np.zeros_like(color_image)
            seg_image[mask == True] = color_image[mask == True]
            if count1 >= 1000:
                break
            print(count1)
            cv2.imwrite(f"/home/cyx/project/IsaacLab/251009-charge-sim/seg/{count1:03d}-{count2:03d}.jpg", seg_image)
            cv2.imwrite(f"/home/cyx/project/IsaacLab/251009-charge-sim/img/{count1:03d}-{count2:03d}.jpg", color_image)
            
        else:
            vs_env.step()
            obs = vs_env.get_observation()
