import argparse

from omni.isaac.lab.app import AppLauncher
from xlib.device.sensor.camera import Camera

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
from xlib.algo.vs.vs_controller.ibvs import IBVS, CNSv2

sys.path.append(".")
sys.path.append("/home/cyx/project/latent-diffusion/")
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from source.env.vs_scene import SceneCfg
from source.env.env import env
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from xlib.algo.vs.kp_matcher import RomaMatchAlgo
from xlib.algo.utils.metric import calc_ssim
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R


def transform(img, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    batch = img * 2.0 - 1.0
    batch = batch.to(device)
    return batch


def pose_to_mtx(pose):
    pos = pose[:3]
    rot = pose[3:]
    mtx = np.eye(4)
    mtx[:3, :3] = R.from_quat(rot).as_matrix()
    mtx[:3, 3] = pos
    return mtx


def calc_err(pose1_mtx, pose2_mtx):
    err_mtx = pose1_mtx @ np.linalg.inv(pose2_mtx)
    err_rot = R.from_matrix(err_mtx[:3, :3]).as_rotvec()
    print(err_mtx)
    err_rot_norm = np.linalg.norm(err_rot)
    err_vec_norm = np.linalg.norm(err_mtx[:3, 3])
    return err_rot_norm, err_vec_norm


if __name__ == "__main__":
    scale = 0.5
    camera = Camera()
    camera.set_param(
        fx=602.8553466796875 * scale,
        fy=602.9547119140625 * scale,
        cx=328.829833984375 * scale,
        cy=242.49514770507812 * scale,
    )
    # ibvs_controller = IBVS(camera=camera)
    # ibvs_controller = CNSv2(camera=camera)
    ibvs_controller = IBVS(camera=camera, kp_algo=RomaMatchAlgo, device="cuda:1")
    config = OmegaConf.load(
        "/home/cyx/project/latent-diffusion/logs/screwdriver-sim-m/2025-09-04T14-06-33-project.yaml"
    )
    model = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load(
            "/home/cyx/project/latent-diffusion/logs/screwdriver-sim-m/epoch=000632.ckpt"
        )["state_dict"],
        strict=True,
    )
    device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)
    sampler = DDIMSampler(model)
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
    # average_rot_err = 2.5551570728937465
    # average_pos_err = 0.2972824682947248
    # average_time_step = 18070
    # total_count = 44
    # success_count = 43
    average_rot_err = 0
    average_pos_err = 0
    average_time_step = 0
    total_count = 0
    success_count = 0
    while simulation_app.is_running():
        # print(vs_env.sim_step)
        if vs_env.sim_step >= 10:
            if vs_env.sim_step - 10 == 0:
                vs_env.reset()
                vs_env.step()
                obs = vs_env.get_observation()
                ref_color_image = obs["color_img"].cpu().numpy()[0]
                ref_color_image = cv2.cvtColor(ref_color_image, cv2.COLOR_RGB2BGR)
                ref_color_image = cv2.resize(
                    ref_color_image, (0, 0), fx=scale, fy=scale
                )
                ref_depth = obs["depth"].cpu().numpy()[0]
                ref_depth = cv2.resize(ref_depth, (0, 0), fx=scale, fy=scale)
                # cv2.imshow("ref_depth", ref_depth)
                # cv2.imshow("ref_color_image", ref_color_image)
                ref_color_image = transform(ref_color_image, device=device)
                tool_target_pose = vs_env.tool_pose_trans_matrix[0].cpu().numpy()

            elif vs_env.sim_step - 10 == 1:
                vs_env.reset()
                vs_env.step()
                obs = vs_env.get_observation()

                cur_color_image = obs["color_img"].cpu().numpy()[0]
                cur_color_image = cv2.cvtColor(cur_color_image, cv2.COLOR_RGB2BGR)

                cur_seg_mask = obs["mask"].cpu().numpy()[0]
                cur_seg_image = np.zeros_like(cur_color_image)
                cur_seg_image[cur_seg_mask == True] = cur_color_image[
                    cur_seg_mask == True
                ]
                cur_color_image = cv2.resize(
                    cur_color_image, (0, 0), fx=scale, fy=scale
                )
                cur_seg_image = cv2.resize(cur_seg_image, (0, 0), fx=scale, fy=scale)
                # cv2.imshow("cur_color_image", cur_color_image)
                # cv2.imshow("cur_seg_image", cur_seg_image)
                conditioning = transform(cur_seg_image, device=device)
                count = 0
                with torch.no_grad():
                    with model.ema_scope():
                        c1 = model.cond_stage_model.encode(conditioning)
                        c2 = model.cond_stage_model.encode(ref_color_image)
                        c = torch.cat([c1, c2], dim=1)
                        c = model.position_enc(c)
                        shape = (8, 30, 40)
                        sample, _ = sampler.sample(
                            S=50,
                            conditioning=c,
                            batch_size=c.shape[0],
                            shape=shape,
                            verbose=False,
                            eta=0,
                            x_T=torch.zeros((c.shape[0], *shape), device=device),
                        )
                        x_sample = model.first_stage_model.decode(sample)
                        reference_image = torch.clamp(
                            (x_sample + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        reference_image = (
                            reference_image.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                        )
                        reference_image = cv2.cvtColor(
                            reference_image, cv2.COLOR_RGB2BGR
                        ).astype(np.uint8)

            elif vs_env.sim_step < 1500:
                vs_env.step()
                obs = vs_env.get_observation()
                tool_cur_pose = vs_env.tool_pose_trans_matrix[0].cpu().numpy()

                cur_color_image = obs["color_img"].cpu().numpy()[0]
                cur_color_image = cv2.cvtColor(cur_color_image, cv2.COLOR_RGB2BGR)
                cur_color_image = cv2.resize(
                    cur_color_image, (0, 0), fx=scale, fy=scale
                )
                vs_mask = ~obs["switch_mask"].cpu().numpy()[0]
                vs_mask = cv2.resize(
                    vs_mask.astype(np.uint8), (0, 0), fx=scale, fy=scale
                ).astype(np.bool_)
                cur_color_mask_image = cur_color_image.copy()
                cur_color_mask_image[vs_mask] = 0
                cur_depth = obs["depth"].cpu().numpy()[0]
                cur_depth = cv2.resize(cur_depth, (0, 0), fx=scale, fy=scale)
                ibvs_controller.update(
                    cur_img=cur_color_mask_image,
                    tar_img=reference_image,
                    cur_depth=cur_depth,
                    tar_depth=ref_depth,
                )
                # vel, score, match_img = ibvs_controller.calc_vel(depth_hint=np.mean(ref_depth), mask=vs_mask)
                _, vel, _, match_img = ibvs_controller.calc_vel(mask=vs_mask)
                score = calc_ssim(cur_color_image, reference_image)
                print(score)
                if score is not None and score > 0.92:
                    count += 1
                    if count > 60:
                        rot_err, pos_err = calc_err(tool_cur_pose, tool_target_pose)
                        average_rot_err += rot_err
                        average_pos_err += pos_err
                        total_count += 1
                        success_count += 1
                        average_time_step += vs_env.sim_step - 10
                        print(f"total count: {total_count}")
                        print(f"success rate: {success_count / total_count}")
                        print(f"average rot err: {average_rot_err / success_count}")
                        print(f"average pos err: {average_pos_err / success_count}")
                        print(f"average time step: {average_time_step / success_count}")
                        vs_env.sim_step = 10
                        with open(f"./roma_log.txt", "a") as f:
                            f.write(
                                f"{total_count},{success_count / total_count},{average_rot_err / success_count},{average_pos_err / success_count},{average_time_step / success_count}\n"
                            )
                else:
                    count = 0
                if vel is not None:
                    vel = vel
                    vs_env.apply_action(torch.tensor(vel, device=device).unsqueeze(0))

                # if match_img is not None:
                #     cv2.imshow("match_img", match_img)
                # cv2.waitKey(1)
            else:
                total_count += 1
                if score > 0.8:
                    rot_err, pos_err = calc_err(tool_cur_pose, tool_target_pose)
                    success_count += 1
                    average_time_step += vs_env.sim_step - 10
                    average_rot_err += rot_err
                    average_pos_err += pos_err
                vs_env.sim_step = 10
                print(f"total count: {total_count}")
                print(f"success rate: {success_count / total_count}")
                print(f"average rot err: {average_rot_err / success_count}")
                print(f"average pos err: {average_pos_err / success_count}")
                print(f"average time step: {average_time_step / success_count}")
                with open(f"./roma_log.txt", "a") as f:
                    f.write(
                        f"{total_count},{success_count / total_count},{average_rot_err / success_count},{average_pos_err / success_count},{average_time_step / success_count}\n"
                    )

        else:
            vs_env.step()
            obs = vs_env.get_observation()
