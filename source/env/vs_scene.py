import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCameraCfg
from omni.isaac.lab.utils import configclass
from scipy.spatial.transform import Rotation as R
import numpy as np

K = [
    602.8553466796875,
    0.0,
    328.829833984375,
    0.0,
    602.9547119140625,
    242.49514770507812,
    0.0,
    0.0,
    1.0,
]


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75)),
    )
    # sensors
    switch = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/switch",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/mnt/workspace/cyxovo/assets/dvs/switch.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            semantic_tags=[("class", "switch")],
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=R.from_euler("XYZ", [90, 0, 0], True).as_quat(scalar_first=True),
        ),
        collision_group=0,
    )
    screwdriver = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/screwdriver",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/mnt/workspace/cyxovo/assets/dvs/screwdriver.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            semantic_tags=[("class", "screwdriver")],
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.05, -0.035, 0.23],
            rot=R.from_euler("XYZ", [90, 90, 0], True).as_quat(scalar_first=True),
        ),
        collision_group=0,
    )
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/screwdriver/camera",
        update_period=1 / 60.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=K, width=640, height=480
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.085, 0.03),
            rot=R.from_euler("XYZ", [90, 0, 180], True).as_quat(scalar_first=True),
            convention="ros",
        ),
    )
