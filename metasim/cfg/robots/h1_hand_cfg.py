from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class H1HandCfg(BaseRobotCfg):
    name: str = "h1_hand"
    num_joints: int = 76
    mjcf_path: str = "roboverse_data/robots/h1_hand/mjcf/h1_hand.xml"
    mjx_mjcf_path: str = "roboverse_data/robots/h1/mjcf/mjx_h1_hand.mjx"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_yaw": BaseActuatorCfg(),
        "left_hip_roll": BaseActuatorCfg(),
        "left_hip_pitch": BaseActuatorCfg(),
        "left_knee": BaseActuatorCfg(),
        "left_ankle": BaseActuatorCfg(),
        "right_hip_yaw": BaseActuatorCfg(),
        "right_hip_roll": BaseActuatorCfg(),
        "right_hip_pitch": BaseActuatorCfg(),
        "right_knee": BaseActuatorCfg(),
        "right_ankle": BaseActuatorCfg(),
        "torso": BaseActuatorCfg(),
        "left_shoulder_pitch": BaseActuatorCfg(),
        "left_shoulder_roll": BaseActuatorCfg(),
        "left_shoulder_yaw": BaseActuatorCfg(),
        "left_elbow": BaseActuatorCfg(),
        "left_wrist_yaw": BaseActuatorCfg(),
        "right_shoulder_pitch": BaseActuatorCfg(),
        "right_shoulder_roll": BaseActuatorCfg(),
        "right_shoulder_yaw": BaseActuatorCfg(),
        "right_elbow": BaseActuatorCfg(),
        "right_wrist_yaw": BaseActuatorCfg(),
        "lh_WRJ2": BaseActuatorCfg(),
        "lh_WRJ1": BaseActuatorCfg(),
        "lh_THJ5": BaseActuatorCfg(),
        "lh_THJ4": BaseActuatorCfg(),
        "lh_THJ3": BaseActuatorCfg(),
        "lh_THJ2": BaseActuatorCfg(),
        "lh_THJ1": BaseActuatorCfg(),
        "lh_FFJ4": BaseActuatorCfg(),
        "lh_FFJ3": BaseActuatorCfg(),
        "lh_FFJ2": BaseActuatorCfg(),
        "lh_FFJ1": BaseActuatorCfg(),
        "lh_MFJ4": BaseActuatorCfg(),
        "lh_MFJ3": BaseActuatorCfg(),
        "lh_MFJ2": BaseActuatorCfg(),
        "lh_MFJ1": BaseActuatorCfg(),
        "lh_RFJ4": BaseActuatorCfg(),
        "lh_RFJ3": BaseActuatorCfg(),
        "lh_RFJ2": BaseActuatorCfg(),
        "lh_RFJ1": BaseActuatorCfg(),
        "lh_LFJ5": BaseActuatorCfg(),
        "lh_LFJ4": BaseActuatorCfg(),
        "lh_LFJ3": BaseActuatorCfg(),
        "lh_LFJ2": BaseActuatorCfg(),
        "lh_LFJ1": BaseActuatorCfg(),
        "rh_WRJ2": BaseActuatorCfg(),
        "rh_WRJ1": BaseActuatorCfg(),
        "rh_THJ5": BaseActuatorCfg(),
        "rh_THJ4": BaseActuatorCfg(),
        "rh_THJ3": BaseActuatorCfg(),
        "rh_THJ2": BaseActuatorCfg(),
        "rh_THJ1": BaseActuatorCfg(),
        "rh_FFJ4": BaseActuatorCfg(),
        "rh_FFJ3": BaseActuatorCfg(),
        "rh_FFJ2": BaseActuatorCfg(),
        "rh_FFJ1": BaseActuatorCfg(),
        "rh_MFJ4": BaseActuatorCfg(),
        "rh_MFJ3": BaseActuatorCfg(),
        "rh_MFJ2": BaseActuatorCfg(),
        "rh_MFJ1": BaseActuatorCfg(),
        "rh_RFJ4": BaseActuatorCfg(),
        "rh_RFJ3": BaseActuatorCfg(),
        "rh_RFJ2": BaseActuatorCfg(),
        "rh_RFJ1": BaseActuatorCfg(),
        "rh_LFJ5": BaseActuatorCfg(),
        "rh_LFJ4": BaseActuatorCfg(),
        "rh_LFJ3": BaseActuatorCfg(),
        "rh_LFJ2": BaseActuatorCfg(),
        "rh_LFJ1": BaseActuatorCfg(),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_yaw": (-0.43, 0.43),
        "left_hip_roll": (-0.43, 0.43),
        "left_hip_pitch": (-3.14, 2.53),
        "left_knee": (-0.26, 2.05),
        "left_ankle": (-0.87, 0.52),
        "right_hip_yaw": (-0.43, 0.43),
        "right_hip_roll": (-0.43, 0.43),
        "right_hip_pitch": (-3.14, 2.53),
        "right_knee": (-0.26, 2.05),
        "right_ankle": (-0.87, 0.52),
        "torso": (-2.35, 2.35),
        "left_shoulder_pitch": (-2.87, 2.87),
        "left_shoulder_roll": (-0.34, 3.11),
        "left_shoulder_yaw": (-1.3, 4.45),
        "left_elbow": (-1.25, 2.61),
        "left_wrist_yaw": (-0.15, 1.57),
        "right_shoulder_pitch": (-2.87, 2.87),
        "right_shoulder_roll": (-3.11, 0.34),
        "right_shoulder_yaw": (-4.45, 1.3),
        "right_elbow": (-1.25, 2.61),
        "right_wrist_yaw": (-0.15, 1.57),
    }
