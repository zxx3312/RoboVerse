import torch

# from metasim.cfg.checkers import BaseChecker
from metasim.cfg.checkers.checkers import _ReachChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.queries.site import SitePos
from metasim.utils import configclass
from metasim.utils.humanoid_reward_util import tolerance_tensor
from metasim.utils.humanoid_robot_util import (
    dof_pos_tensor,
    dof_vel_tensor,
    robot_site_pos_tensor,
    robot_velocity_tensor,
)

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class ReachReward(HumanoidBaseReward):
    def __init__(self, robot_name="h1"):
        super().__init__(robot_name)

    def __call__(self, states):
        robot = states.robots[self.robot_name]
        target = states.objects["target"]

        hand_pos = robot.extra["left_hand_pos"]  # (B, 3)
        target_pos = target.root_state[:, :3]  # (B, 3)
        dist = torch.norm(hand_pos - target_pos, dim=-1)

        torso_up_z = robot.body_state[:, 1, 8]  # torso orientation z-axis
        healthy_reward = 5.0 * torso_up_z

        qvel = robot.joint_vel  # (B, DoF)
        motion_penalty = torch.sum(qvel[:, :-1] ** 2, dim=-1)  # remove last yaw dof

        reward_close = (dist < 1.0).float() * 5.0
        reward_success = (dist < 0.05).float() * 10.0

        reward = healthy_reward - 0.0001 * motion_penalty + reward_close + reward_success
        return reward


@configclass
class ReachCfg(HumanoidTaskCfg):
    episode_length = 1000
    traj_filepath = "roboverse_data/trajs/humanoidbench/reach/v2/initial_state_v2.json"

    checker = _ReachChecker(object_name="target", robot_name="h1")
    reward_weights = [1.0]
    reward_functions = [ReachReward]

    extra_queries = {
        "left_hand_pos": SitePos("left_hand"),
        "head_pos": SitePos("head"),
    }

    objects = [
        RigidObjCfg(
            name="target",
            mjcf_path="roboverse_data/assets/humanoidbench/reach/target.xml",
            fix_base_link=True,
            physics=PhysicStateType.GEOM,
        )
    ]
