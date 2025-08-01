"""Push cube on table task for humanoid robots."""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _PushChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import (
    object_position_tensor,
    robot_site_pos_tensor,
)

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg


class SuccessReward(HumanoidBaseReward):
    """Reward function for the push task: success if object is at goal."""

    def __init__(self, robot_name="h1_hand_hb"):
        super().__init__(robot_name)

    def __call__(self, states) -> torch.FloatTensor:
        """Batched success reward: 1000 if close to destination."""
        box_pos = object_position_tensor(states, "object")  # (B, 3)
        dest_pos = object_position_tensor(states, "destination")  # (B, 3)

        dgoal = torch.norm(box_pos - dest_pos, dim=-1)  # (B,)
        reward = torch.where(
            dgoal < 0.05,
            torch.full_like(dgoal, 1000.0),
            torch.zeros_like(dgoal),
        )
        return reward


class GoalDistanceReward(HumanoidBaseReward):
    """Reward based on how close the object is to the goal."""

    def __init__(self, robot_name="h1_hand_hb"):
        super().__init__(robot_name)

    def __call__(self, states) -> torch.FloatTensor:
        """Batched goal distance reward: negative distance to the destination."""
        box_pos = object_position_tensor(states, "object")
        dest_pos = object_position_tensor(states, "destination")
        reward = -torch.norm(box_pos - dest_pos, dim=-1)  # (B,)
        return reward


class HandDistanceReward(HumanoidBaseReward):
    """Reward based on how close the hand is to the object."""

    def __init__(self, robot_name="h1_hand_hb"):
        super().__init__(robot_name)

    def __call__(self, states) -> torch.FloatTensor:
        """Batched hand distance reward: negative distance to the object."""
        box_pos = object_position_tensor(states, "object")  # (B, 3)
        hand_pos = robot_site_pos_tensor(states, self.robot_name, "left_hand")  # (B, 3)
        reward = -torch.norm(box_pos - hand_pos, dim=-1)  # (B,)
        return reward


@configclass
class PushCfg(HumanoidTaskCfg):
    """Push cube on table task for humanoid robots."""

    episode_length = 500
    objects = [
        RigidObjCfg(
            name="table",
            mjcf_path="roboverse_data/assets/humanoidbench/push/table/mjcf/table.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
        RigidObjCfg(
            name="object",
            mjcf_path="roboverse_data/assets/humanoidbench/push/object/mjcf/object.xml",
            physics=PhysicStateType.GEOM,
        ),
        RigidObjCfg(
            name="destination",
            mjcf_path="roboverse_data/assets/humanoidbench/push/destination/mjcf/destination.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/push/v2/initial_state_v2.json"
    checker = _PushChecker()
    reward_weights = [1.0, 1.0, 0.1]  # αs, αt, αh
    reward_functions = [SuccessReward, GoalDistanceReward, HandDistanceReward]

    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}
