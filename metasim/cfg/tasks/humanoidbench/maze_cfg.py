"""Maze task for humanoid robots.

TODO: Not Implemented because of collision detection issues.
"""

from __future__ import annotations

import torch

from metasim.cfg.checkers import _MazeChecker
from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass
from metasim.utils.humanoid_reward_util import tolerance_tensor
from metasim.utils.humanoid_robot_util import (
    actuator_forces_tensor,
    robot_position_tensor,
    robot_site_pos_tensor,
    robot_velocity_tensor,
    torso_upright_tensor,
)

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg

_STAND_HEIGHT = 1.65
_MOVE_SPEED = 2.0

_CHECKPOINTS = torch.tensor(
    [
        [0, 0, 1],
        [3, 0, 1],
        [3, 6, 1],
        [6, 6, 1],
        [6, 6, 1],
    ],
    dtype=torch.float,
)


class MazeReward(HumanoidBaseReward):
    """Reward function for the humanoid maze task.

    Computes the reward for each environment in the batch based on standing, control effort,
    movement direction, checkpoint proximity, stage progression, and wall collisions.
    """

    def __init__(self, robot_name="h1"):
        super().__init__(robot_name)
        global _STAND_HEIGHT
        if robot_name == "g1":
            _STAND_HEIGHT = 1.28
        self.checkpoints = _CHECKPOINTS

    def __call__(self, states):
        """Compute the reward for the humanoid maze task.

        Parameters
        ----------
        states : object
            The current simulation state containing robot and environment information.

        Returns:
        -------
        torch.Tensor
            The computed reward for each environment in the batch.
        """
        B = states.robots[self.robot_name].root_state.shape[0]

        device = states.robots[self.robot_name].root_state.device

        # ✅ 初始化 maze_stage
        if "maze_stage" not in states.extras:
            states.extras["maze_stage"] = torch.zeros(B, dtype=torch.long, device=device)

        head_z = robot_site_pos_tensor(states, self.robot_name, "head").select(dim=1, index=2)
        upright = torso_upright_tensor(states, self.robot_name)
        forces = actuator_forces_tensor(states, self.robot_name)
        com_v = robot_velocity_tensor(states, self.robot_name)[:, :2]  # (B,2)
        imu_xy = robot_site_pos_tensor(states, self.robot_name, "imu")[:, :2]
        pelvis_xy = robot_position_tensor(states, self.robot_name)[:, :2]

        # 1. stand_reward
        standing = tolerance_tensor(head_z, (_STAND_HEIGHT, float("inf")), margin=_STAND_HEIGHT / 4)
        uprightness = tolerance_tensor(upright, (0.9, float("inf")), margin=1.9, sigmoid="linear", value_at_margin=0)
        stand_reward = standing * uprightness

        # 2. small_control
        small_control = tolerance_tensor(forces, margin=10.0, value_at_margin=0.0, sigmoid="quadratic").mean(dim=1)
        small_control = (4 + small_control) / 5.0

        # 3. maze stage + move_direction
        stage_tensor = states.extras["maze_stage"]  # shape (B,)
        move_dir_table = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.float, device=device)
        move_dir = move_dir_table[stage_tensor]  # (B, 2)

        # 4. move reward
        v_err = com_v - move_dir * _MOVE_SPEED
        move = tolerance_tensor(v_err[:, 0], margin=1.0, sigmoid="linear") * tolerance_tensor(
            v_err[:, 1], margin=1.0, sigmoid="linear"
        )
        move = (5.0 * move + 1.0) / 6.0
        move = torch.where(stage_tensor == 4, torch.ones_like(move), move)

        # 5. checkpoint proximity
        checkpoint_xy = self.checkpoints.to(stage_tensor.device)[stage_tensor][:, :2]
        cp_dist = torch.norm(imu_xy - checkpoint_xy, dim=1)
        cp_reward = tolerance_tensor(cp_dist, margin=1.0)

        # 6. stage_convert_reward（靠近 checkpoint 更新）
        # boolean mask of those who crossed threshold
        crossed = cp_dist < 0.4
        new_stage = torch.minimum(stage_tensor + crossed.long(), torch.tensor(4, device=device))
        stage_reward = (new_stage - stage_tensor).float() * 100.0
        states.extras["maze_stage"] = new_stage  # write-back stage update ✅

        # 7. wall_collision_discount（基于 contact_flags）
        # contact_flags = states.extras["contact_with_wall"]  # (B,) float or bool
        contact_flags = torch.zeros(B, dtype=torch.bool, device=device)
        wall_discount = torch.where(contact_flags.bool(), 0.1, 1.0)

        # 8. Final reward
        reward = (0.2 * (stand_reward * small_control) + 0.4 * move + 0.4 * cp_reward) * wall_discount + stage_reward

        return reward


@configclass
class MazeCfg(HumanoidTaskCfg):
    """Maze task for humanoid robots."""

    episode_length = 1000
    objects = [
        RigidObjCfg(
            name="maze",
            mjcf_path="roboverse_data/assets/humanoidbench/maze/wall/mjcf/wall.xml",
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        ),
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/maze/v2/initial_state_v2.json"
    checker = _MazeChecker()
    reward_weights = [1.0]
    reward_functions = [MazeReward]
