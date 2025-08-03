from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from dataclasses import dataclass

import torch
import tyro
from loguru import logger as log

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_env_class

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@dataclass
class Args:
    num_envs: int = 1
    sim: str = "isaaclab"
    robot: str = "franka"
    z_pos: float = 0.0
    decimation: int = 20


def main():
    args = tyro.cli(Args)
    scenario = ScenarioCfg(robots=[args.robot], sim=args.sim, num_envs=args.num_envs, decimation=args.decimation)

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)

    robot = scenario.robots[0]
    init_states = [{"robots": {robot.name: {"pos": [0.0, 0.0, args.z_pos]}}, "objects": {}}] * scenario.num_envs
    env.reset(states=init_states)

    joint_min = {jn: robot.joint_limits[jn][0] for jn in robot.joint_limits.keys()}
    joint_max = {jn: robot.joint_limits[jn][1] for jn in robot.joint_limits.keys()}
    step = 0
    while True:
        log.debug(f"Step {step}")
        actions = [
            {
                robot.name: {
                    "dof_pos_target": {
                        jn: (torch.rand(1).item() * (joint_max[jn] - joint_min[jn]) + joint_min[jn])
                        for jn in robot.actuators.keys()
                        if robot.actuators[jn].fully_actuated
                    }
                }
            }
            for _ in range(scenario.num_envs)
        ]
        env.step(actions)
        env.render()
        step += 1

    env.handler.close()


if __name__ == "__main__":
    main()
