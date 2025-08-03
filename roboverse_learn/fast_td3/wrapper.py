from __future__ import annotations

import numpy as np
import torch
from torchvision.utils import make_grid

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import list_state_to_tensor


class FastTD3EnvWrapper:
    """
    A thin, synchronous wrapper around a MetaSim vector environment.
    Most calls mirror what `Sb3EnvWrapper` does, but the interface is
    reduced to **reset / step / render / close** and all tensors live
    on the same Torch device so that the learner never touches NumPy.
    """

    def __init__(
        self,
        scenario: ScenarioCfg,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Build the underlying MetaSim environment
        EnvironmentClass = get_sim_env_class(SimType(scenario.sim))
        self.env = EnvironmentClass(scenario)

        self.num_envs = scenario.num_envs
        self.robot = scenario.robots[0]
        self.task = scenario.task
        # ----------- initial states --------------------------------------------------
        initial_states, _, _ = get_traj(self.task, self.robot, self.env.handler)
        # Duplicate / trim list so that its length matches num_envs
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        if scenario.sim == "mjx":
            self._initial_states = list_state_to_tensor(self.env.handler, self._initial_states)
        self.env.reset(states=self._initial_states)
        states = self.env.handler.get_states()
        first_obs = self.get_humanoid_observation(states)
        self.num_obs = first_obs.shape[-1]
        self._raw_observation_cache = first_obs.clone()

        # ----------- action normalisation helpers -----------------------------------
        limits = self.robot.joint_limits  # dict: {joint_name: (low, high)}
        self.joint_names = self.env.handler.get_joint_names(self.robot.name)

        self._action_low = torch.tensor(
            [limits[j][0] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self._action_high = torch.tensor(
            [limits[j][1] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self.num_actions = self._action_low.shape[0]

        # Meta information that the learner will query
        self.max_episode_steps = self.env.handler.task.episode_length
        self.asymmetric_obs = False  # privileged critic input not used (for now)

    # ---------------------------------------------------------------------- public API
    def reset(self) -> torch.Tensor:
        """Reset *all* sub-environments and return the first observation tensor."""
        self.env.reset(states=self._initial_states)
        states = self.env.handler.get_states()
        # print(f"[DEBUG/reset] extras keys: {states.extras.keys()}")
        observation = self.get_humanoid_observation(states)
        observation = observation.to(self.device)

        # remember this first frame so we can later place it in info["observations"]["raw"]["obs"]
        self._raw_observation_cache.copy_(observation)
        return observation

    def step(self, actions: torch.Tensor):
        import time

        def _now():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            return time.time()

        t0 = _now()
        real_action = self._unnormalise_action(actions)
        states, _, terminated, truncated, _ = self.env.step_actions(real_action)
        t1 = _now()

        obs_now = self.get_humanoid_observation(states).to(self.device)
        reward_now = self.get_humanoid_reward(states).to(self.device)
        t2 = _now()

        done_flag = terminated.to(self.device, torch.bool)
        time_out_flag = truncated.to(self.device, torch.bool)

        info = {
            "time_outs": time_out_flag,
            "observations": {"raw": {"obs": self._raw_observation_cache.clone().to(self.device)}},
        }

        if (done_indices := (done_flag | time_out_flag).nonzero(as_tuple=False).squeeze(-1)).numel():
            self.env.reset(states=self._initial_states, env_ids=done_indices.tolist())
            reset_states = self.env.handler.get_states()
            reset_obs_full = self.get_humanoid_observation(reset_states).to(self.device)
            obs_now[done_indices] = reset_obs_full[done_indices]
            self._raw_observation_cache[done_indices] = reset_obs_full[done_indices]
        else:
            keep_mask = (~done_flag).unsqueeze(-1)
            self._raw_observation_cache = torch.where(keep_mask, self._raw_observation_cache, obs_now)
        t3 = _now()

        # print(
        #     f"[timing] step_actions: {(t1 - t0) * 1e3:.2f} ms | obs+reward: {(t2 - t1) * 1e3:.2f} ms | reset: {(t3 - t2) * 1e3:.2f} ms | total: {(t3 - t0) * 1e3:.2f} ms"
        # )

        return obs_now, reward_now, done_flag, info

    def render(self) -> None:
        state = self.env.handler.get_states()
        rgb_data = next(iter(state.cameras.values())).rgb
        image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        return image

    def close(self) -> None:
        self.env.close()

    # -------------------------------------------------------------------- helpers
    def get_humanoid_observation(self, states) -> torch.Tensor:
        """Flatten humanoid states and move them onto the training device."""
        return self.task.humanoid_obs_flatten_func(states).to(self.device)

    def get_humanoid_reward(self, states) -> torch.Tensor:
        total_reward = torch.zeros(self.num_envs, device=self.device)
        for reward_fn, weight in zip(self.task.reward_functions, self.task.reward_weights):
            total_reward += reward_fn(self.robot.name)(states).to(self.device) * weight
        return total_reward

    def _unnormalise_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map actions from [-1, 1] to the robot's joint-limit range."""
        return (action + 1) / 2 * (self._action_high - self._action_low) + self._action_low
