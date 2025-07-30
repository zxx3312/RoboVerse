from __future__ import annotations

import os

# Disable JAX GPU memory preallocation to avoid conflicts with MuJoCo rendering.
# Docs: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco

try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np
import torch
from dm_control import mjcf
from loguru import logger as log
from mujoco import mjx

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import TaskType
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState

from .mjx_helper import (
    get_extras,
    j2t,
    pack_body_state,
    pack_root_state,
    process_entity,
    sorted_actuator_ids,
    sorted_body_ids,
    sorted_joint_info,
    t2j,
)


class MJXHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg, *, seed: int | None = None):
        super().__init__(scenario)

        self._scenario = scenario
        self._seed = seed or 0
        self._mjx_model = None
        self._robot = scenario.robots[0]
        print(f"Using robot: {self._robot.name}")
        print("Robot config class:", type(scenario.robots[0]))
        self._robot_path = self._robot.mjx_mjcf_path
        print("Robot MJCF path:", self._robot_path)
        self.cameras = []
        for camera in scenario.cameras:
            self.cameras.append(camera)

        self._renderer = None

        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32)
        self._object_root_bid_cache: dict[str, int] = {}
        self._fix_path_cache: dict[str, int] = {}
        self._gravity_compensation = not self._robot.enabled_gravity

        if self.task is not None and self.task.task_type == TaskType.LOCOMOTION:
            self.decimation = self.scenario.decimation
        else:
            log.warning("Warning: hard coding decimation to 25 for replaying trajectories")
            self.decimation = 25

    def launch(self) -> None:
        mjcf_root = self._init_mujoco()

        # set timestep
        if self.scenario.sim_params.dt is not None:
            mjcf_root.option.timestep = self.scenario.sim_params.dt

        tmp_dir = tempfile.mkdtemp()
        mjcf.export_with_assets(mjcf_root, tmp_dir)
        xml_path = next(Path(tmp_dir).glob("*.xml"))
        self._mj_model = mujoco.MjModel.from_xml_path(str(xml_path))

        self.body_names = [
            mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self._mj_model.nbody)
        ]
        self.robot_body_names = [n for n in self.body_names if n.startswith(self._mujoco_robot_name)]

        if self.cameras:
            max_w = max(c.width for c in self.cameras)
            max_h = max(c.height for c in self.cameras)
            self._renderer = mujoco.Renderer(self._mj_model, width=max_w, height=max_h)
            self._render_data = mujoco.MjData(self._mj_model)

        if not self.headless:
            self._viewer = mujoco.viewer.launch_passive(self._mj_model, self._render_data)
        log.info(f"MJXHandler launched · envs={self.num_envs}")
        log.warning("MJX currently does not support batch rendering — only env_id = 0 will be used for camera output")

    def _simulate(self) -> None:
        if self._gravity_compensation:
            self._disable_robotgravity()
        self._data = self._substep(self._mjx_model, self._data)

    def _get_states(self, env_ids: list[int] | None = None):
        """Return a structured snapshot of all robots / objects in the scene."""
        data = self._data  # mjx_env.Data  (N, …)
        N = data.qpos.shape[0]
        idx_np = np.arange(N) if env_ids is None else np.asarray(env_ids, int)
        idx = jnp.asarray(idx_np, dtype=jnp.int32)

        robots: dict[str, RobotState] = {}
        objects: dict[str, ObjectState] = {}

        # --------------------------- ROBOT ---------------------------------------
        r_cfg = self._scenario.robots[0]  # FIXME support multiple robots
        prefix = f"{r_cfg.name}/"

        # body IDs ---------------------------------------------------------------
        root_bid_r = self._object_root_bid_cache.get(
            r_cfg.name,
            sorted_body_ids(self._mjx_model, prefix)[0][0],  # fallback find
        )
        qadr_r, vadr_r = sorted_joint_info(self._mjx_model, prefix)
        aid_r = sorted_actuator_ids(self._mjx_model, prefix)
        bid_r, bnames_r = sorted_body_ids(self._mjx_model, prefix)
        root_state_r = pack_root_state(data, idx, root_bid_r)  # (N,13)
        body_state_r = pack_body_state(data, idx, jnp.asarray(bid_r))  # (N,B,13)

        robots[r_cfg.name] = RobotState(
            root_state=j2t(root_state_r),
            body_names=bnames_r,
            body_state=j2t(body_state_r),
            joint_pos=j2t(data.qpos[idx[:, None], qadr_r]),
            joint_vel=j2t(data.qvel[idx[:, None], vadr_r]),
            joint_pos_target=j2t(data.ctrl[idx[:, None], aid_r]),
            joint_vel_target=None,
            joint_effort_target=j2t(data.actuator_force[idx[:, None], aid_r]),
        )

        # -------------------------- OBJECTS -------------------------------------
        for obj in self._scenario.objects:
            prefix = f"{obj.name}/"

            root_bid_o = self._object_root_bid_cache[obj.name]
            bid_o, bnames_o = sorted_body_ids(self._mjx_model, prefix)
            root_state_o = pack_root_state(data, idx, root_bid_o)  # (N, 13)

            if isinstance(obj, ArticulationObjCfg):  # articulated
                qadr_o, vadr_o = sorted_joint_info(self._mjx_model, prefix)
                body_state_o = pack_body_state(data, idx, jnp.asarray(bid_o))  # (N,Nbody,13)
                objects[obj.name] = ObjectState(
                    root_state=j2t(root_state_o),
                    body_names=bnames_o,
                    body_state=j2t(body_state_o),
                    joint_pos=j2t(data.qpos[idx[:, None], qadr_o]),
                    joint_vel=j2t(data.qvel[idx[:, None], vadr_o]),
                )
            else:  # rigid object
                objects[obj.name] = ObjectState(
                    root_state=j2t(root_state_o),
                )

        # ===================== Cameras ===================================
        camera_states = {}
        want_any_rgb = any("rgb" in cam.data_types for cam in self.cameras)
        want_any_dep = any("depth" in cam.data_types for cam in self.cameras)

        # TODO : support multiple env_ids for rendering
        if want_any_rgb or want_any_dep:
            env_id = 0  # only env_id=0 is supported for rendering
            slice_data = jax.tree_util.tree_map(lambda x: x[env_id], data)
            mjx.get_data_into(self._render_data, self._mj_model, slice_data)
            mujoco.mj_forward(self._mj_model, self._render_data)

            for cam in self.cameras:
                cam_id = f"{cam.name}_custom"
                want_rgb = "rgb" in cam.data_types
                want_dep = "depth" in cam.data_types

                rgb_tensor, dep_tensor = None, None

                if want_rgb:
                    self._renderer.disable_depth_rendering()
                    self._renderer.update_scene(self._render_data, camera=cam_id)
                    rgb = self._renderer.render()
                    rgb_tensor = torch.from_numpy(rgb.copy()).unsqueeze(0)  # shape: (1, H, W, 3)

                if want_dep:
                    self._renderer.enable_depth_rendering()
                    self._renderer.update_scene(self._render_data, camera=cam_id)
                    depth = self._renderer.render()
                    dep_tensor = torch.from_numpy(depth.copy()).unsqueeze(0)  # shape: (1, H, W)

                camera_states[cam.name] = CameraState(
                    rgb=rgb_tensor if want_rgb else None,
                    depth=dep_tensor if want_dep else None,
                )
        # ===================== Sensors ==================================
        sensors: dict[str, torch.Tensor] = {}
        # `sens_batch` has shape (batch, total_dim)
        sens_batch = data.sensordata[idx]
        for name, sl in self._sensor_slices:
            # Convert JAX → PyTorch; result shape (batch, dim)
            sensors[name] = j2t(sens_batch[:, sl])

        extras = get_extras(self._data, self._mj_model, env_ids)
        return TensorState(objects=objects, robots=robots, cameras=camera_states, sensors=sensors, extras=extras)

    def _set_states(
        self,
        ts: TensorState,
        env_ids: list[int] | None = None,
        *,
        zero_vel: bool = True,
    ) -> None:
        # ts = list_state_to_tensor(self, ts)
        self._init_mjx_once(ts)

        data = self._data
        model = self._mjx_model

        N = data.qpos.shape[0]
        idx = jnp.arange(N, dtype=jnp.int32) if env_ids is None else jnp.asarray(env_ids, dtype=jnp.int32)
        self._ensure_id_cache(ts)

        qpos, qvel, ctrl = data.qpos, data.qvel, data.ctrl

        fixed_set = set(self._fix_path_cache.keys())

        for name, robot_state in ts.robots.items():
            qpos, qvel, ctrl = process_entity(
                name,
                robot_state,
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                idx=idx,
                model=model,
                joint_id_map=self._robot_joint_ids,
                actuator_id_map=self._robot_act_ids,
                fixed_root_names=fixed_set,
                zero_vel=zero_vel,
            )

        for name, obj_state in ts.objects.items():
            qpos, qvel, ctrl = process_entity(
                name,
                obj_state,
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                idx=idx,
                model=model,
                joint_id_map=self._object_joint_ids,
                actuator_id_map=self._object_act_ids,
                fixed_root_names=fixed_set,
                zero_vel=zero_vel,
            )

        self._data = self._data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        self._data = self._forward(self._mjx_model, self._data)

    def _ensure_id_cache(self, ts: TensorState):
        """Build joint-/actuator-ID lookup tables (one-time per handler)."""
        if hasattr(self, "_robot_joint_ids"):
            return

        mjm = self._mj_model
        mjx_m = self._mjx_model
        # print(ts.robots.keys(), ts.objects.keys(), ts.robots.items(), ts.objects.items())

        # ----- robots ----------------------------------------------------
        self._robot_joint_ids, self._robot_act_ids = {}, {}
        for rname in ts.robots:
            full_jnames = [f"{rname}/{jn}" for jn in self._get_jnames(rname, sort=True)]
            j_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, n) for n in full_jnames]
            a_ids = sorted_actuator_ids(mjx_m, f"{rname}/")
            self._robot_joint_ids[rname] = jnp.asarray(j_ids, dtype=jnp.int32)
            self._robot_act_ids[rname] = jnp.asarray(a_ids, dtype=jnp.int32)

        # print("Robot joint IDs:", self._robot_joint_ids, "Robot actuator IDs:", self._robot_act_ids)

        # ----- objects ---------------------------------------------------
        self._object_joint_ids, self._object_act_ids = {}, {}
        for oname in ts.objects:
            full_jnames = [f"{oname}/{jn}" for jn in self._get_jnames(oname, sort=True)]
            j_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, n) for n in full_jnames]
            a_ids = sorted_actuator_ids(mjx_m, f"{oname}/")
            self._object_joint_ids[oname] = jnp.asarray(j_ids, dtype=jnp.int32)
            self._object_act_ids[oname] = jnp.asarray(a_ids, dtype=jnp.int32)

    def _disable_robotgravity(self):
        """Apply m·g wrench to each robot body to emulate gravity compensation."""
        g_vec = jnp.array([0.0, 0.0, -9.81])
        body_ids = jnp.asarray([
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.robot_body_names
        ])

        mass = self._mjx_model.body_mass[body_ids]  # (B,)
        force = -g_vec * mass[:, None]  # (B, 3)

        xfrc = self._data.xfrc_applied.at[:, :, :].set(0.0)  # clear previous
        xfrc = xfrc.at[:, body_ids, 0:3].set(force)  # apply −m·g
        self._data = self._data.replace(xfrc_applied=xfrc)

    def _init_mjx_once(self, ts: TensorState) -> None:
        """One-time MJX initialisation"""
        if getattr(self, "_mjx_done", False):
            return

        def _write_fixed_body(name, root_state):
            pos = root_state[0, :3].cpu().numpy()
            quat = root_state[0, 3:7].cpu().numpy()
            full = self._fix_path_cache[name]
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, full)
            self._mj_model.body_pos[bid] = pos
            self._mj_model.body_quat[bid] = quat

        # place all pre-fixed bodies
        for name in self._fix_path_cache:
            if name in ts.objects:
                _write_fixed_body(name, ts.objects[name].root_state)
            elif name in ts.robots:
                _write_fixed_body(name, ts.robots[name].root_state)

        self._init_mjx()  # compile & allocate batched data
        self._mjx_done = True

    def set_dof_targets(
        self,
        obj_name: str,
        actions: list[Action],
    ) -> None:
        self._actions_cache = actions

        # -------- build (N, J) tensor ---------------------------------
        jnames_local = self.get_joint_names(obj_name, sort=True)
        tgt_torch = torch.stack(
            [
                torch.tensor(
                    [actions[e]["dof_pos_target"][jn] for jn in jnames_local],
                    dtype=torch.float32,
                )
                for e in range(self.num_envs)
            ],
            dim=0,
        )  # (N, J)
        tgt_jax = t2j(tgt_torch)

        # -------- id maps ---------------------------------------------
        if obj_name == self._scenario.robots[0].name:
            a_ids = self._robot_act_ids.get(obj_name)
        else:
            a_ids = self._object_act_ids.get(obj_name)

        data = self._data
        new_ctrl = data.ctrl.at[:, a_ids].set(tgt_jax)
        self._data = data.replace(ctrl=new_ctrl)

    def set_actions(
        self,
        obj_name: str,
        actions,
    ) -> None:
        self._actions_cache = actions

        tgt_jax = t2j(actions)

        # -------- id maps ---------------------------------------------
        if obj_name == self._scenario.robots[0].name:
            a_ids = self._robot_act_ids.get(obj_name)
        else:
            a_ids = self._object_act_ids.get(obj_name)

        data = self._data
        new_ctrl = data.ctrl.at[:, a_ids].set(tgt_jax)
        self._data = data.replace(ctrl=new_ctrl)

    def close(self):
        pass

    ############################################################
    ## Utils
    ############################################################
    def _init_mujoco(self) -> mjcf.RootElement:
        """Construct a self-contained MJCF tree (one robot + objects + cameras)."""
        mjcf_model = mjcf.RootElement()

        # -------------------- cameras ------------------------------------
        cam_w, cam_h = 640, 480
        for cam in self.cameras:
            # compute right/up vectors so view aligns with look_at
            dir_vec = np.array(cam.look_at) - np.array(cam.pos)
            dir_vec /= np.linalg.norm(dir_vec)
            up = np.array([0, 0, 1])
            right = np.cross(dir_vec, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, dir_vec)

            mjcf_model.worldbody.add(
                "camera",
                name=f"{cam.name}_custom",
                pos=f"{cam.pos[0]} {cam.pos[1]} {cam.pos[2]}",
                mode="fixed",
                fovy=cam.vertical_fov,
                xyaxes=f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
                resolution=f"{cam.width} {cam.height}",
            )
            cam_w, cam_h = max(cam_w, cam.width), max(cam_h, cam.height)

        # off-screen framebuffer size
        for child in mjcf_model.visual._children:
            if child.tag == "global":
                child.offwidth, child.offheight = cam_w, cam_h

        # -------------------- ground plane --------------------------------
        mjcf_model.asset.add(
            "texture",
            name="texplane",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=[0, 0, 0],
            rgb2=[1, 1, 1],
        )
        mjcf_model.asset.add(
            "material", name="matplane", reflectance="0.2", texture="texplane", texrepeat=[1, 1], texuniform=True
        )
        mjcf_model.worldbody.add(
            "geom",
            type="plane",
            pos="0 0 0",
            size="100 100 0.001",
            quat="1 0 0 0",
            condim="3",
            conaffinity="15",
            material="matplane",
        )

        # -------------------- load objects --------------------------------
        self.object_body_names, self.mj_objects = [], {}
        for obj in self.objects:
            xml = (
                mjcf.from_xml_string(self._create_primitive_xml(obj))
                if isinstance(obj, (PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg))
                else mjcf.from_path(obj.mjx_mjcf_path)
            )
            xml.model = obj.name
            attached = mjcf_model.attach(xml)
            if obj.fix_base_link:
                self._fix_path_cache[obj.name] = attached.full_identifier
            else:
                attached.add("freejoint")
            self.mj_objects[obj.name] = attached
        # -------------------- load robot ----------------------------------
        robot_xml = mjcf.from_path(self._robot_path)
        robot_attached = mjcf_model.attach(robot_xml)
        if self._robot.fix_base_link:
            self._fix_path_cache[self._robot.name] = robot_attached.full_identifier
        else:
            robot_attached.add("freejoint")
        self.mj_objects[self._robot.name] = robot_attached
        self._mujoco_robot_name = robot_attached.full_identifier

        return mjcf_model

    ############################################################
    ## Misc
    ###########################################################
    def refresh_render(self):
        """Sync env-0 to the passive viewer window."""
        if getattr(self, "_viewer", None) is None:
            return

        slice_data = jax.tree_util.tree_map(lambda x: x[0], self._data)
        mjx.get_data_into(self._render_data, self._mj_model, slice_data)
        mujoco.mj_forward(self._mj_model, self._render_data)

        self._viewer.sync()

    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            names = [self._mj_model.body(i).name for i in range(self._mj_model.nbody)]
            names = [name.split("/")[-1] for name in names if name.split("/")[0] == obj_name]
            names = [name for name in names if name != ""]
            if sort:
                names.sort()
            return names
        else:
            return []

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = [
                self._mj_model.joint(joint_id).name
                for joint_id in range(self._mj_model.njnt)
                if self._mj_model.joint(joint_id).name.startswith(obj_name + "/")
            ]
            joint_names = [name.split("/")[-1] for name in joint_names]
            joint_names = [name for name in joint_names if name != ""]
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def _get_jnames(self, obj_name: str, sort: bool = True) -> list[str]:
        joint_names = [
            self._mj_model.joint(joint_id).name
            for joint_id in range(self._mj_model.njnt)
            if self._mj_model.joint(joint_id).name.startswith(obj_name + "/")
        ]
        joint_names = [name.split("/")[-1] for name in joint_names]
        if sort:
            joint_names.sort()
        return joint_names

    # ------------------------------------------------------------
    #  MJX helpers
    # ------------------------------------------------------------
    def _build_joint_name_map(self) -> None:
        pool = self._mjx_model.names
        adr = self._mjx_model.name_jntadr
        robot_prefix = self._scenario.robots[0].name
        self._joint_name2id = {}

        for jid, a in enumerate(adr):
            raw = pool[int(a) : pool.find(b"\0", int(a))].decode()
            self._joint_name2id[raw] = jid
            self._joint_name2id[raw.split("/")[-1]] = jid
            if "/" not in raw:
                self._joint_name2id[f"{robot_prefix}/{raw}"] = jid

    def _build_root_bid_cache(self) -> None:
        for name, mjcf_body in self.mj_objects.items():
            full = mjcf_body.full_identifier
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, full)
            self._object_root_bid_cache[name] = bid + 1  # +1 because mjcf attaches a wrapper body

    def _build_sensor_cache(self) -> None:
        """
        Create a one-time lookup table that stores every sensor's
        name and its corresponding slice in `sensordata`.
        """
        self._sensor_slices: list[tuple[str, slice]] = []
        start = 0
        for i in range(self._mj_model.nsensor):
            dim = int(self._mj_model.sensor_dim[i])
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self._sensor_slices.append((name, slice(start, start + dim)))
            start += dim

    def _init_mjx(self) -> None:
        if self._mj_model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
            self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self._mjx_model = mjx.put_model(self._mj_model)
        self._build_joint_name_map()
        self._build_root_bid_cache()
        self._build_sensor_cache()

        # batched empty data
        data_single = mjx.make_data(self._mjx_model)

        def _broadcast(x):
            return jax.tree_util.tree_map(lambda y: jnp.broadcast_to(y, (self.num_envs, *y.shape)), x)

        self._data = _broadcast(data_single)

        # sub-step & forward kernel
        self._substep = self._make_substep(self.decimation)
        self._forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))

    def _make_substep(self, n_sub: int):
        def _one_env(model, data):
            def body(d, _):
                d = mjx.step(model, d)
                return d, None

            data, _ = jax.lax.scan(body, data, None, length=n_sub)
            return data

        batched = jax.vmap(_one_env, in_axes=(None, 0))
        return jax.jit(batched)

    def _create_primitive_xml(self, obj):
        if isinstance(obj, PrimitiveCubeCfg):
            size_str = f"{obj.half_size[0]} {obj.half_size[1]} {obj.half_size[2]}"
            type_str = "box"
        elif isinstance(obj, PrimitiveCylinderCfg):
            size_str = f"{obj.radius} {obj.height}"
            type_str = "cylinder"
        elif isinstance(obj, PrimitiveSphereCfg):
            size_str = f"{obj.radius}"
            type_str = "sphere"
        else:
            raise ValueError("Unknown primitive type")

        rgba_str = f"{obj.color[0]} {obj.color[1]} {obj.color[2]} 1"
        xml = f"""
        <mujoco model="{obj.name}_model">
        <worldbody>
            <body name="{type_str}_body" pos="{0} {0} {0}">
            <geom name="{type_str}_geom" type="{type_str}" size="{size_str}" rgba="{rgba_str}"/>
            </body>
        </worldbody>
        </mujoco>
        """
        return xml.strip()

    @property
    def num_envs(self) -> int:
        return self._scenario.num_envs

    @property
    def episode_length_buf(self) -> list[int]:
        return [self._episode_length_buf]

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


MJXEnv: type[EnvWrapper[MJXHandler]] = GymEnvWrapper(MJXHandler)
