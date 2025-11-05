import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    # pragma: no cover
    p = None
    # Allows static analysis; runtime must have pybullet


class MultiAgentRacecarFormationEnv:

    def __init__(self,
        n_agents: int = 3,
        num_rays: int = 36,
        lidar_max_dist: float = 15.0,
        neighbor_max: int = 3,
        map_half_size: float = 12.0,
        gui: bool = False,
        render: Optional[bool] = None,
        seed: Optional[int] = None,
        frame_skip: int = 4,
        randomize_formation_on_reset: bool = True,
        fixed_offsets: Optional[List[Tuple[float, float]]] = None,
        max_steps: int = 1000,
        leader_waypoint_change_interval: int = 200,
        leader_max_vel: float = 2.0,
        leader_max_accel: float = 1.5,
        agent_max_speed: float = 2.0,
        agent_min_speed: float = 0.0,
        wheel_base: float = 0.67,
        wheel_radius: float = 0.165,
        max_lin_accel: float = 2.0,
        min_lin_accel: float = -2.0,
        max_ang_vel: float = math.pi,
        min_ang_vel: float = -math.pi,
        slot_distance_reward_scale: float = 1.5,
        velocity_mismatch_penalty_scale: float = 0.5,
        action_smooth_penalty: float = 0.05,
        proximity_penalty_scale: float = 1.5,
        sense_neighbors_radius: float = 15.0,
        draw_leader_traj: bool = True,
        leader_traj_max_len: int = 300,
        build_boundaries: bool = True,
        enable_obstacles: bool = False,
        fixed_obstacles: bool = True,
        obstacle_count: int = 0,
        leader_prediction_steps: int = 3,
        slot_smoothing_factor: float = 0.3,
        circular_obstacle_center: Optional[Tuple[float, float]] = None,
        circular_obstacle_radius: float = 5.0,
        # Reward design parameters (rebalanced for stable training)
        proximity_weight: float = 5.0,  # Increased from 3.0 for stronger distance signal
        proximity_scale: float = 2.5,
        progress_clip: float = 0.3,
        progress_weight_pos: float = 5.0,  # Reduced from 50.0 to balance with other rewards
        progress_weight_neg: float = 7.5,  # Reduced from 75.0 to balance with other rewards
        gate_distance: float = 2.0,
        gate_scale: float = 3.0,
        speed_weight: float = 2.0,
        speed_sigma: float = 1.0,
        smooth_positive_weight: float = 0.5,
        smooth_negative_slope: float = 0.25,
        smooth_threshold: float = 0.3,
        time_penalty_base: float = 0.1,  # Increased from 0.01 to be more noticeable
        time_penalty_slope: float = 0.2,  # Increased from 0.02 to be more noticeable
        time_penalty_ref_dist: float = 5.0,
        time_penalty_cap: float = 2.0,
        # Success rewards
        success_step_enabled: bool = True,
        success_dist_threshold: float = 0.6,
        success_vel_mismatch_threshold: float = 0.2,
        success_step_reward: float = 1.0,
        success_terminal_enabled: bool = True,
        success_required_steps: int = 10,
        success_terminal_reward: float = 30.0,
        terminate_on_success: bool = False,
        success_team_all: bool = True,
        ):
        assert p is not None, "pybullet is required. Please install 'pybullet' and try again."

        self.n_agents = int(n_agents)
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.num_rays = int(num_rays)
        self.lidar_max_dist = float(lidar_max_dist)
        self.neighbor_max = int(neighbor_max)
        self.map_half_size = float(map_half_size)
        self.enable_obstacles = bool(enable_obstacles)
        self.obstacle_count = int(obstacle_count)
        self.frame_skip = int(frame_skip)
        self.randomize_formation_on_reset = bool(randomize_formation_on_reset)
        self.fixed_offsets = fixed_offsets
        self.max_steps = int(max_steps)
        self.leader_waypoint_change_interval = int(leader_waypoint_change_interval)
        self.leader_max_vel = float(leader_max_vel)
        self.leader_max_accel = float(leader_max_accel)
        self.agent_max_speed = float(agent_max_speed)
        self.agent_min_speed = float(agent_min_speed)
        self.wheel_base = float(wheel_base)
        self.wheel_radius = float(wheel_radius)
        self.max_lin_accel = float(max_lin_accel)
        self.min_lin_accel = float(min_lin_accel)
        self.max_ang_vel = float(max_ang_vel)
        self.min_ang_vel = float(min_ang_vel)

        # Circular obstacle parameters
        self.circular_obstacle_center = circular_obstacle_center
        self.circular_obstacle_radius = float(circular_obstacle_radius)
        self.has_circular_obstacle = circular_obstacle_center is not None
        self.slot_distance_reward_scale = float(slot_distance_reward_scale)
        self.velocity_mismatch_penalty_scale = float(velocity_mismatch_penalty_scale)
        self.action_smooth_penalty = float(action_smooth_penalty)
        self.proximity_penalty_scale = float(proximity_penalty_scale)
        self.sense_neighbors_radius = float(sense_neighbors_radius)

        # Reward params
        self.proximity_weight = float(proximity_weight)
        self.proximity_scale = float(proximity_scale)
        self.progress_clip = float(progress_clip)
        self.progress_weight_pos = float(progress_weight_pos)
        self.progress_weight_neg = float(progress_weight_neg)
        self.gate_distance = float(gate_distance)
        self.gate_scale = float(gate_scale)
        self.speed_weight = float(speed_weight)
        self.speed_sigma = float(speed_sigma)
        self.smooth_positive_weight = float(smooth_positive_weight)
        self.smooth_negative_slope = float(smooth_negative_slope)
        self.smooth_threshold = float(smooth_threshold)
        self.time_penalty_base = float(time_penalty_base)
        self.time_penalty_slope = float(time_penalty_slope)
        self.time_penalty_ref_dist = float(time_penalty_ref_dist)
        self.time_penalty_cap = float(time_penalty_cap)
        # Success params
        self.success_step_enabled = bool(success_step_enabled)
        self.success_dist_threshold = float(success_dist_threshold)
        self.success_vel_mismatch_threshold = float(success_vel_mismatch_threshold)
        self.success_step_reward = float(success_step_reward)
        self.success_terminal_enabled = bool(success_terminal_enabled)
        self.success_required_steps = int(success_required_steps)
        self.success_terminal_reward = float(success_terminal_reward)
        self.terminate_on_success = bool(terminate_on_success)
        self.success_team_all = bool(success_team_all)

        # Prediction and smoothing parameters
        self.leader_prediction_steps = int(leader_prediction_steps)
        self.slot_smoothing_factor = float(slot_smoothing_factor)

        # RNG strategy: separate per-step and per-reset RNGs for reproducible variety
        self.base_seed: Optional[int] = int(seed) if seed is not None else None
        self.episode_idx: int = 0
        self.step_rng = np.random.default_rng(self.base_seed)
        # Per-episode RNG (for random formations, etc.), varies across episodes
        self.reset_rng = np.random.default_rng((self.base_seed + 99991) if self.base_seed is not None else None)
        # Obstacle RNG: if fixed_obstacles=True, reset to same seed before each reset to ensure fixed layout
        self.fixed_obstacles = bool(fixed_obstacles)
        self.obstacle_seed = (self.base_seed + 4242) if self.base_seed is not None else None
        self.obstacle_rng = np.random.default_rng(self.obstacle_seed)

        # obs_dim: lidar + rel_slot_pos + speed + heading_err + neighbors + leader_vel + slot_vel + dist_to_circular_obstacle + dist_change
        # Removed: collision_flag (redundant with lidar), constraint_violations (always 0 due to hard constraints)
        # Added: dist_change (historical info for better learning)
        self.obs_dim = self.num_rays + 2 + 1 + 1 + 3 * self.neighbor_max + 2 + 2 + 1 + 1
        self.action_dim = 2

        self.gui = bool(gui)
        self.render_enabled = bool(render) if render is not None else self.gui
        self.draw_leader_traj = bool(draw_leader_traj)
        self.leader_traj_max_len = int(leader_traj_max_len)
        self.build_boundaries = bool(build_boundaries)

        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set shadows BEFORE loading world to ensure consistent texture/color state
        if self.gui:
            # Enable shadows for ground texture
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            except Exception:
                pass
            # Disable grid lines for cleaner view
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_GRID, 0)
            except Exception:
                pass
            # Set top-down camera - distance scaled to 0.9x map size for closer view
            p.resetDebugVisualizerCamera(
                cameraDistance=self.map_half_size * 0.9,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0]
            )
            # Enable rendering after settings are configured
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if self.render_enabled else 0)

        self.plane_id: Optional[int] = None
        self._load_world()

        # Agent state tracking (for kinematic model)
        # Each agent maintains [x, y, v, psi] - position, velocity, heading
        # Must be initialized before _spawn_agents
        self.agent_states: List[Dict[str, float]] = [
        {"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0} for _ in range(self.n_agents)
        ]

        self.cars: List[int] = []
        self.car_info: List[dict] = []
        self._spawn_agents()
        self._spawn_obstacles()
        # Generate obstacles after spawning vehicles
        self.leader_pos = np.zeros(2, dtype=np.float32)
        self.leader_vel = np.array([self.leader_max_vel, 0.0], dtype=np.float32)
        # Initial velocity: constant speed
        self.leader_heading = 0.0
        # Current target waypoint
        self.leader_ang_vel: float = 0.0  # Leader angular velocity (rad/s)
        # Slot smoothing: store previous slot positions
        self.prev_world_slots: Optional[List[np.ndarray]] = None
        self.slot_velocities: List[np.ndarray] = [np.zeros(2, dtype=np.float32) for _ in range(self.n_agents)]

        # Visualization helpers (GUI only)
        self.leader_marker_id: Optional[int] = None
        # Target trajectory (GUI drawing cache)
        self.leader_traj_pts: List[List[float]] = []
        self.leader_line_ids: List[int] = []
        self.formation_shadow_ids: List[int] = []
        # Formation slot shadow markers (current position, gray)
        self.predicted_shadow_ids: List[int] = []
        # Formation slot shadow markers (predicted position, blue)
        self.boundary_line_ids: List[int] = []
        # Leader boundary dashed lines
        # Initialize formation offsets (must be initialized in all modes, as _get_world_slots needs it)
        self.formation_offsets = self._make_formation_offsets()

        if self.gui and self.render_enabled:
            self._init_leader_marker()
            self._init_target_marker()
            self._draw_leader_boundary()
            # Initialize formation shadows
            self._init_formation_shadows()

        self.step_count = 0
        self.prev_actions: Dict[str, np.ndarray] = {aid: np.zeros(self.action_dim, dtype=np.float32) for aid in self.agent_ids}
        self.last_lidar_scans: Dict[str, np.ndarray] = {}

        # Initialize variables used in reset() and step() - fix for uninitialized variable bug
        self.prev_distances: Dict[str, float] = {}
        self.success_counters: Dict[str, int] = {aid: 0 for aid in self.agent_ids}

        # -------------------- Public API --------------------
    def reset(self, randomize_formation: Optional[bool] = None) -> Dict[str, np.ndarray]:
        if randomize_formation is None:
            randomize_formation = self.randomize_formation_on_reset

        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Advance episode index and reseed per-episode RNG (formation, etc.)
        self.episode_idx += 1
        if self.base_seed is not None:
            self.reset_rng = np.random.default_rng(self.base_seed + 10007 * self.episode_idx)
        else:
            self.reset_rng = np.random.default_rng(None)

        # Control obstacle RNG based on fixed_obstacles:
        # - True: reset to same seed before each reset, yielding identical obstacle layout
        # - False: use different seed each episode, making obstacles vary
        if self.fixed_obstacles:
            self.obstacle_rng = np.random.default_rng(self.obstacle_seed)
        else:
            if self.base_seed is not None:
                self.obstacle_rng = np.random.default_rng(self.base_seed + 20011 * self.episode_idx)
            else:
                self.obstacle_rng = np.random.default_rng(None)

        self._load_world()

        self.cars.clear()
        self.car_info.clear()
        self._spawn_agents()
        self._spawn_obstacles()
        # Generate obstacles after spawning vehicles
        self.leader_pos = np.zeros(2, dtype=np.float32)
        # Initialize heading and first target waypoint
        self.leader_heading = 0.0
        self.leader_target = self._generate_new_waypoint()
        self.step_count = 0

        # Note: agent_states already initialized by _spawn_agents(), no need to reset
        # Reset slot smoothing related variables
        self.prev_world_slots = None
        self.slot_velocities = [np.zeros(2, dtype=np.float32) for _ in range(self.n_agents)]

        # Reset visualization (don't actively remove old debug items to avoid PyBullet warnings)
        self.leader_traj_pts = []
        self.leader_line_ids = []
        self.formation_shadow_ids = []
        self.predicted_shadow_ids = []
        if self.gui and self.render_enabled:
            self._init_leader_marker()
            self._init_target_marker()

        # Fixed formation (not random), keep slot positions consistent
        self.formation_offsets = self._make_formation_offsets()

        # Reinitialize formation shadows
        if self.gui and self.render_enabled:
            self._init_formation_shadows()

        self.prev_actions = {aid: np.zeros(self.action_dim, dtype=np.float32) for aid in self.agent_ids}
        self.last_lidar_scans = {}
        self.prev_distances = {}
        self.success_counters = {aid: 0 for aid in self.agent_ids}
        # Clear distance records, used for progress reward calculation

        if self.gui:
            # Re-enable rendering after all objects are loaded
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if self.render_enabled else 0)
            # Ensure shadows enabled and grid disabled (already set before loading world)
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_GRID, 0)
            except Exception:
                pass
            # Reset camera with closer view after reset
            p.resetDebugVisualizerCamera(
                cameraDistance=self.map_half_size * 0.9,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0]
            )

        return self._get_obs()

    def step(self, action_dict: Dict[str, np.ndarray]):
        self._update_leader()

        # Apply actions for all agents and update previous actions
        for i, aid in enumerate(self.agent_ids):
            act = np.asarray(action_dict.get(aid, np.zeros(self.action_dim, dtype=np.float32)), dtype=np.float32)
            act = np.clip(act, -1.0, 1.0)
            self._apply_action(self.cars[i], self.car_info[i], act, agent_idx=i)
            self.prev_actions[aid] = act

        # Get observations and compute rewards
        obs = self._get_obs()
        reward, info = self._compute_reward_and_info(action_dict)

        # Ensure all agents have reward and info entries with required keys
        for aid in self.agent_ids:
            if aid not in reward:
                reward[aid] = 0.0
            if aid not in info:
                info[aid] = {}
            # Ensure required keys exist
            if "slot_distance" not in info[aid]:
                info[aid]["slot_distance"] = 0.0
            if "collision" not in info[aid]:
                info[aid]["collision"] = False
            if "vel_mismatch" not in info[aid]:
                info[aid]["vel_mismatch"] = 0.0

        # Check for termination conditions
        self.step_count += 1
        time_limit = (self.step_count >= self.max_steps)
        collision_any = any(info[aid].get("collision", False) for aid in self.agent_ids)
        done_any = time_limit or collision_any

        # Update info and done dictionaries
        done = {}
        # Include success termination if configured
        success_any = any(info.get(aid, {}).get('success_terminal', False) for aid in self.agent_ids)
        if self.terminate_on_success and success_any:
            done_any = True
        for aid in self.agent_ids:
            info[aid]["time_limit"] = time_limit
            info[aid]["success_terminate"] = bool(self.terminate_on_success and success_any)
            done[aid] = done_any

        return obs, reward, done, info

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass

    def set_render(self, enabled: bool):
        self.render_enabled = bool(enabled)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if self.render_enabled else 0)
            # Enable shadows for ground texture
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            except Exception:
                pass

    # -------------------- Helpers --------------------
    def get_dim_info(self) -> Dict[str, Tuple[int, int]]:
        return {aid: (self.obs_dim, self.action_dim) for aid in self.agent_ids}

    def _load_world(self):
        self.plane_id = p.loadURDF("plane.urdf")
        # Keep default plane texture for visual appeal
        # Don't modify visual properties to preserve texture
        
        if self.build_boundaries:
            self._build_boundaries()
        # Obstacle generation delayed until after vehicle spawning to avoid duplication

    def _spawn_obstacles(self):
        self.obstacles: List[int] = []

        # Only generate obstacles when enable_obstacles is True
        if not self.enable_obstacles:
            return

        # Collect current vehicle positions
        car_positions = []
        for cid in self.cars:
            pos = p.getBasePositionAndOrientation(cid)[0][:2]
            car_positions.append(pos)

        safe_distance = 2.0
        for _ in range(self.obstacle_count):
            sx = float(self.obstacle_rng.uniform(0.5, 2.0))
            sy = float(self.obstacle_rng.uniform(0.5, 2.0))
            sz = float(self.obstacle_rng.uniform(0.5, 1.5))

            max_attempts = 100
            found = False
            px, py = 0.0, 0.0
            for _attempt in range(max_attempts):
                px = float(self.obstacle_rng.uniform(-self.map_half_size * 0.8, self.map_half_size * 0.8))
                py = float(self.obstacle_rng.uniform(-self.map_half_size * 0.8, self.map_half_size * 0.8))

                too_close = False
                for car_pos in car_positions:
                    dist = math.sqrt((px - car_pos[0])**2 + (py - car_pos[1])**2)
                    if dist < safe_distance:
                        too_close = True
                        break
                if not too_close:
                    found = True
                    break

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx / 2, sy / 2, sz / 2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx / 2, sy / 2, sz / 2], rgbaColor=[0.6, 0.3, 0.2, 1])
            pz = sz / 2
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[px, py, pz],
            )
            self.obstacles.append(bid)

    def _build_boundaries(self):
        # Record boundary IDs and set thickness - taller and thicker walls for visibility
        self.boundary_ids: List[int] = []
        thickness = 0.5  # Increased thickness for visibility
        height = 2.5     # Increased height for visibility
        half_x = self.map_half_size
        half_y = self.map_half_size
        # Long walls along Y (left/right) - dark gray
        col_lr = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness / 2, half_y + 1.0, height / 2])
        vis_lr = p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness / 2, half_y + 1.0, height / 2],
                                      rgbaColor=[0.2, 0.2, 0.2, 1])
        for x in (-half_x - thickness / 2, half_x + thickness / 2):
            bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_lr, baseVisualShapeIndex=vis_lr,
                                   basePosition=[x, 0.0, height / 2])
            self.boundary_ids.append(bid)
        # Long walls along X (bottom/top) - dark gray
        col_bt = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_x + 1.0, thickness / 2, height / 2])
        vis_bt = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_x + 1.0, thickness / 2, height / 2],
                                      rgbaColor=[0.2, 0.2, 0.2, 1])
        for y in (-half_y - thickness / 2, half_y + thickness / 2):
            bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_bt, baseVisualShapeIndex=vis_bt,
                                   basePosition=[0.0, y, height / 2])
            self.boundary_ids.append(bid)

    def _spawn_agents(self):
        # Larger initial radius, giving agents more safe space, reducing warmup collisions
        start_radius = min(5.0, self.map_half_size * 0.3)  # Increased from 3m to 5m
        # Ensure agents are not spawned inside circular obstacle
        if self.has_circular_obstacle:
            x_o, y_o = self.circular_obstacle_center
            dist_to_center = math.sqrt(x_o**2 + y_o**2)
            # If obstacle center near origin, need larger initial radius
            min_safe_radius = self.circular_obstacle_radius + 2.0  # Obstacle radius + 2m safety margin
            if dist_to_center < min_safe_radius:
                start_radius = max(start_radius, min_safe_radius)

        angles = np.linspace(0, 2 * math.pi, self.n_agents, endpoint=False)
        for i in range(self.n_agents):
            x = start_radius * math.cos(angles[i])
            y = start_radius * math.sin(angles[i])
            yaw = angles[i] + math.pi
            # Load Racecar robot
            cid = p.loadURDF("racecar/racecar.urdf", [x, y, 0.1],
                             p.getQuaternionFromEuler([0, 0, yaw]))
            info = self._get_joint_info(cid)
            self.cars.append(cid)
            self.car_info.append(info)

            # Initialize agent_states
            self.agent_states[i]["x"] = float(x)
            self.agent_states[i]["y"] = float(y)
            self.agent_states[i]["v"] = 0.0
            self.agent_states[i]["psi"] = float(yaw)

    def _get_joint_info(self, car_id: int) -> dict:
        joint_info = {}
        # For Husky robot: has 4 wheels (front_left, front_right, rear_left, rear_right)
        # We need to control left and right wheels
        left_wheels = []
        right_wheels = []

        for j in range(p.getNumJoints(car_id)):
            info = p.getJointInfo(car_id, j)
            joint_name = info[1].decode('utf-8').lower()
            # Support multiple robot joint naming conventions
            if 'left' in joint_name and 'wheel' in joint_name:
                left_wheels.append(j)
            elif 'right' in joint_name and 'wheel' in joint_name:
                right_wheels.append(j)

        # Store all left and right wheel indices
        joint_info['left_wheels'] = left_wheels
        joint_info['right_wheels'] = right_wheels

        return joint_info

    def _make_formation_offsets(self) -> List[np.ndarray]:
        # Default fixed formation offsets - leader at formation center
        if self.fixed_offsets is not None:
            assert len(self.fixed_offsets) >= self.n_agents, "fixed_offsets length < n_agents"
            return [np.array(self.fixed_offsets[i], dtype=np.float32) for i in range(self.n_agents)]

        # Default lateral spacing, leader at center, other vehicles arranged around leader
        spacing = 3.0
        offsets = []

        if self.n_agents == 1:
            # Single agent, at center
            offsets.append(np.array([0.0, 0.0], dtype=np.float32))
        elif self.n_agents == 2:
            # Two agents, left-right symmetric
            offsets.append(np.array([0.0, -spacing/2], dtype=np.float32))
            offsets.append(np.array([0.0, spacing/2], dtype=np.float32))
        elif self.n_agents == 3:
            # Three agents, horizontal line, center is leader position
            offsets.append(np.array([0.0, -spacing], dtype=np.float32))  # Left
            offsets.append(np.array([0.0, 0.0], dtype=np.float32))        # Center (leader)
            offsets.append(np.array([0.0, spacing], dtype=np.float32))    # Right
        else:
            # Multiple agents, arranged in circle or polygon around leader
            # Leader at center (0,0), other vehicles evenly distributed on circumference
            radius = spacing * 1.5
            for i in range(self.n_agents):
                angle = 2 * math.pi * i / self.n_agents
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                offsets.append(np.array([x, y], dtype=np.float32))

        return offsets

    def _random_formation_offsets(self) -> List[np.ndarray]:
        # Random formation - leader at center, other vehicles randomly distributed around
        offsets = []
        for _ in range(self.n_agents):
            r = float(self.reset_rng.uniform(2.5, 6.0))
            theta = float(self.reset_rng.uniform(0, 2 * math.pi))
            # Random distribution over full circle
            offsets.append(np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32))
        return offsets

    def _generate_new_waypoint(self) -> np.ndarray:
        # Generate a new leader waypoint within bounds, avoiding obstacles
        # Leader activity area: 70% of map range
        bound = self.map_half_size * 0.7
        max_attempts = 50

        for _ in range(max_attempts):
            # Randomly generate target point
            x = float(self.step_rng.uniform(-bound, bound))
            y = float(self.step_rng.uniform(-bound, bound))

            # Check if outside circular obstacle safe distance
            if self.has_circular_obstacle:
                x_o, y_o = self.circular_obstacle_center
                dist = math.sqrt((x - x_o)**2 + (y - y_o)**2)
                # Maintain safe distance (radius + 5m)
                if dist < self.circular_obstacle_radius + 5.0:
                    continue

            return np.array([x, y], dtype=np.float32)

        # If all attempts fail, return origin
        return np.zeros(2, dtype=np.float32)

    def _update_leader(self):
        # Update virtual leader - straight motion, only turns when switching waypoints (follows kinematic model)
        dt = (self.frame_skip / 240.0)
        prev_heading = float(self.leader_heading)

        # Calculate direction and distance to target
        to_target = self.leader_target - self.leader_pos
        dist_to_target = float(np.linalg.norm(to_target))

        # Reached target (within 0.5m), generate new target immediately
        if dist_to_target < 0.5:
            self.leader_target = self._generate_new_waypoint()
            to_target = self.leader_target - self.leader_pos
            dist_to_target = float(np.linalg.norm(to_target))

        if dist_to_target > 0.1:
            # Target heading
            target_heading = float(math.atan2(to_target[1], to_target[0]))

            # Calculate heading error (normalized to [-π, π])
            heading_error = target_heading - self.leader_heading
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi

            # Determine if turn needed: only turn when heading error exceeds threshold
            turn_threshold = 0.05  # About 3 degrees
            if abs(heading_error) > turn_threshold:
                # Turning phase: rotate in place or slow turn
                # Angular velocity control (proportional control)
                Kp_omega = 2.0
                desired_omega = Kp_omega * heading_error
                desired_omega = np.clip(desired_omega, -self.max_ang_vel, self.max_ang_vel)

                # Slow down during turn
                current_speed = float(np.linalg.norm(self.leader_vel))
                desired_speed = min(current_speed * 0.5, 0.5)  # Max 0.5m/s

                # Smooth speed transition
                speed_error = desired_speed - current_speed
                u = np.clip(speed_error / dt, -self.leader_max_accel, self.leader_max_accel)
                new_speed = current_speed + u * dt
                new_speed = np.clip(new_speed, 0.0, self.leader_max_vel)

                # Update heading
                self.leader_heading = self.leader_heading + desired_omega * dt
                while self.leader_heading > math.pi:
                    self.leader_heading -= 2 * math.pi
                while self.leader_heading < -math.pi:
                    self.leader_heading += 2 * math.pi

                # Calculate angular velocity
                ang_diff = self.leader_heading - prev_heading
                if ang_diff > math.pi:
                    ang_diff -= 2 * math.pi
                elif ang_diff < -math.pi:
                    ang_diff += 2 * math.pi
                self.leader_ang_vel = float(ang_diff / dt)

                # Velocity vector
                self.leader_vel = new_speed * np.array([
                    math.cos(self.leader_heading),
                    math.sin(self.leader_heading)
                ], dtype=np.float32)
            else:
                # Straight motion phase: heading aligned, constant speed forward
                self.leader_ang_vel = 0.0  # Angular velocity is zero

                # Adjust speed based on distance
                if dist_to_target > 5.0:
                    desired_speed = self.leader_max_vel
                elif dist_to_target > 2.0:
                    desired_speed = self.leader_max_vel * 0.8
                else:
                    # Slow down when approaching target
                    desired_speed = max(0.5, self.leader_max_vel * (dist_to_target / 2.0))

                # Smooth speed transition
                current_speed = float(np.linalg.norm(self.leader_vel))
                speed_error = desired_speed - current_speed
                u = np.clip(speed_error / dt, -self.leader_max_accel, self.leader_max_accel)
                new_speed = current_speed + u * dt
                new_speed = np.clip(new_speed, 0.0, self.leader_max_vel)

                # Velocity vector (maintain current heading)
                self.leader_vel = new_speed * np.array([
                    math.cos(self.leader_heading),
                    math.sin(self.leader_heading)
                ], dtype=np.float32)
        else:
            # Reached target, stop
            self.leader_vel = np.zeros(2, dtype=np.float32)
            self.leader_ang_vel = 0.0

        # Position update (kinematic equation)
        new_pos = self.leader_pos + self.leader_vel * dt

        # Boundary check (hard constraint) - leader activity range is 70% of map
        bound = self.map_half_size * 0.7
        new_pos = np.clip(new_pos, -bound, bound)

        self.leader_pos = new_pos.astype(np.float32)

        # Update visualization
        if self.gui and self.render_enabled:
            self._update_leader_visuals()

    def _init_leader_marker(self):
        # Leader is virtual trajectory, no physical marker
        self.leader_marker_id = None

    def _init_target_marker(self):
        # Initialize target marker (green sphere)
        try:
            # Slightly larger than leader
            r = 0.3
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.2, 1.0, 0.2, 0.8])
            # Green, semi-transparent
            self.target_marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
            basePosition=[float(self.leader_target[0]), float(self.leader_target[1]), 0.2])
        except Exception:
            self.target_marker_id = None

    def _init_formation_shadows(self):
        # Initialize formation slot shadows (faint circular markers)
        try:
            self.formation_shadow_ids = []
            self.predicted_shadow_ids = []

            # Current position slots (gray) - use current position for visualization
            world_slots = self._get_world_slots(use_prediction=False)
            for i in range(self.n_agents):
                slot_pos = world_slots[i]
                r = 0.3  # Radius slightly smaller than actual robot
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=0.01,
                                         rgbaColor=[0.2, 0.2, 0.2, 0.5])  # Dark gray
                shadow_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                             basePosition=[float(slot_pos[0]), float(slot_pos[1]), 0.01])
                self.formation_shadow_ids.append(shadow_id)

            # Predicted position slots (blue) - use prediction for training
            predicted_slots = self._get_world_slots(use_prediction=True)
            for i in range(self.n_agents):
                slot_pos = predicted_slots[i]
                r = 0.3
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=0.01,
                                         rgbaColor=[0.2, 0.4, 0.8, 0.4])  # Blue, more transparent
                shadow_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                             basePosition=[float(slot_pos[0]), float(slot_pos[1]), 0.02])
                self.predicted_shadow_ids.append(shadow_id)
        except Exception:
            self.formation_shadow_ids = []
            self.predicted_shadow_ids = []

    def _draw_leader_boundary(self):
        # Draw leader workspace boundary (dashed lines) - blue marks leader's 70% activity area
        try:
            bound = self.map_half_size * 0.7  # Leader activity range is 70% of map
            z = 0.1
            color = [0.0, 0.4, 0.8]  # Blue

            # Draw boundary using dashed line segments
            dash_length = 1.0
            gap_length = 0.5
            corners = [
                (-bound, -bound), (bound, -bound),
                (bound, -bound), (bound, bound),
                (bound, bound), (-bound, bound),
                (-bound, bound), (-bound, -bound)
            ]

            for i in range(0, len(corners), 2):
                x1, y1 = corners[i]
                x2, y2 = corners[i+1]

                # Calculate direction and length
                dx, dy = x2 - x1, y2 - y1
                length = math.sqrt(dx**2 + dy**2)
                dx_norm, dy_norm = dx / length, dy / length

                # Draw dashed segments
                current_pos = 0.0
                while current_pos < length:
                    start_x = x1 + dx_norm * current_pos
                    start_y = y1 + dy_norm * current_pos
                    end_pos = min(current_pos + dash_length, length)
                    end_x = x1 + dx_norm * end_pos
                    end_y = y1 + dy_norm * end_pos

                    line_id = p.addUserDebugLine(
                        [start_x, start_y, z],
                        [end_x, end_y, z],
                        lineColorRGB=color,
                        lineWidth=3  # Increased line width for visibility
                    )
                    self.boundary_line_ids.append(line_id)

                    current_pos += dash_length + gap_length
        except Exception:
            pass

    def _update_leader_visuals(self):
        # Leader is virtual trajectory, no physical marker, only trajectory line shown

        # Move target marker
        if self.target_marker_id is not None:
            try:
                p.resetBasePositionAndOrientation(self.target_marker_id,
                [float(self.leader_target[0]), float(self.leader_target[1]), 0.2],
                [0, 0, 0, 1])
            except Exception:
                pass

        # Update formation shadows (gray: current position, blue: predicted position)
        if self.formation_shadow_ids:
            try:
                # Current position slots (gray)
                world_slots = self._get_world_slots(use_prediction=False)
                for i, shadow_id in enumerate(self.formation_shadow_ids):
                    slot_pos = world_slots[i]
                    p.resetBasePositionAndOrientation(shadow_id,
                    [float(slot_pos[0]), float(slot_pos[1]), 0.01],
                    [0, 0, 0, 1])
            except Exception:
                pass

        # Update predicted formation shadows (blue)
        if self.predicted_shadow_ids:
            try:
                # Predicted position slots (blue)
                predicted_slots = self._get_world_slots(use_prediction=True)
                for i, shadow_id in enumerate(self.predicted_shadow_ids):
                    slot_pos = predicted_slots[i]
                    p.resetBasePositionAndOrientation(shadow_id,
                    [float(slot_pos[0]), float(slot_pos[1]), 0.02],
                    [0, 0, 0, 1])
            except Exception:
                pass

        # Draw trajectory lines - leader trajectory in red
        if self.draw_leader_traj and self.leader_traj_max_len > 1:
            pt = [float(self.leader_pos[0]), float(self.leader_pos[1]), 0.2]
            self.leader_traj_pts.append(pt)
            if len(self.leader_traj_pts) >= 2:
                a = self.leader_traj_pts[-2]
                b = self.leader_traj_pts[-1]
                try:
                    lid = p.addUserDebugLine(a, b, lineColorRGB=[1, 0, 0], lineWidth=3)
                    self.leader_line_ids.append(lid)
                except Exception:
                    pass
                # Trim old segments
                while len(self.leader_line_ids) > self.leader_traj_max_len:
                    try:
                        old = self.leader_line_ids.pop(0)
                        p.removeUserDebugItem(old)
                    except Exception:
                        break

    def _predict_leader_position(self) -> Tuple[np.ndarray, float]:
        # Adaptive prediction of leader future position: adjust prediction time (0-5s) based on average distance
        # Calculate current slots (no prediction)
        c, s = math.cos(self.leader_heading), math.sin(self.leader_heading)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)

        # Calculate average distance from all followers to current slots
        total_dist = 0.0
        for i, offset in enumerate(self.formation_offsets):
            current_slot = self.leader_pos + rot @ offset
            agent_pos = np.array([self.agent_states[i]['x'],
            self.agent_states[i]['y']])
            dist = np.linalg.norm(current_slot - agent_pos)
            total_dist += dist

        avg_dist = total_dist / max(1, self.n_agents)

        # Adaptive prediction time: farther distance → longer prediction
        if avg_dist < 2.0:
            prediction_time = 0.0  # Very close, no prediction
        elif avg_dist < 5.0:
            # 2-5m: predict 1-2.5s
            prediction_time = 0.5 * (avg_dist - 2.0) + 1.0
        elif avg_dist < 10.0:
            # 5-10m: predict 2.5-4.5s
            prediction_time = 0.4 * (avg_dist - 5.0) + 2.5
        else:
            # >10m: predict 4.5s
            prediction_time = 4.5

        # Limit max prediction time
        prediction_time = min(prediction_time, 5.0)

        # Linear prediction of future position and heading
        predicted_pos = self.leader_pos + self.leader_vel * prediction_time
        predicted_heading = self.leader_heading

        return predicted_pos.astype(np.float32), predicted_heading

    def _get_world_slots(self, use_prediction: bool = True) -> List[np.ndarray]:
        # Compute target formation slot positions relative to leader
        # 1. Use predicted leader position for better tracking performance (training)
        #    or current position for visualization (GUI)
        if use_prediction:
            predicted_pos, predicted_heading = self._predict_leader_position()
            current_leader_pos = predicted_pos
            current_heading = predicted_heading
        else:
            # For visualization: use actual current position
            current_leader_pos = self.leader_pos
            current_heading = self.leader_heading

        # 2. Calculate raw slots based on current position
        c, s = math.cos(current_heading), math.sin(current_heading)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)

        raw_slots = []
        rotated_offsets: List[np.ndarray] = []
        # Record each slot's relative offset under current heading
        for offset in self.formation_offsets:
            rotated_offset = rot @ offset
            rotated_offsets.append(rotated_offset)
            slot_pos = current_leader_pos + rotated_offset
            raw_slots.append(slot_pos)

        # 3. Slot position smoothing
        if self.prev_world_slots is None:
            smoothed_slots = raw_slots
        else:
            smoothed_slots = []
            alpha = float(self.slot_smoothing_factor)
            for raw, prev in zip(raw_slots, self.prev_world_slots):
                smoothed = alpha * raw + (1 - alpha) * prev
                smoothed_slots.append(smoothed.astype(np.float32))

        # 4. Slot velocity (physically accurate): translation + rotation component
        # v_slot = v_leader + ω × r_i, where r_i is slot relative offset under current heading
        # In 2D: ω × [rx, ry] = ω * [-ry, rx]
        for i in range(self.n_agents):
            r = rotated_offsets[i]
            tangential = self.leader_ang_vel * np.array([-r[1], r[0]], dtype=np.float32)
            self.slot_velocities[i] = (self.leader_vel + tangential).astype(np.float32)

        # 5. Update historical slots
        self.prev_world_slots = smoothed_slots

        return smoothed_slots

    # Controls - Kinematic model implementation (revised)
    def _apply_action(self, car_id: int, info: dict, action: np.ndarray, agent_idx: int):
        # Kinematic model: x+=v*dt*cos(psi); y+=v*dt*sin(psi); v+=u*dt; psi+=omega*dt
        dt = self.frame_skip / 240.0

        # Scale normalized inputs to physical limits
        u = action[0] * (self.max_lin_accel if action[0] >= 0 else abs(self.min_lin_accel))
        omega = action[1] * (self.max_ang_vel if action[1] >= 0 else abs(self.min_ang_vel))

        # Clip to bounds
        u = np.clip(u, self.min_lin_accel, self.max_lin_accel)
        omega = np.clip(omega, self.min_ang_vel, self.max_ang_vel)

        # Get current state
        state = self.agent_states[agent_idx]
        x, y, v, psi = state["x"], state["y"], state["v"], state["psi"]

        # ========== Kinematic model update ==========
        # First update position (using current velocity and heading)
        x_new = x + v * dt * math.cos(psi)
        y_new = y + v * dt * math.sin(psi)

        # Update velocity and heading
        v_new = v + u * dt
        psi_new = psi + omega * dt

        # ========== Constraint handling ==========
        # Collision flags
        boundary_collision = False
        obstacle_collision = False

        # 1. Boundary constraint - physically correct velocity reflection
        # Decompose velocity into vector form for proper reflection
        vx_new = v_new * math.cos(psi_new)
        vy_new = v_new * math.sin(psi_new)

        # Reflect velocity components when hitting boundaries
        if x_new < -self.map_half_size or x_new > self.map_half_size:
            x_new = np.clip(x_new, -self.map_half_size, self.map_half_size)
            vx_new = -vx_new * 0.3  # Reflect x-component, lose 70% energy
            boundary_collision = True

        if y_new < -self.map_half_size or y_new > self.map_half_size:
            y_new = np.clip(y_new, -self.map_half_size, self.map_half_size)
            vy_new = -vy_new * 0.3  # Reflect y-component, lose 70% energy
            boundary_collision = True

        # Recompute speed and heading from reflected velocity vector
        if boundary_collision:
            v_new = math.sqrt(vx_new**2 + vy_new**2)
            psi_new = math.atan2(vy_new, vx_new)

        # 2. Circular obstacle constraint - revised: velocity zero on collision
        if self.has_circular_obstacle:
            if not self._check_circular_obstacle_constraint(x_new, y_new):
                # Keep previous position and stop if entering obstacle zone
                x_new, y_new = x, y
                v_new = 0.0
                obstacle_collision = True

        # 3. Velocity clamping
        v_new = np.clip(v_new, self.agent_min_speed, self.agent_max_speed)

        # 4. Normalize heading to [-π, π]
        psi_new = self._normalize_angle(psi_new)

        # ========== Update internal state ==========
        self.agent_states[agent_idx]["x"] = float(x_new)
        self.agent_states[agent_idx]["y"] = float(y_new)
        self.agent_states[agent_idx]["v"] = float(v_new)
        self.agent_states[agent_idx]["psi"] = float(psi_new)

        # Record collision info (for reward calculation)
        self.agent_states[agent_idx]["boundary_collision"] = boundary_collision
        self.agent_states[agent_idx]["obstacle_collision"] = obstacle_collision

        # ========== Sync to PyBullet (only for visualization and collision detection) ==========
        orientation = p.getQuaternionFromEuler([0, 0, psi_new])
        p.resetBasePositionAndOrientation(car_id, [x_new, y_new, 0.3], orientation)

        # Note: don't set velocity, don't control wheels
        # PyBullet's physics simulation does not affect kinematic state

    # Observations and rewards
    def _get_obs(self) -> Dict[str, np.ndarray]:
        # Build observations directly from internal agent_states (not from PyBullet)
        obs: Dict[str, np.ndarray] = {}

        # Get state from agent_states (not from PyBullet)
        car_positions = [
        np.array([s["x"], s["y"]], dtype=np.float32)
        for s in self.agent_states
        ]
        car_velocities = [
        np.array([s["v"] * math.cos(s["psi"]), s["v"] * math.sin(s["psi"])], dtype=np.float32)
        for s in self.agent_states
        ]
        headings = [s["psi"] for s in self.agent_states]
        world_slots = self._get_world_slots()

        for i, aid in enumerate(self.agent_ids):
            pos = car_positions[i]
            vel = car_velocities[i]
            yaw = headings[i]

            lidar = self._lidar_scan(self.cars[i], pos, yaw)
            self.last_lidar_scans[aid] = lidar

            slot_world = world_slots[i]
            rel_slot = (slot_world - pos) / self.map_half_size

            speed = float(np.linalg.norm(vel)) / self.agent_max_speed
            heading_error = self._angle_diff(math.atan2(rel_slot[1], rel_slot[0]), yaw) / math.pi

            neighbor_feats = self._neighbor_features(i, car_positions, car_velocities)

            leader_vel_norm = self.leader_vel / self.leader_max_vel if self.leader_max_vel > 0 else self.leader_vel

            # Add slot velocity to observation (normalized)
            slot_vel_norm = self.slot_velocities[i] / self.agent_max_speed if self.agent_max_speed > 0 else self.slot_velocities[i]

            # Distance to circular obstacle (normalized, for awareness not violation)
            if self.has_circular_obstacle:
                x_o, y_o = self.circular_obstacle_center
                dist_to_obstacle = math.sqrt((pos[0] - x_o)**2 + (pos[1] - y_o)**2)
                # Normalized distance: 0 means on boundary, >0 is safer
                norm_dist = (dist_to_obstacle - self.circular_obstacle_radius) / self.circular_obstacle_radius
            else:
                # No circular obstacle
                norm_dist = 1.0
            dist_to_obstacle_vec = np.array([norm_dist], dtype=np.float32)

            # Historical info: distance change (provides velocity information toward slot)
            slot_world = world_slots[i]
            current_dist_to_slot = float(np.linalg.norm(slot_world - pos))
            if aid in self.prev_distances:
                dist_change = (self.prev_distances[aid] - current_dist_to_slot) / self.map_half_size
            else:
                dist_change = 0.0
            dist_change_vec = np.array([dist_change], dtype=np.float32)

            vec = np.concatenate([
                lidar,
                rel_slot.astype(np.float32),
                np.array([speed], dtype=np.float32),
                np.array([heading_error], dtype=np.float32),
                neighbor_feats,
                leader_vel_norm.astype(np.float32),
                slot_vel_norm.astype(np.float32),
                dist_to_obstacle_vec,
                dist_change_vec,
            ], axis=0)
            assert vec.shape[0] == self.obs_dim, f"obs_dim mismatch: {vec.shape[0]} != {self.obs_dim}"
            obs[aid] = vec

        return obs

    def _compute_reward_and_info(self, action_dict: Dict[str, np.ndarray]):
        # Compute rewards and per-agent debug info
        reward: Dict[str, float] = {}
        info: Dict[str, dict] = {}

        car_positions = [
        np.array([s["x"], s["y"]], dtype=np.float32)
        for s in self.agent_states
        ]
        car_velocities = [
        np.array([s["v"] * math.cos(s["psi"]), s["v"] * math.sin(s["psi"])], dtype=np.float32)
        for s in self.agent_states
        ]
        world_slots = self._get_world_slots()

        max_action_change = math.sqrt(8.0)

        for i, aid in enumerate(self.agent_ids):
            pos = car_positions[i]
            vel = car_velocities[i]
            slot_world = world_slots[i]
            current_dist = float(np.linalg.norm(slot_world - pos))
            collided = self._has_collision(self.cars[i])

            # ==================== Large collision penalty ====================
            if collided:
                r = -50.0  # One-time large penalty
                reward[aid] = float(r)
                info[aid] = {
                    "slot_distance": current_dist,
                    "collision": True,
                    "vel_mismatch": 0.0,
                    "reward_breakdown": {
                        "collision_penalty": -50.0,
                        "total": -50.0,
                    }
                }
                continue

            # ==================== 1. Slot distance: core reward ====================
            # Negative proximity cost (saturating with tanh): closer -> ~0, far -> -proximity_weight
            proximity_reward = - self.proximity_weight * math.tanh(current_dist / max(1e-6, self.proximity_scale))

            # ==================== 2. Progress reward (potential-based shaping) ====================
            # Approximates F(s') - F(s) where F(s) = -distance_to_slot
            # Clipped to prevent excessive reward from single large steps
            progress_reward = 0.0
            if aid in self.prev_distances:
                dist_delta = float(self.prev_distances[aid] - current_dist)
                # Clip to prevent single-step jumps from dominating
                dist_delta = float(np.clip(dist_delta, -self.progress_clip, self.progress_clip))
                # Use symmetric weights for theoretically sound potential shaping
                # Positive delta (approaching) and negative delta (retreating) treated consistently
                progress_reward = self.progress_weight_pos * dist_delta

            # ==================== 3. Velocity matching reward ====================
            slot_vel = self.slot_velocities[i]
            slot_speed = float(np.linalg.norm(slot_vel))
            agent_speed = float(np.linalg.norm(vel))
            speed_diff = abs(slot_speed - agent_speed)

            # Simplified gate function: only reward speed matching when close to slot
            if current_dist < self.gate_distance:
                speed_reward = self.speed_weight * math.exp(-speed_diff / max(1e-6, self.speed_sigma))
            else:
                speed_reward = 0.0

            # ==================== 4. Obstacle penalty (improved: smooth continuous penalty) ====================
            obstacle_penalty = 0.0
            lidar_reading = self.last_lidar_scans.get(aid, np.ones(self.num_rays, dtype=np.float32))
            min_obstacle_dist = float(np.min(lidar_reading)) * self.lidar_max_dist

            # Multi-level continuous penalty for better gradient signal
            critical_threshold = 0.5  # Very close - critical danger
            danger_threshold = 1.0    # Close - danger zone
            safety_threshold = 3.0    # Warning zone

            if min_obstacle_dist < critical_threshold:
                # Critical: exponential penalty to strongly discourage
                obstacle_penalty = -10.0 * math.exp(-(min_obstacle_dist / max(1e-6, critical_threshold)))
            elif min_obstacle_dist < danger_threshold:
                # Danger: strong quadratic penalty
                proximity_ratio = 1.0 - (min_obstacle_dist - critical_threshold) / (danger_threshold - critical_threshold)
                obstacle_penalty = -5.0 * (proximity_ratio ** 2)
            elif min_obstacle_dist < safety_threshold:
                # Warning: mild quadratic penalty
                proximity_ratio = 1.0 - (min_obstacle_dist - danger_threshold) / (safety_threshold - danger_threshold)
                obstacle_penalty = -1.0 * (proximity_ratio ** 2)

            # ==================== 5. Inter-agent distance penalty ====================
            neighbor_penalty = 0.0
            agent_safety_threshold = 1.5
            for j in range(self.n_agents):
                if j == i:
                    continue
                other_pos = car_positions[j]
                agent_dist = float(np.linalg.norm(pos - other_pos))
                if agent_dist < agent_safety_threshold:
                    proximity_ratio = 1.0 - (agent_dist / agent_safety_threshold)
                    neighbor_penalty += -5.0 * (proximity_ratio ** 2)

            # ==================== 6. Action smoothness ====================
            current_action = np.asarray(action_dict.get(aid, np.zeros(self.action_dim)), dtype=np.float32)
            prev_action = self.prev_actions[aid]
            action_change = float(np.linalg.norm(current_action - prev_action))
            norm_jerk = action_change / max_action_change

            # Simplified: only penalize/reward smoothness when close to slot
            if current_dist < self.gate_distance:
                if norm_jerk < self.smooth_threshold:
                    smoothness_reward = self.smooth_positive_weight * (1.0 - norm_jerk / max(1e-6, self.smooth_threshold))
                else:
                    # Cap the penalty to prevent unbounded negative rewards
                    # Maximum penalty when norm_jerk reaches 1.0 (theoretical max)
                    capped_jerk = min(norm_jerk - self.smooth_threshold, 1.0 - self.smooth_threshold)
                    smoothness_reward = -self.smooth_negative_slope * capped_jerk
            else:
                smoothness_reward = 0.0

            # ==================== 7. Time penalty ====================
            # r_time = - (b0 + b1 * min(d/d_ref, cap))
            time_penalty = - (self.time_penalty_base + self.time_penalty_slope * min(current_dist / max(1e-6, self.time_penalty_ref_dist), self.time_penalty_cap))

            # ==================== 8. Success rewards ====================
            success_step_bonus = 0.0
            if self.success_step_enabled:
                if (current_dist < self.success_dist_threshold) and (speed_diff < self.success_vel_mismatch_threshold):
                    self.success_counters[aid] = self.success_counters.get(aid, 0) + 1
                    success_step_bonus = self.success_step_reward
                else:
                    self.success_counters[aid] = 0

            # ==================== Total reward ====================
            r = (proximity_reward + progress_reward + speed_reward +
                 obstacle_penalty + neighbor_penalty +
                 smoothness_reward + time_penalty + success_step_bonus)

            # Typical ranges (updated by design):

            reward[aid] = float(r)

            # Compute nearest neighbor distance for diagnostics
            nearest_neighbor_dist = float('inf')
            for j in range(self.n_agents):
                if j == i:
                    continue
                other_pos = car_positions[j]
                d = float(np.linalg.norm(pos - other_pos))
                if d < nearest_neighbor_dist:
                    nearest_neighbor_dist = d

            # Provide progress delta if available
            dist_delta_val = 0.0
            if hasattr(self, 'prev_distances') and aid in self.prev_distances:
                dist_delta_val = float(self.prev_distances[aid] - current_dist)

            # Fill per-agent info with detailed breakdown for logging/analysis
            info[aid] = {
                "slot_distance": current_dist,
                "collision": False,
                "vel_mismatch": float(speed_diff),
                "min_obstacle_dist": float(min_obstacle_dist),
                "nearest_neighbor_dist": float(nearest_neighbor_dist if nearest_neighbor_dist != float('inf') else 0.0),
                "action_change": float(action_change),
                "norm_jerk": float(norm_jerk),
                "dist_delta": float(dist_delta_val),
                "reward_breakdown": {
                    "slot_distance": float(proximity_reward),
                    "progress": float(progress_reward),
                    "speed_match": float(speed_reward),
                    "obstacle_penalty": float(obstacle_penalty),
                    "neighbor_penalty": float(neighbor_penalty),
                    "smoothness": float(smoothness_reward),
                    "time_penalty": float(time_penalty),
                    "success_step": float(success_step_bonus),
                    "total": float(r),
                }
            }

        # Terminal success: award and flag if configured
        team_success = False
        if self.success_terminal_enabled:
            if self.success_team_all:
                team_success = all(self.success_counters.get(aid, 0) >= self.success_required_steps for aid in self.agent_ids)
            else:
                team_success = any(self.success_counters.get(aid, 0) >= self.success_required_steps for aid in self.agent_ids)
            if team_success:
                for aid in self.agent_ids:
                    reward[aid] = reward.get(aid, 0.0) + self.success_terminal_reward
                    if 'reward_breakdown' in info.get(aid, {}):
                        info[aid]['reward_breakdown']['success_terminal'] = float(self.success_terminal_reward)
                    info[aid]['success_terminal'] = True

        # Save current distance for next step's progress reward calculation (update every step)
        for i, aid in enumerate(self.agent_ids):
            pos = car_positions[i]
            slot_world = world_slots[i]
            dist = float(np.linalg.norm(slot_world - pos))
            self.prev_distances[aid] = dist

        return reward, info

    # Geometry and sensing
    def _lidar_scan(self, car_id: int, pos_xy: np.ndarray, yaw: float) -> np.ndarray:
        angles = np.linspace(-math.pi * 0.75, math.pi * 0.75, self.num_rays)
        from_pts = []
        to_pts = []
        z = 0.2
        for a in angles:
            dir_world = np.array([math.cos(yaw + a), math.sin(yaw + a)], dtype=np.float32)
            start = np.array([pos_xy[0], pos_xy[1], z], dtype=np.float32)
            end = np.array([pos_xy[0] + dir_world[0] * self.lidar_max_dist,
            pos_xy[1] + dir_world[1] * self.lidar_max_dist, z], dtype=np.float32)
            from_pts.append(start.tolist())
            to_pts.append(end.tolist())

        hits = p.rayTestBatch(from_pts, to_pts)
        dists = np.empty(self.num_rays, dtype=np.float32)
        for i, h in enumerate(hits):
            hit_frac = h[2]
            if hit_frac < 0:
                dists[i] = self.lidar_max_dist
            else:
                dists[i] = hit_frac * self.lidar_max_dist
        return (dists / self.lidar_max_dist).astype(np.float32)

    def _neighbor_features(self, idx: int, positions: List[np.ndarray], velocities: List[np.ndarray]) -> np.ndarray:
        feats: List[float] = []
        my_pos = positions[idx]
        my_vel = velocities[idx]
        dists = []
        for j in range(self.n_agents):
            if j == idx:
                continue
            d = float(np.linalg.norm(positions[j] - my_pos))
            if d <= self.sense_neighbors_radius:
                dists.append((d, j))
        dists.sort(key=lambda x: x[0])

        count = 0
        for _, j in dists:
            if count >= self.neighbor_max:
                break
            rel = (positions[j] - my_pos) / self.sense_neighbors_radius
            # Normalize, avoiding division by zero
            norm_rel_speed = (float(np.linalg.norm(velocities[j] - my_vel)) / (2 * self.agent_max_speed)
                              if self.agent_max_speed > 0 else 0.0)
            feats.extend([float(rel[0]), float(rel[1]), norm_rel_speed])
            count += 1

        while count < self.neighbor_max:
            feats.extend([0.0, 0.0, 0.0])
            count += 1

        return np.asarray(feats, dtype=np.float32)

    def _has_collision(self, car_id: int) -> bool:
        contact_points = p.getContactPoints(bodyA=car_id)
        # Exclude ground and other agents, only detect collision with obstacles and boundaries
        for cp in contact_points:
            body_b = cp[2]
            if body_b != self.plane_id and body_b not in self.cars:
                return True
        return False

    def _get_yaw(self, car_id: int) -> float:
        _, orn = p.getBasePositionAndOrientation(car_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return float(yaw)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        # Normalize angle to [-pi, pi]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _check_circular_obstacle_constraint(self, x: float, y: float) -> bool:
        # Check circular obstacle constraint (inside/outside circle)
        if not self.has_circular_obstacle:
            return True

        x_o, y_o = self.circular_obstacle_center
        dist_sq = (x - x_o) ** 2 + (y - y_o) ** 2
        radius_sq = self.circular_obstacle_radius ** 2

        return dist_sq >= radius_sq

    def _check_all_constraints(self, agent_idx: int) -> Dict[str, bool]:
        # Check whether an agent violates any constraints. Returns a dict of flags.
        state = self.agent_states[agent_idx]
        x, y, v, psi = state["x"], state["y"], state["v"], state["psi"]

        violations = {
        "position_x": not (-self.map_half_size <= x <= self.map_half_size),
        "position_y": not (-self.map_half_size <= y <= self.map_half_size),
        "velocity": not (self.agent_min_speed <= v <= self.agent_max_speed),
        "circular_obstacle": False
        }

        if self.has_circular_obstacle:
            violations["circular_obstacle"] = not self._check_circular_obstacle_constraint(x, y)

        return violations
