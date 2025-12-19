import math
import numpy as np
import irsim
from typing import Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
try:
    from .map_randomizer import MapRandomizer
except:
    from map_randomizer import MapRandomizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dune.dune import DUNE

V_MIN = -1.0
V_MAX = 2.0
W_MIN = -1.5
W_MAX = 1.5
REACH_THRESHOLD = 0.3
MAX_STEPS = 150
MAX_REL_DIST = 2.0

class DiffRobotEnv(gym.Env):
    def __init__(self, irsim_yaml="irsim_env.yaml", display=True, dune: DUNE = None):
        super().__init__()
        self.display = display
        self.dune = dune
        self.min_dist = None
        # 初始化irsim环境
        self._env = irsim.make(irsim_yaml, display=display)
        self._robot = self._env.robot_list[0]
        
        # 状态空间：[dist, cos(theta), sin(theta)] + lidar
        lidar_beams = int(self._robot.lidar.number)
        state_dim = 4 + lidar_beams
        self._angles = np.linspace(0, 2 * math.pi, lidar_beams, endpoint=False)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([V_MIN, W_MIN], dtype=np.float32),
            high=np.array([V_MAX, W_MAX], dtype=np.float32),
            dtype=np.float32
        )
        
        # 内部状态
        self._rng = np.random.RandomState(42)
        self._randomizer = MapRandomizer(self._rng)
        self._start_pos: Optional[Tuple[float, float]] = None
        self._goal_pos: Optional[Tuple[float, float]] = None
        self._step_count = 0
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._randomizer = MapRandomizer(self._rng)
        
        # 随机化障碍物
        self._randomizer.randomize_obstacles(self._env.obstacle_list)
        self._env.build_tree()
        
        # 随机化起点和终点
        self._start_pos, self._goal_pos = self._randomizer.generate_start_goal()
        start_theta = self._rng.uniform(0, 2 * math.pi)
        goal_theta = self._rng.uniform(0, 2 * math.pi)
        start = np.array([[self._start_pos[0]], [self._start_pos[1]], [start_theta]])
        goal = np.array([[self._goal_pos[0]], [self._goal_pos[1]], [goal_theta]])
        self._robot.set_state(start, init=True)
        self._robot.set_goal(goal)
        
        # 重置内部状态
        self._step_count = 0
        self._env._world.reset()
        self._env._world.count = 0
        
        # 更新可视化
        if self.display:
            self._env._env_plot.step("all", self._env.objects)
        
        # 返回初始状态和信息
        robot_state = self._robot.state.flatten()
        pre_action = (0, 0)
        lidar_ranges = self._robot.lidar.get_scan()["ranges"]
        next_state = self._get_state(robot_state, self._goal_pos, pre_action, lidar_ranges, self.dune)

        info = {
            "target_distance": self._distance_to_goal(),
            "start": start,
            "goal": goal,
            "num_obstacles": len(self._randomizer.obstacles)
        }
        
        return next_state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行动作"""
        self._step_count += 1
        dist_prev = self._distance_to_goal()

        # 裁剪并执行动作
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 转换为列向量格式 (2, 1)
        action_col = np.array([[action[0]], [action[1]]])
        self._env.step(action_col, action_id=0)

        # 检查终止条件
        dist_curr = self._distance_to_goal()
        is_reach = dist_curr <= REACH_THRESHOLD
        is_collision = self._robot.collision
        
        # 更新状态
        robot_state = self._robot.state.flatten()
        goal = self._goal_pos
        pre_action = action
        lidar_ranges = self._robot.lidar.get_scan()["ranges"]
        next_state = self._get_state(robot_state, goal, pre_action, lidar_ranges, self.dune)  
        dist_curr = self._distance_to_goal()
        # 计算奖励
        reward = self._compute_reward(dist_prev, dist_curr, is_collision, is_reach, action)
        
        # 判断终止和截断
        terminated = is_reach or is_collision
        truncated = self._step_count >= MAX_STEPS and not terminated
        
        info = {
            "is_reach_target": is_reach,
            "is_collision": is_collision,
            "current_step": self._step_count,
            "target_distance": dist_curr,
            "min_target_distance": self.min_dist  # 添加缺失的字段
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self, robot_state, goal, pre_action, lidar_ranges, dune: DUNE = None) -> np.ndarray:
        x, y, theta = robot_state
        sin_theta, cos_theta = math.sin(theta), math.cos(theta)
        
        dx_world = goal[0] - x
        dy_world = goal[1] - y
        dx_robot = dx_world * cos_theta + dy_world * sin_theta
        dy_robot = -dx_world * sin_theta + dy_world * cos_theta
        
        dist_to_goal = math.hypot(dx_robot, dy_robot)
        alpha = math.atan2(dy_robot, dx_robot)
        
        if dune is not None:
            dist, self.min_dist = dune(lidar_ranges, self._angles)
        else:
            dist, self.min_dist = lidar_ranges, min(lidar_ranges)
        
        state = np.array([dist_to_goal, alpha, pre_action[0], pre_action[1]] + dist.tolist(), dtype=np.float32)
        return state

    def _distance_to_goal(self) -> float:
        x, y, _ = self._robot._state.flatten()[:3]
        return math.hypot(self._goal_pos[0] - x, self._goal_pos[1] - y)
    
    def _compute_reward(self, dist_prev, dist_curr, is_collision, is_reach, action) -> float:
        """
        改进的奖励函数 - 解决卡在角落和泛化性问题
        核心改进：
        1. 大幅增强前进动力（距离进度奖励）
        2. 严厉惩罚停滞行为（原地不动/转圈）
        3. 强制鼓励探索（速度奖励）
        4. 平衡避障与前进
        """
        v, w = action[0], action[1]
        
        # 1. 终端奖励
        if is_reach:
            time_bonus = max(0, 1.0 - self._step_count / MAX_STEPS)
            return 200.0 + 50.0 * time_bonus
        
        if is_collision:
            return -200.0
        
        # 2. 距离进度奖励（核心驱动力）- 大幅提升权重
        distance_progress = dist_prev - dist_curr
        reward = distance_progress * 150.0  # 从100提升到150
        
        # 3. 获取环境信息
        x, y, theta = self._robot.state.flatten()[:3]
        target_theta = math.atan2(self._goal_pos[1] - y, self._goal_pos[0] - x)
        angle_to_goal = abs(math.atan2(math.sin(theta - target_theta),
                                       math.cos(theta - target_theta)))
        
        # 获取最小障碍物距离
        try:
            lidar_data = self._robot.lidar.get_scan()["ranges"]
            valid_ranges = [d for d in lidar_data if not (math.isinf(d) or math.isnan(d))]
            min_distance = min(valid_ranges) if valid_ranges else 10.0
            
            # 前方区域
            num_beams = len(lidar_data)
            front_start = int(num_beams * 5/12)
            front_end = int(num_beams * 7/12)
            front_ranges = [d for d in lidar_data[front_start:front_end]
                           if not (math.isinf(d) or math.isnan(d))]
            front_min = min(front_ranges) if front_ranges else 10.0
        except:
            min_distance = front_min = 10.0
        
        # 4. 强化速度奖励 - 鼓励前进
        if angle_to_goal < np.deg2rad(45) and front_min > 1.0:
            # 朝向目标且前方相对安全，强烈鼓励高速前进
            reward += v * 5.0  # 从2.0提升到5.0
            # 惩罚不必要的转向
            reward -= abs(w) * 1.5
        elif front_min > 0.8:
            # 即使角度不对，只要前方安全也鼓励前进
            reward += v * 3.0  # 从0.5提升到3.0
            # 鼓励转向调整
            if angle_to_goal > np.deg2rad(30):
                reward += abs(w) * 0.5  # 需要转向时不惩罚
        else:
            # 前方有障碍，适度前进
            reward += v * 1.0
        
        # 5. 朝向奖励 - 鼓励面向目标
        if dist_curr > 0.5:
            heading_reward = (1.0 - angle_to_goal / math.pi) * 3.0  # 从2.0提升到3.0
            reward += heading_reward
        
        # 6. 避障惩罚 - 只在真正危险时
        if min_distance < 0.25:
            reward -= 30.0  # 从20.0提升到30.0
        elif min_distance < 0.4:
            reward -= 8.0  # 从5.0提升到8.0
        
        # 7. 严厉惩罚停滞和原地晃动（关键改进）
        if v < 0.2:  # 从0.1提升到0.2，更严格
            # 几乎不动 - 大幅增加惩罚
            reward -= 15.0  # 从5.0提升到15.0
            if abs(w) > 0.1:
                # 原地转圈 - 极重惩罚
                reward -= 25.0  # 从10.0提升到25.0
        
        # 8. 惩罚过度转向（防止蛇形前进）
        if abs(w) > 1.2:
            reward -= abs(w) * 3.0  # 从2.0提升到3.0
        
        # 9. 时间惩罚 - 轻微增加
        reward -= 0.1  # 从0.05提升到0.1
        
        # 10. 额外：距离相关的基础奖励（鼓励接近目标）
        # 距离越近，基础奖励越高
        if dist_curr < 5.0:
            reward += (5.0 - dist_curr) * 0.5
        
        return reward

    
    def render(self):
        if self.display:
            for obs in self._env.obstacle_list:
                if obs.shape == 'circle' and hasattr(obs, 'object_patch'):
                    obs.object_patch.set_radius(obs.radius)
                elif obs.shape == 'rectangle' and hasattr(obs, 'object_patch'):
                    new_vertices = obs.gf.original_vertices.T
                    obs.object_patch.set_xy(new_vertices)
            self._env.render()
    
    def close(self):
        self._env.end(ending_time=0.1)




