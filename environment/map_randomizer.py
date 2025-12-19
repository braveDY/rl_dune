import math
import numpy as np
from typing import Tuple, List
from shapely.geometry import Point, Polygon

MAP_WIDTH, MAP_HEIGHT = 20.0, 20.0
MIN_START_GOAL_DIST = 8.0  # 起点和终点的最小距离
MAX_START_GOAL_DIST = 15.0  # 起点和终点的最大距离


class MapRandomizer:
    """地图随机化器：负责障碍物和起点终点的随机生成"""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)
    
    def randomize_obstacles(self, env_obstacles: List):
        """随机化所有障碍物位置"""
        self.obstacles.clear()
        
        # 分类障碍物（排除围墙）
        circles = []
        rectangles = []
        polygons = []
        
        for obs in env_obstacles:
            # 跳过围墙（通过尺寸判断：长度或宽度超过10的长方形视为围墙）
            if obs.shape == 'rectangle' and self._is_wall(obs):
                # 围墙不添加到障碍物列表，因为我们通过边界margin来避免
                # 围墙位于地图边缘，只要起点/终点在安全范围内就不会碰撞
                continue
            
            if obs.shape == 'circle':
                circles.append(obs)
            elif obs.shape == 'rectangle':
                rectangles.append(obs)
            elif obs.shape == 'polygon':
                polygons.append(obs)
        
        # 随机化圆形障碍物
        for obs in circles:
            self._randomize_circle(obs)
        
        # 随机化长方形障碍物
        for obs in rectangles:
            self._randomize_rectangle(obs)
        
        # 随机化多边形障碍物
        for obs in polygons:
            self._randomize_polygon(obs)
    
    def _is_wall(self, obs) -> bool:
        """判断障碍物是否为围墙"""
        # 围墙的特征：长度或宽度超过10，或者颜色为gray
        if obs.length > 10 or obs.width > 10:
            return True
        if hasattr(obs, 'color') and obs.color == 'gray':
            return True
        return False
    
    def _randomize_circle(self, obs, max_attempts: int = 1000):
        """随机化单个圆形障碍物"""
        # 随机半径
        radius = self.rng.uniform(0.1, 0.4)
        obs.gf._original_geometry = Point([0, 0]).buffer(radius)
        obs.gf.length, obs.gf.width = obs.gf.cal_length_width(obs.gf._original_geometry)
        
        # 尝试找到有效位置
        for _ in range(max_attempts):
            x = self.rng.uniform(1.5, MAP_WIDTH - 1.5)
            y = self.rng.uniform(1.5, MAP_HEIGHT - 1.5)
            
            if self._is_obstacle_valid(x, y, radius):
                obs.set_state(np.array([[x], [y], [0]]), init=True)
                self.obstacles.append((x, y, radius))
                return
        
        # 失败则使用随机位置
        x = self.rng.uniform(1.5, MAP_WIDTH - 1.5)
        y = self.rng.uniform(1.5, MAP_HEIGHT - 1.5)
        obs.set_state(np.array([[x], [y], [0]]), init=True)
        self.obstacles.append((x, y, radius))
    
    def _randomize_rectangle(self, obs, max_attempts: int = 1000):
        """随机化单个长方形障碍物"""
        # 随机长度和宽度
        length = self.rng.uniform(4, 10)  # 长度范围
        width = self.rng.uniform(0.3, 0.5)   # 宽度范围
        
        # 创建长方形几何体
        half_length, half_width = length / 2, width / 2
        rect_points = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        obs.gf._original_geometry = Polygon(rect_points)
        obs.gf.length, obs.gf.width = length, width
        
        # 存储尺寸信息用于渲染更新（避免直接设置只读属性）
        obs._custom_length = length
        obs._custom_width = width
        
        # 计算碰撞半径（对角线的一半）
        collision_radius = math.sqrt(length**2 + width**2) / 2
        
        # 尝试找到有效位置
        margin = max(1.5, collision_radius + 0.5)
        for _ in range(max_attempts):
            x = self.rng.uniform(margin, MAP_WIDTH - margin)
            y = self.rng.uniform(margin, MAP_HEIGHT - margin)
            theta = self.rng.uniform(0, 2 * math.pi)  # 随机朝向
            
            if self._is_obstacle_valid(x, y, collision_radius):
                obs.set_state(np.array([[x], [y], [theta]]), init=True)
                self.obstacles.append((x, y, collision_radius))
                return
        
        # 失败则使用随机位置
        x = self.rng.uniform(margin, MAP_WIDTH - margin)
        y = self.rng.uniform(margin, MAP_HEIGHT - margin)
        theta = self.rng.uniform(0, 2 * math.pi)
        obs.set_state(np.array([[x], [y], [theta]]), init=True)
        self.obstacles.append((x, y, collision_radius))
    
    def _randomize_polygon(self, obs, max_attempts: int = 1000):
        """随机化单个多边形障碍物"""
        radius = 0.8  # 保守估计的碰撞半径
        margin = 1.5  # 边界安全边距
        
        # 尝试找到有效位置
        for _ in range(max_attempts):
            x = self.rng.uniform(margin, MAP_WIDTH - margin)
            y = self.rng.uniform(margin, MAP_HEIGHT - margin)
            theta = self.rng.uniform(0, 2 * math.pi)
            
            if self._is_obstacle_valid(x, y, radius):
                obs.set_state(np.array([[x], [y], [theta]]), init=True)
                self.obstacles.append((x, y, radius))
                return
        
        # 失败则使用随机位置
        x = self.rng.uniform(margin, MAP_WIDTH - margin)
        y = self.rng.uniform(margin, MAP_HEIGHT - margin)
        obs.set_state(np.array([[x], [y], [0]]), init=True)
        self.obstacles.append((x, y, radius))
    
    def _is_obstacle_valid(self, x: float, y: float, radius: float) -> bool:
        """检查障碍物位置是否与已有障碍物重叠"""
        for ox, oy, oradius in self.obstacles:
            if math.hypot(x - ox, y - oy) < (radius + oradius):
                return False
        return True
    
    def generate_start_goal(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """生成起点和终点（确保距离足够远且避开障碍物）"""
        start = self._generate_valid_position()
        
        # 生成终点，确保与起点距离足够远
        for _ in range(500):
            goal = self._generate_valid_position()
            if MIN_START_GOAL_DIST <= self._distance(start, goal) < MAX_START_GOAL_DIST:
                return start, goal
        
        # 失败则返回默认位置
        return start, self._generate_valid_position()
    
    def _generate_valid_position(self, max_attempts: int = 500) -> Tuple[float, float]:
        """生成一个有效的位置（避开障碍物和围墙）"""
        # 安全边距，确保不会生成在围墙附近
        margin = 1.5
        
        # 尝试在安全范围内随机生成
        for _ in range(max_attempts):
            x = self.rng.uniform(margin, MAP_WIDTH - margin)
            y = self.rng.uniform(margin, MAP_HEIGHT - margin)
            if self._is_position_valid(x, y):
                return (x, y)
        
        # 尝试地图四角（带安全边距）
        corners = [
            (2.0, 2.0),
            (2.0, MAP_HEIGHT - 2.0),
            (MAP_WIDTH - 2.0, 2.0),
            (MAP_WIDTH - 2.0, MAP_HEIGHT - 2.0)
        ]
        for corner in corners:
            if self._is_position_valid(corner[0], corner[1]):
                return corner
        
        # 最后返回地图中心
        return (MAP_WIDTH / 2, MAP_HEIGHT / 2)
    
    def _is_position_valid(self, x: float, y: float) -> bool:
        """检查位置是否有效"""
        # 检查障碍物冲突
        for ox, oy, oradius in self.obstacles:
            min_dist = oradius + 0.8
            if math.hypot(x - ox, y - oy) < min_dist:
                return False
        
        return True
    
    @staticmethod
    def _distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间距离"""
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
