"""
rrt_planner.py
基于RRT（快速探索随机树）的路径规划器
用于AirSim无人机避障导航

功能：
- RRT树节点和树结构
- 障碍物占据栅格地图构建
- RRT路径规划算法
- 路径平滑优化
- 动态重规划检测
"""

import numpy as np
import math
import time
from typing import List, Tuple, Optional

# AirSim导入（仅在实际使用时需要）
try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    # 为测试创建模拟的Box3D类
    class Box3D:
        def __init__(self):
            self.min = type('obj', (object,), {'x_val': 0, 'y_val': 0, 'z_val': 0})()
            self.max = type('obj', (object,), {'x_val': 0, 'y_val': 0, 'z_val': 0})()
    airsim = type('module', (object,), {'Box3D': Box3D})()


class RRTNode:
    """RRT树节点"""
    def __init__(self, position: np.ndarray, parent=None):
        """
        初始化RRT节点
        :param position: 节点位置 [x, y]
        :param parent: 父节点
        """
        self.position = np.array(position, dtype=float)
        self.parent = parent
        self.cost = 0.0  # 从起点到该节点的代价
        
    def path_to_root(self) -> List[np.ndarray]:
        """返回从该节点到根节点的路径"""
        path = []
        node = self
        while node is not None:
            path.append(node.position.copy())
            node = node.parent
        return list(reversed(path))


class OccupancyGrid:
    """占据栅格地图"""
    def __init__(self, center: np.ndarray, size: float, resolution: float):
        """
        初始化占据栅格
        :param center: 地图中心位置 [x, y]
        :param size: 地图边长（米）
        :param resolution: 栅格分辨率（米/格）
        """
        self.center = np.array(center, dtype=float)
        self.size = size
        self.resolution = resolution
        self.grid_size = int(size / resolution)
        
        # 0=自由, 1=障碍物
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
    def world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        relative = world_pos - self.center + self.size / 2
        grid_x = int(relative[0] / self.resolution)
        grid_y = int(relative[1] / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """栅格坐标转世界坐标"""
        world_x = grid_x * self.resolution + self.center[0] - self.size / 2
        world_y = grid_y * self.resolution + self.center[1] - self.size / 2
        return np.array([world_x, world_y])
    
    def is_valid_grid(self, grid_x: int, grid_y: int) -> bool:
        """检查栅格坐标是否有效"""
        return 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size
    
    def is_occupied(self, world_pos: np.ndarray) -> bool:
        """检查世界坐标位置是否被占据"""
        gx, gy = self.world_to_grid(world_pos)
        if not self.is_valid_grid(gx, gy):
            return True  # 超出边界视为障碍
        return self.grid[gx, gy] == 1
    
    def add_obstacle_box(self, box_min: np.ndarray, box_max: np.ndarray, safety_margin: float = 0.5):
        """
        添加矩形障碍物
        :param box_min: 障碍物最小坐标 [x, y]
        :param box_max: 障碍物最大坐标 [x, y]
        :param safety_margin: 安全边距（米）
        """
        # 扩展安全边距
        box_min_safe = box_min - safety_margin
        box_max_safe = box_max + safety_margin
        
        # 转换为栅格坐标
        gx_min, gy_min = self.world_to_grid(box_min_safe)
        gx_max, gy_max = self.world_to_grid(box_max_safe)
        
        # 限制在有效范围内
        gx_min = max(0, gx_min)
        gy_min = max(0, gy_min)
        gx_max = min(self.grid_size - 1, gx_max)
        gy_max = min(self.grid_size - 1, gy_max)
        
        # 标记为障碍物
        self.grid[gx_min:gx_max+1, gy_min:gy_max+1] = 1
    
    def clear(self):
        """清空地图"""
        self.grid.fill(0)


class RRTPlanner:
    """RRT路径规划器"""
    def __init__(self, 
                 step_size: float = 2.0,
                 goal_sample_rate: float = 0.2,
                 max_iterations: int = 3000,
                 goal_tolerance: float = 2.0):
        """
        初始化RRT规划器
        :param step_size: 树扩展步长（米）
        :param goal_sample_rate: 目标采样概率（0-1）
        :param max_iterations: 最大迭代次数
        :param goal_tolerance: 目标容差（米）
        """
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.goal_tolerance = goal_tolerance
        
        self.nodes: List[RRTNode] = []
        
    def plan(self, 
             start: np.ndarray, 
             goal: np.ndarray, 
             occupancy_grid: OccupancyGrid,
             debug: bool = False) -> Optional[List[np.ndarray]]:
        """
        使用RRT规划路径
        :param start: 起点 [x, y]
        :param goal: 终点 [x, y]
        :param occupancy_grid: 占据栅格地图
        :param debug: 是否打印调试信息
        :return: 路径点列表，失败返回None
        """
        start_time = time.time()
        
        # 检查起点和终点是否有效
        if occupancy_grid.is_occupied(start):
            if debug:
                print("[RRT] 起点位置有障碍物！")
            return None
        if occupancy_grid.is_occupied(goal):
            if debug:
                print("[RRT] 终点位置有障碍物！")
            return None
        
        # 初始化树
        self.nodes = [RRTNode(start)]
        
        for i in range(self.max_iterations):
            # 1. 采样随机点
            if np.random.random() < self.goal_sample_rate:
                sample_point = goal
            else:
                # 在地图范围内随机采样
                sample_point = self._random_sample(occupancy_grid)
            
            # 2. 找到最近的节点
            nearest_node = self._get_nearest_node(sample_point)
            
            # 3. 向采样点扩展
            new_position = self._steer(nearest_node.position, sample_point)
            
            # 4. 碰撞检测
            if self._is_collision_free(nearest_node.position, new_position, occupancy_grid):
                # 创建新节点
                new_node = RRTNode(new_position, parent=nearest_node)
                new_node.cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)
                self.nodes.append(new_node)
                
                # 5. 检查是否到达目标
                if np.linalg.norm(new_position - goal) <= self.goal_tolerance:
                    # 创建目标节点
                    goal_node = RRTNode(goal, parent=new_node)
                    self.nodes.append(goal_node)
                    
                    # 提取路径
                    path = goal_node.path_to_root()
                    
                    if debug:
                        elapsed = time.time() - start_time
                        print(f"[RRT] 找到路径！迭代次数: {i+1}, 节点数: {len(self.nodes)}, "
                              f"耗时: {elapsed:.2f}s, 路径长度: {len(path)}点")
                    
                    return path
        
        if debug:
            print(f"[RRT] 未找到路径，已达最大迭代次数 {self.max_iterations}")
        return None
    
    def _random_sample(self, occupancy_grid: OccupancyGrid) -> np.ndarray:
        """在地图范围内随机采样"""
        half_size = occupancy_grid.size / 2
        x = occupancy_grid.center[0] + np.random.uniform(-half_size, half_size)
        y = occupancy_grid.center[1] + np.random.uniform(-half_size, half_size)
        return np.array([x, y])
    
    def _get_nearest_node(self, point: np.ndarray) -> RRTNode:
        """找到距离给定点最近的节点"""
        distances = [np.linalg.norm(node.position - point) for node in self.nodes]
        nearest_idx = np.argmin(distances)
        return self.nodes[nearest_idx]
    
    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """从from_pos向to_pos扩展step_size距离"""
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_pos
        else:
            return from_pos + (direction / distance) * self.step_size
    
    def _is_collision_free(self, 
                          from_pos: np.ndarray, 
                          to_pos: np.ndarray, 
                          occupancy_grid: OccupancyGrid) -> bool:
        """检查从from_pos到to_pos的路径是否无碰撞"""
        # 沿路径采样多个点进行碰撞检测
        distance = np.linalg.norm(to_pos - from_pos)
        num_checks = int(distance / occupancy_grid.resolution) + 1
        
        for i in range(num_checks + 1):
            t = i / max(num_checks, 1)
            check_point = from_pos + t * (to_pos - from_pos)
            if occupancy_grid.is_occupied(check_point):
                return False
        
        return True


def smooth_path(path: List[np.ndarray], 
                occupancy_grid: OccupancyGrid,
                max_iterations: int = 100) -> List[np.ndarray]:
    """
    使用快捷方式优化平滑路径
    :param path: 原始路径
    :param occupancy_grid: 占据栅格地图
    :param max_iterations: 最大优化迭代次数
    :return: 平滑后的路径
    """
    if len(path) <= 2:
        return path
    
    smoothed = path.copy()
    
    for _ in range(max_iterations):
        if len(smoothed) <= 2:
            break
        
        # 随机选择两个不相邻的点
        i = np.random.randint(0, len(smoothed) - 2)
        j = np.random.randint(i + 2, len(smoothed))
        
        # 检查是否可以直接连接
        if _is_path_collision_free(smoothed[i], smoothed[j], occupancy_grid):
            # 移除中间的点
            smoothed = smoothed[:i+1] + smoothed[j:]
    
    return smoothed


def _is_path_collision_free(from_pos: np.ndarray, 
                            to_pos: np.ndarray, 
                            occupancy_grid: OccupancyGrid) -> bool:
    """检查路径是否无碰撞（用于路径平滑）"""
    distance = np.linalg.norm(to_pos - from_pos)
    num_checks = int(distance / occupancy_grid.resolution) + 1
    
    for i in range(num_checks + 1):
        t = i / max(num_checks, 1)
        check_point = from_pos + t * (to_pos - from_pos)
        if occupancy_grid.is_occupied(check_point):
            return False
    
    return True


def build_occupancy_grid_from_obstacles(obstacles: List[airsim.Box3D],
                                       drone_position: np.ndarray,
                                       drone_yaw: float,
                                       map_size: float = 50.0,
                                       resolution: float = 0.5,
                                       safety_margin: float = 1.0) -> OccupancyGrid:
    """
    从AirSim障碍物列表构建占据栅格地图
    :param obstacles: 障碍物列表（相对于无人机的局部坐标）
    :param drone_position: 无人机当前位置 [x, y]
    :param drone_yaw: 无人机当前偏航角（弧度）
    :param map_size: 地图大小（米）
    :param resolution: 栅格分辨率（米）
    :param safety_margin: 安全边距（米）
    :return: 占据栅格地图
    """
    # 创建以无人机为中心的地图
    grid = OccupancyGrid(drone_position, map_size, resolution)
    
    # 旋转矩阵（从机体坐标系到世界坐标系）
    cos_yaw = math.cos(drone_yaw)
    sin_yaw = math.sin(drone_yaw)
    R = np.array([[cos_yaw, -sin_yaw],
                  [sin_yaw, cos_yaw]])
    
    # 添加障碍物
    for obstacle in obstacles:
        # 障碍物在机体坐标系中的位置
        local_min = np.array([obstacle.min.x_val, obstacle.min.y_val])
        local_max = np.array([obstacle.max.x_val, obstacle.max.y_val])
        
        # 转换到世界坐标系
        world_min = drone_position + R @ local_min
        world_max = drone_position + R @ local_max
        
        # 确保min < max
        box_min = np.minimum(world_min, world_max)
        box_max = np.maximum(world_min, world_max)
        
        # 添加到地图
        grid.add_obstacle_box(box_min, box_max, safety_margin)
    
    return grid


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 60)
    print("RRT路径规划器测试")
    print("=" * 60)
    
    # 创建测试地图
    grid = OccupancyGrid(center=np.array([0, 0]), size=50, resolution=0.5)
    
    # 添加一些障碍物
    grid.add_obstacle_box(np.array([5, -5]), np.array([15, 5]), safety_margin=0.5)
    grid.add_obstacle_box(np.array([-10, 10]), np.array([-5, 20]), safety_margin=0.5)
    grid.add_obstacle_box(np.array([10, 15]), np.array([20, 18]), safety_margin=0.5)
    
    # 创建RRT规划器
    planner = RRTPlanner(
        step_size=2.0,
        goal_sample_rate=0.2,
        max_iterations=2000,
        goal_tolerance=2.0
    )
    
    # 规划路径
    start = np.array([0, 0])
    goal = np.array([20, 20])
    
    print(f"\n起点: {start}")
    print(f"终点: {goal}")
    print("\n开始RRT规划...")
    
    path = planner.plan(start, goal, grid, debug=True)
    
    if path is not None:
        print(f"\n原始路径点数: {len(path)}")
        
        # 路径平滑
        print("\n开始路径平滑...")
        smoothed_path = smooth_path(path, grid, max_iterations=100)
        print(f"平滑后路径点数: {len(smoothed_path)}")
        
        print("\n平滑后的路径:")
        for i, point in enumerate(smoothed_path):
            print(f"  {i}: ({point[0]:.2f}, {point[1]:.2f})")
        
        print("\n✅ 测试成功！")
    else:
        print("\n❌ 未找到路径")
