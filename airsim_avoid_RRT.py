"""
airsim_avoid_RRT.py
基于RRT算法的AirSim无人机避障导航

功能：
- 使用异步障碍物检测器获取实时障碍物信息
- 构建占据栅格地图
- 使用RRT规划路径
- 动态重规划
- 平滑路径跟随控制
"""

import numpy as np
import math
import time
import airsim
from typing import List, Optional
from async_obstacle_detector import AsyncObstacleDetector
from rrt_planner import (RRTPlanner, OccupancyGrid, smooth_path, 
                         build_occupancy_grid_from_obstacles)
from UavAgent import get_state


class RRTNavigationController:
    """基于RRT的导航控制器"""
    
    def __init__(self, 
                 client: airsim.MultirotorClient,
                 vehicle_name: str = 'UAV0',
                 map_size: float = 50.0,
                 grid_resolution: float = 0.5,
                 safety_margin: float = 1.0,
                 replan_interval: float = 3.0,
                 replan_distance_threshold: float = 5.0):
        """
        初始化RRT导航控制器
        :param client: AirSim客户端
        :param vehicle_name: 无人机名称
        :param map_size: 占据栅格地图大小（米）
        :param grid_resolution: 栅格分辨率（米）
        :param safety_margin: 障碍物安全边距（米）
        :param replan_interval: 重规划检查间隔（秒）
        :param replan_distance_threshold: 触发重规划的障碍物变化阈值（米）
        """
        self.client = client
        self.vehicle_name = vehicle_name
        self.map_size = map_size
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.replan_interval = replan_interval
        self.replan_distance_threshold = replan_distance_threshold
        
        # RRT规划器（优化参数以提高安全性）
        self.planner = RRTPlanner(
            step_size=1.5,           # 减小步长，路径更精细
            goal_sample_rate=0.25,   # 增加目标采样率，更快找到路径
            max_iterations=4000,     # 增加迭代次数
            goal_tolerance=2.0
        )
        
        # 当前路径和目标
        self.current_path: Optional[List[np.ndarray]] = None
        self.current_goal: Optional[np.ndarray] = None
        self.path_index = 0
        
        # 重规划控制
        self.last_replan_time = 0
        self.last_obstacle_count = 0
        
    def navigate_to_waypoints(self,
                             waypoints: List[airsim.Vector3r],
                             Q_search: float = 12.0,
                             max_speed: float = 3.0,
                             waypoint_tolerance: float = 2.0,
                             debug: bool = True):
        """
        导航到一系列航点
        :param waypoints: 航点列表
        :param Q_search: 障碍物搜索距离（米）
        :param max_speed: 最大飞行速度（米/秒）
        :param waypoint_tolerance: 航点到达容差（米）
        :param debug: 是否打印调试信息
        """
        # 启动异步障碍物检测器（更频繁的检测）
        detector = AsyncObstacleDetector(
            Q_search=Q_search,
            vehicle_name=self.vehicle_name,
            detection_interval=0.1,  # 100ms检测一次，更及时
            port=41451
        )
        
        try:
            detector.start()
            time.sleep(1.0)  # 等待首次检测完成
            
            if debug:
                print("\n" + "=" * 60)
                print("开始RRT导航")
                print("=" * 60)
            
            for wp_idx, waypoint in enumerate(waypoints):
                goal_2d = np.array([waypoint.x_val, waypoint.y_val])
                goal_height = waypoint.z_val
                
                if debug:
                    print(f"\n>>> 前往航点 {wp_idx + 1}/{len(waypoints)}: "
                          f"({goal_2d[0]:.1f}, {goal_2d[1]:.1f}, {goal_height:.1f})")
                
                self._navigate_to_goal(
                    goal_2d=goal_2d,
                    goal_height=goal_height,
                    detector=detector,
                    max_speed=max_speed,
                    waypoint_tolerance=waypoint_tolerance,
                    debug=debug
                )
                
                if debug:
                    print(f"✅ 到达航点 {wp_idx + 1}")
            
            if debug:
                print("\n" + "=" * 60)
                print("所有航点导航完成！")
                print("=" * 60)
                
        finally:
            detector.stop()
    
    def _navigate_to_goal(self,
                         goal_2d: np.ndarray,
                         goal_height: float,
                         detector: AsyncObstacleDetector,
                         max_speed: float,
                         waypoint_tolerance: float,
                         debug: bool):
        """导航到单个目标点"""
        self.current_goal = goal_2d
        self.current_path = None
        self.path_index = 0
        
        loop_count = 0
        
        while True:
            loop_start = time.time()
            
            # 获取当前状态
            state = get_state(self.client, vehicle_name=self.vehicle_name)
            position = np.array(state['position'])
            position_2d = position[0:2]
            yaw = -state['orientation'][2]
            
            # 检查是否到达目标
            distance_to_goal = np.linalg.norm(goal_2d - position_2d)
            if distance_to_goal < waypoint_tolerance:
                if debug:
                    print(f"  到达目标，距离: {distance_to_goal:.2f}m")
                break
            
            # 获取障碍物（添加调试信息）
            obstacles = detector.get_obstacles()
            
            # 调试：检查障碍物数据
            if debug and loop_count % 10 == 0:
                print(f"  [调试] 障碍物类型: {type(obstacles)}, 数量: {len(obstacles) if isinstance(obstacles, list) else 'N/A'}")
                if isinstance(obstacles, list) and len(obstacles) > 0:
                    print(f"  [调试] 第一个障碍物: {obstacles[0]}")
            
            # 检查是否需要重规划
            need_replan = self._should_replan(
                position_2d, obstacles, loop_count, debug
            )
            
            if need_replan or self.current_path is None:
                if debug:
                    if self.current_path is None:
                        print(f"\n  [规划] 初始规划... 当前障碍物数量: {len(obstacles) if isinstance(obstacles, list) else 0}")
                    else:
                        print(f"\n  [重规划] 障碍物变化或偏离路径")
                
                # 构建占据栅格
                occupancy_grid = build_occupancy_grid_from_obstacles(
                    obstacles=obstacles,
                    drone_position=position_2d,
                    drone_yaw=yaw,
                    map_size=self.map_size,
                    resolution=self.grid_resolution,
                    safety_margin=self.safety_margin
                )
                
                # 调试：检查栅格地图
                if debug and self.current_path is None:
                    obstacle_cells = np.sum(occupancy_grid.grid == 1)
                    total_cells = occupancy_grid.grid.size
                    print(f"  [调试] 栅格地图: {obstacle_cells}/{total_cells} 个障碍格子")
                
                # RRT规划
                path = self.planner.plan(
                    start=position_2d,
                    goal=goal_2d,
                    occupancy_grid=occupancy_grid,
                    debug=debug
                )
                
                if path is None:
                    if debug:
                        print("  ⚠️ RRT未找到路径，尝试直接飞向目标")
                    # 如果找不到路径，尝试直接飞
                    path = [position_2d, goal_2d]
                else:
                    # 路径平滑
                    path = smooth_path(path, occupancy_grid, max_iterations=50)
                    if debug:
                        print(f"  平滑后路径: {len(path)}个点")
                
                self.current_path = path
                self.path_index = 0
                self.last_replan_time = time.time()
                self.last_obstacle_count = len(obstacles) if isinstance(obstacles, list) else 0
            
            # 路径跟随
            if self.current_path is not None and len(self.current_path) > 0:
                target_point = self._get_lookahead_point(position_2d)
                
                # 计算速度命令
                direction = target_point - position_2d
                distance = np.linalg.norm(direction)
                
                if distance > 0.1:
                    # 速度控制：距离近时减速，提高安全性
                    speed_factor = min(1.0, distance / 5.0)  # 5米内开始减速
                    target_speed = min(max_speed * speed_factor, distance * 1.0)
                    velocity = (direction / distance) * target_speed
                else:
                    velocity = np.array([0.0, 0.0])
                
                # 发送速度命令
                self.client.moveByVelocityAsync(
                    float(velocity[0]),
                    float(velocity[1]),
                    0,  # 垂直速度由高度控制
                    duration=1.0,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(False, 0),
                    vehicle_name=self.vehicle_name
                )
                
                # 调试输出
                if debug and loop_count % 10 == 0:
                    speed = np.linalg.norm(velocity)
                    obs_count = len(obstacles) if isinstance(obstacles, list) else -999
                    print(f"  [{loop_count:04d}] 位置:({position_2d[0]:.1f},{position_2d[1]:.1f}) "
                          f"距目标:{distance_to_goal:.1f}m 速度:{speed:.2f}m/s "
                          f"障碍:{obs_count} 路径点:{self.path_index}/{len(self.current_path)}")
            
            loop_count += 1
            
            # 控制循环频率
            elapsed = time.time() - loop_start
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
    
    def _should_replan(self, 
                      position_2d: np.ndarray, 
                      obstacles: List,
                      loop_count: int,
                      debug: bool) -> bool:
        """判断是否需要重规划"""
        # 首次规划
        if self.current_path is None:
            return True
        
        # 定期检查
        current_time = time.time()
        if current_time - self.last_replan_time < self.replan_interval:
            return False
        
        # 障碍物数量显著变化
        obstacle_count_change = abs(len(obstacles) - self.last_obstacle_count)
        if obstacle_count_change > 3:
            if debug:
                print(f"  障碍物数量变化: {self.last_obstacle_count} -> {len(obstacles)}")
            return True
        
        # 偏离路径过远
        if self.path_index < len(self.current_path):
            target = self.current_path[self.path_index]
            deviation = np.linalg.norm(position_2d - target)
            if deviation > self.replan_distance_threshold:
                if debug:
                    print(f"  偏离路径: {deviation:.2f}m")
                return True
        
        return False
    
    def _get_lookahead_point(self, position_2d: np.ndarray, lookahead_distance: float = 4.0) -> np.ndarray:
        """获取前瞻点用于路径跟随（增加前瞻距离提高安全性）"""
        if self.current_path is None or len(self.current_path) == 0:
            return self.current_goal if self.current_goal is not None else position_2d
        
        # 更新路径索引（跳过已经过的点）
        while self.path_index < len(self.current_path) - 1:
            dist_to_current = np.linalg.norm(self.current_path[self.path_index] - position_2d)
            if dist_to_current < 2.0:  # 增加到2米，确保已通过
                self.path_index += 1
            else:
                break
        
        # 寻找前瞻点
        for i in range(self.path_index, len(self.current_path)):
            dist = np.linalg.norm(self.current_path[i] - position_2d)
            if dist >= lookahead_distance:
                return self.current_path[i]
        
        # 如果没有足够远的点，返回最后一个点
        return self.current_path[-1]


# ============ 主程序 ============
if __name__ == "__main__":
    # 连接AirSim
    client = airsim.MultirotorClient(port=41451)
    client.confirmConnection()
    client.reset()
    client.enableApiControl(True, vehicle_name='UAV0')
    client.armDisarm(True, vehicle_name='UAV0')
    client.simFlushPersistentMarkers()
    
    # 起飞
    print("起飞中...")
    client.takeoffAsync(vehicle_name='UAV0').join()
    client.moveToZAsync(-3, 1, vehicle_name='UAV0').join()
    print("起飞完成，开始导航\n")
    
    # 定义航点
    waypoints = [
        airsim.Vector3r(60, 0, -3),
        airsim.Vector3r(70, -80, -3),
        airsim.Vector3r(55, -120, -3),
        airsim.Vector3r(0, 0, -3)
    ]
    
    # 创建RRT导航控制器（优化参数提高安全性）
    controller = RRTNavigationController(
        client=client,
        vehicle_name='UAV0',
        map_size=60.0,          # 地图大小60米
        grid_resolution=0.4,    # 0.4米分辨率（更精细）
        safety_margin=2.5,      # 2.5米安全边距（增加！）
        replan_interval=1.5,    # 每1.5秒检查（更频繁）
        replan_distance_threshold=3.0  # 偏离3米触发重规划（更敏感）
    )
    
    # 执行导航
    try:
        controller.navigate_to_waypoints(
            waypoints=waypoints,
            Q_search=15.0,      # 障碍物检测距离15米（增加！）
            max_speed=2.0,      # 最大速度2m/s（降低！更安全）
            waypoint_tolerance=2.5,  # 2.5米到达容差
            debug=True
        )
        
        print("\n任务完成！降落中...")
        client.landAsync(vehicle_name='UAV0').join()
        
    except KeyboardInterrupt:
        print("\n用户中断，降落中...")
        client.landAsync(vehicle_name='UAV0').join()
    
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("紧急降落...")
        client.landAsync(vehicle_name='UAV0').join()
    
    finally:
        client.armDisarm(False, vehicle_name='UAV0')
        client.enableApiControl(False, vehicle_name='UAV0')
        print("程序结束")
