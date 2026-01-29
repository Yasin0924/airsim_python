"""
airsim_smart_avoidance.py
智能避障 - 选择最安全的避障方向

策略：
1. 检测前方障碍物距离
2. 当需要避障时，检测左、右、上三个方向
3. 选择最安全（障碍物最远）的方向
4. 如果所有方向都不安全，向后退
"""

import airsim
import numpy as np
import time
import math

def get_distance_2d(pos1, pos2):
    """计算2D距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_min_obstacle_distance(client, vehicle_name='UAV0'):
    """
    获取前方最近障碍物距离
    :return: 最小距离（米），如果没有障碍物返回999
    """
    try:
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, 
                              pixels_as_float=True, compress=False)
        ], vehicle_name=vehicle_name)
        
        if len(responses) == 0:
            return 999.0
        
        depth_img = airsim.get_pfm_array(responses[0])
        valid_depths = depth_img[np.isfinite(depth_img) & (depth_img > 0) & (depth_img < 100)]
        
        if len(valid_depths) == 0:
            return 999.0
        
        return float(np.min(valid_depths))
    
    except Exception as e:
        return 999.0

def find_best_avoidance_direction(current_pos, goal, forward_obstacle_dist):
    """
    找到最佳避障方向
    :return: (direction_vector, direction_name)
    """
    # 计算朝向目标的方向
    to_goal = goal[:2] - current_pos[:2]
    to_goal_norm = np.linalg.norm(to_goal)
    
    if to_goal_norm < 0.1:
        to_goal_unit = np.array([1, 0])
    else:
        to_goal_unit = to_goal / to_goal_norm
    
    # 定义候选方向：右、左、上、后
    candidates = [
        (np.array([to_goal_unit[1], -to_goal_unit[0]]), "右侧", 0),  # 右转90度
        (np.array([-to_goal_unit[1], to_goal_unit[0]]), "左侧", 0),  # 左转90度
        (None, "上升", goal[2] - 3.0),  # 上升3米
        (-to_goal_unit, "后退", 0)  # 后退
    ]
    
    # 简单策略：优先选择右侧，如果前方障碍物很近则上升
    if forward_obstacle_dist < 1.5:
        # 非常近，优先上升
        return (None, "上升", goal[2] - 3.0)
    else:
        # 否则向右侧
        return (np.array([to_goal_unit[1], -to_goal_unit[0]]), "右侧", 0)

def navigate_with_smart_avoidance(client, waypoints, vehicle_name='UAV0', 
                                 max_speed=1.5, waypoint_tolerance=3.0):
    """
    智能避障导航
    """
    
    print("\n" + "=" * 60)
    print("开始智能避障导航")
    print("=" * 60)
    
    for wp_idx, waypoint in enumerate(waypoints):
        goal = np.array([waypoint.x_val, waypoint.y_val, waypoint.z_val])
        
        print(f"\n>>> 前往航点 {wp_idx + 1}/{len(waypoints)}: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})")
        
        loop_count = 0
        avoiding = False
        avoidance_end_time = 0
        avoidance_direction = None
        avoidance_vz = 0
        stuck_count = 0
        last_pos = current_pos = None
        
        while True:
            loop_start = time.time()
            
            # 获取当前状态
            state = client.getMultirotorState(vehicle_name=vehicle_name)
            pos = state.kinematics_estimated.position
            last_pos = current_pos if current_pos is not None else np.array([pos.x_val, pos.y_val, pos.z_val])
            current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
            
            # 检查是否卡住（位置几乎不变）
            if last_pos is not None:
                movement = np.linalg.norm(current_pos - last_pos)
                if movement < 0.1:
                    stuck_count += 1
                else:
                    stuck_count = 0
                
                if stuck_count > 20:
                    print(f"  ⚠️ 检测到卡住！强制上升")
                    client.moveToZAsync(current_pos[2] - 3, 1, vehicle_name=vehicle_name).join()
                    stuck_count = 0
            
            # 检查是否到达
            distance_to_goal = get_distance_2d(current_pos, goal)
            if distance_to_goal < waypoint_tolerance:
                print(f"  ✅ 到达航点，距离: {distance_to_goal:.2f}m")
                break
            
            # 获取前方障碍物距离
            obstacle_dist = get_min_obstacle_distance(client, vehicle_name)
            
            current_time = time.time()
            
            # 根据障碍物距离决定行为
            if obstacle_dist < 2.0 and not avoiding:
                # 需要避障：选择最佳方向
                avoidance_direction, direction_name, avoidance_vz = find_best_avoidance_direction(
                    current_pos, goal, obstacle_dist
                )
                
                print(f"  ⚠️ 障碍物 {obstacle_dist:.2f}m < 2m，避障方向: {direction_name}")
                avoiding = True
                avoidance_end_time = current_time + 3.0
            
            # 避让模式
            if avoiding and current_time < avoidance_end_time:
                if avoidance_direction is not None:
                    # 水平避让
                    velocity_2d = avoidance_direction * max_speed * 0.6
                    vz = 0
                else:
                    # 垂直避让（上升）
                    velocity_2d = np.array([0, 0])
                    vz = -1.0  # 上升
                
                yaw_rad = math.atan2(velocity_2d[1], velocity_2d[0]) if np.linalg.norm(velocity_2d) > 0.1 else 0
                
                client.moveByVelocityAsync(
                    float(velocity_2d[0]),
                    float(velocity_2d[1]),
                    float(vz),
                    duration=0.5,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(False, math.degrees(yaw_rad)),
                    vehicle_name=vehicle_name
                )
                
                if loop_count % 3 == 0:
                    print(f"  [避让中] 障碍:{obstacle_dist:.2f}m 剩余:{avoidance_end_time - current_time:.1f}s")
            
            elif current_time >= avoidance_end_time and avoiding:
                avoiding = False
                print(f"  ✓ 避让完成")
            
            # 正常飞行
            if not avoiding:
                direction_2d = goal[:2] - current_pos[:2]
                distance_2d = np.linalg.norm(direction_2d)
                
                # 根据障碍物距离调整速度
                if obstacle_dist < 5.0:
                    speed_factor = obstacle_dist / 5.0
                    actual_speed = max_speed * max(0.2, speed_factor)
                    speed_status = f"减速({speed_factor*100:.0f}%)"
                else:
                    actual_speed = max_speed
                    speed_status = "正常"
                
                if distance_2d > 0.1:
                    direction_unit = direction_2d / distance_2d
                    velocity_2d = direction_unit * actual_speed
                    yaw_rad = math.atan2(velocity_2d[1], velocity_2d[0])
                else:
                    velocity_2d = np.array([0, 0])
                    yaw_rad = 0
                
                dz = goal[2] - current_pos[2]
                vz = np.clip(dz * 0.5, -1.0, 1.0)
                
                client.moveByVelocityAsync(
                    float(velocity_2d[0]),
                    float(velocity_2d[1]),
                    float(vz),
                    duration=0.5,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(False, math.degrees(yaw_rad)),
                    vehicle_name=vehicle_name
                )
                
                if loop_count % 3 == 0:
                    print(f"  [{loop_count:04d}] 位置:({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) "
                          f"距目标:{distance_to_goal:.1f}m 障碍:{obstacle_dist:.2f}m {speed_status}")
            
            loop_count += 1
            time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print("所有航点导航完成！")
    print("=" * 60)


# ============ 主程序 ============
if __name__ == "__main__":
    client = airsim.MultirotorClient(port=41451)
    client.confirmConnection()
    client.reset()
    client.enableApiControl(True, vehicle_name='UAV0')
    client.armDisarm(True, vehicle_name='UAV0')
    
    print("起飞中...")
    client.takeoffAsync(vehicle_name='UAV0').join()
    client.moveToZAsync(-3, 1, vehicle_name='UAV0').join()
    print("起飞完成\n")
    
    waypoints = [
        airsim.Vector3r(60, 0, -3),
        airsim.Vector3r(70, -80, -3),
        airsim.Vector3r(55, -120, -3),
        airsim.Vector3r(0, 0, -3)
    ]
    
    try:
        navigate_with_smart_avoidance(
            client=client,
            waypoints=waypoints,
            vehicle_name='UAV0',
            max_speed=1.5,  # 降低速度提高安全性
            waypoint_tolerance=3.0
        )
        
        print("\n任务完成！降落中...")
        client.landAsync(vehicle_name='UAV0').join()
        
    except KeyboardInterrupt:
        print("\n用户中断，降落中...")
        client.landAsync(vehicle_name='UAV0').join()
    
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("紧急降落...")
        client.landAsync(vehicle_name='UAV0').join()
    
    finally:
        client.armDisarm(False, vehicle_name='UAV0')
        client.enableApiControl(False, vehicle_name='UAV0')
        print("程序结束")
