"""
airsim_simple_avoidance.py
简单但可靠的避障方案 - 使用AirSim内置碰撞检测

策略：
1. 朝目标方向飞行
2. 实时检测碰撞
3. 如果即将碰撞，立即转向
4. 不依赖深度相机（深度相机数据不可靠）
"""

import airsim
import numpy as np
import time
import math

def get_distance_2d(pos1, pos2):
    """计算2D距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def navigate_with_collision_avoidance(client, waypoints, vehicle_name='UAV0', 
                                     max_speed=2.0, waypoint_tolerance=3.0,
                                     collision_check_interval=0.1):
    """
    使用碰撞检测的简单避障导航
    :param client: AirSim客户端
    :param waypoints: 航点列表 [Vector3r, ...]
    :param vehicle_name: 无人机名称
    :param max_speed: 最大速度(m/s)
    :param waypoint_tolerance: 航点到达容差(m)
    :param collision_check_interval: 碰撞检查间隔(s)
    """
    
    print("\n" + "=" * 60)
    print("开始碰撞检测避障导航")
    print("=" * 60)
    
    for wp_idx, waypoint in enumerate(waypoints):
        goal = np.array([waypoint.x_val, waypoint.y_val, waypoint.z_val])
        
        print(f"\n>>> 前往航点 {wp_idx + 1}/{len(waypoints)}: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})")
        
        loop_count = 0
        last_collision_time = 0
        avoidance_end_time = 0
        
        while True:
            loop_start = time.time()
            
            # 获取当前状态
            state = client.getMultirotorState(vehicle_name=vehicle_name)
            pos = state.kinematics_estimated.position
            current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
            
            # 检查是否到达
            distance_to_goal = get_distance_2d(current_pos, goal)
            if distance_to_goal < waypoint_tolerance:
                print(f"  ✅ 到达航点，距离: {distance_to_goal:.2f}m")
                break
            
            # 检查碰撞
            collision_info = client.simGetCollisionInfo(vehicle_name=vehicle_name)
            current_time = time.time()
            
            # 如果发生碰撞
            if collision_info.has_collided and (current_time - last_collision_time > 2.0):
                print(f"  ⚠️ 检测到碰撞！执行紧急避让")
                last_collision_time = current_time
                
                # 计算避让位置（向右侧移动5米）
                to_goal = goal[:2] - current_pos[:2]
                to_goal_norm = np.linalg.norm(to_goal)
                if to_goal_norm > 0.1:
                    to_goal_unit = to_goal / to_goal_norm
                    # 向右转90度
                    avoidance_direction = np.array([to_goal_unit[1], -to_goal_unit[0]])
                else:
                    avoidance_direction = np.array([1, 0])
                
                # 计算避让目标位置
                avoidance_pos = current_pos[:2] + avoidance_direction * 5.0
                
                print(f"  避让到位置: ({avoidance_pos[0]:.1f}, {avoidance_pos[1]:.1f})")
                
                # 飞到避让位置（这会自动旋转朝向）
                client.moveToPositionAsync(
                    float(avoidance_pos[0]),
                    float(avoidance_pos[1]),
                    float(current_pos[2]),
                    velocity=max_speed * 0.5,
                    timeout_sec=5,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(False, 0),
                    vehicle_name=vehicle_name
                ).join()
                
                avoidance_end_time = current_time + 1.0
                print(f"  避让完成，继续前往目标")
            
            # 正常飞行：使用moveToPositionAsync（会自动朝向目标）
            if current_time > avoidance_end_time:
                # 计算中间目标点（每次前进3米，避免一次飞太远）
                direction_2d = goal[:2] - current_pos[:2]
                distance_2d = np.linalg.norm(direction_2d)
                
                if distance_2d > 3.0:
                    # 距离较远，飞向中间点
                    direction_unit = direction_2d / distance_2d
                    intermediate_pos = current_pos[:2] + direction_unit * 3.0
                    target_pos = np.array([intermediate_pos[0], intermediate_pos[1], goal[2]])
                else:
                    # 距离较近，直接飞向目标
                    target_pos = goal
                
                # 使用moveToPositionAsync（自动处理朝向）
                client.moveToPositionAsync(
                    float(target_pos[0]),
                    float(target_pos[1]),
                    float(target_pos[2]),
                    velocity=max_speed,
                    timeout_sec=3,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,  # 朝向飞行方向
                    yaw_mode=airsim.YawMode(False, 0),
                    vehicle_name=vehicle_name
                )
                
                # 调试输出
                if loop_count % 5 == 0:
                    collision_status = "碰撞!" if collision_info.has_collided else "正常"
                    print(f"  [{loop_count:04d}] 位置:({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) "
                          f"距目标:{distance_to_goal:.1f}m 状态:{collision_status}")
            
            loop_count += 1
            
            # 控制循环频率
            elapsed = time.time() - loop_start
            sleep_time = max(0, collision_check_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print("\n" + "=" * 60)
    print("所有航点导航完成！")
    print("=" * 60)


# ============ 主程序 ============
if __name__ == "__main__":
    # 连接AirSim
    client = airsim.MultirotorClient(port=41451)
    client.confirmConnection()
    client.reset()
    client.enableApiControl(True, vehicle_name='UAV0')
    client.armDisarm(True, vehicle_name='UAV0')
    
    # 起飞
    print("起飞中...")
    client.takeoffAsync(vehicle_name='UAV0').join()
    client.moveToZAsync(-3, 1, vehicle_name='UAV0').join()
    print("起飞完成\n")
    
    # 定义航点
    waypoints = [
        airsim.Vector3r(60, 0, -3),
        airsim.Vector3r(70, -80, -3),
        airsim.Vector3r(55, -120, -3),
        airsim.Vector3r(0, 0, -3)
    ]
    
    # 执行导航
    try:
        navigate_with_collision_avoidance(
            client=client,
            waypoints=waypoints,
            vehicle_name='UAV0',
            max_speed=2.0,  # 保守的速度
            waypoint_tolerance=3.0,
            collision_check_interval=0.05  # 50ms检查一次碰撞
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
