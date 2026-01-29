"""
airsim_proactive_avoidance.py
主动避障 - 在碰撞前检测并避开障碍物

策略：
1. 使用深度相机检测前方障碍物距离
2. 距离<3米：减速
3. 距离<1米：执行避障操作
4. 朝向始终对准飞行方向
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
        # 获取深度图
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, 
                              pixels_as_float=True, compress=False)
        ], vehicle_name=vehicle_name)
        
        if len(responses) == 0:
            return 999.0
        
        # 解析深度图
        depth_img = airsim.get_pfm_array(responses[0])
        
        # 过滤有效深度值
        valid_depths = depth_img[np.isfinite(depth_img) & (depth_img > 0) & (depth_img < 100)]
        
        if len(valid_depths) == 0:
            return 999.0
        
        # 返回最小距离
        min_dist = np.min(valid_depths)
        return float(min_dist)
    
    except Exception as e:
        print(f"  [警告] 深度检测错误: {e}")
        return 999.0

def navigate_with_proactive_avoidance(client, waypoints, vehicle_name='UAV0', 
                                     max_speed=2.0, waypoint_tolerance=3.0):
    """
    主动避障导航
    """
    
    print("\n" + "=" * 60)
    print("开始主动避障导航")
    print("=" * 60)
    
    for wp_idx, waypoint in enumerate(waypoints):
        goal = np.array([waypoint.x_val, waypoint.y_val, waypoint.z_val])
        
        print(f"\n>>> 前往航点 {wp_idx + 1}/{len(waypoints)}: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})")
        
        loop_count = 0
        avoiding = False
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
            
            # 获取前方障碍物距离
            obstacle_dist = get_min_obstacle_distance(client, vehicle_name)
            
            current_time = time.time()
            
            # 根据障碍物距离决定行为
            if obstacle_dist < 2.0 and not avoiding:
                # 距离<2米：开始避障
                print(f"  ⚠️ 障碍物距离 {obstacle_dist:.2f}m < 2m，开始避障！")
                avoiding = True
                
                # 计算避让方向（向右侧）
                direction_2d = goal[:2] - current_pos[:2]
                direction_norm = np.linalg.norm(direction_2d)
                if direction_norm > 0.1:
                    direction_unit = direction_2d / direction_norm
                    # 向右转90度
                    avoidance_direction = np.array([direction_unit[1], -direction_unit[0]])
                else:
                    avoidance_direction = np.array([1, 0])
                
                avoidance_end_time = current_time + 3.0  # 避让3秒
                print(f"  → 避让方向: ({avoidance_direction[0]:.2f}, {avoidance_direction[1]:.2f})")
            
            # 避让模式
            if avoiding and current_time < avoidance_end_time:
                # 侧向飞行（非阻塞，可以持续检测障碍物）
                velocity_2d = avoidance_direction * max_speed * 0.6
                vz = 0
                
                # 计算朝向角度
                yaw_rad = math.atan2(velocity_2d[1], velocity_2d[0])
                
                # 发送速度命令（非阻塞）
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
                    print(f"  [避让中] 障碍物:{obstacle_dist:.2f}m 剩余:{avoidance_end_time - current_time:.1f}s")
            
            elif current_time >= avoidance_end_time and avoiding:
                # 避让完成
                avoiding = False
                print(f"  ✓ 避让完成，继续前往目标")
            
            # 正常飞行
            if not avoiding:
                # 计算目标方向
                direction_2d = goal[:2] - current_pos[:2]
                distance_2d = np.linalg.norm(direction_2d)
                
                # 根据障碍物距离调整速度
                if obstacle_dist < 5.0:
                    # 距离<5米：减速
                    speed_factor = obstacle_dist / 5.0  # 线性减速
                    actual_speed = max_speed * max(0.2, speed_factor)  # 最低20%速度
                    speed_status = f"减速({speed_factor*100:.0f}%)"
                else:
                    # 距离>=5米：正常速度
                    actual_speed = max_speed
                    speed_status = "正常"
                
                # 计算速度向量
                if distance_2d > 0.1:
                    direction_unit = direction_2d / distance_2d
                    velocity_2d = direction_unit * actual_speed
                    
                    # 计算朝向角度
                    yaw_rad = math.atan2(velocity_2d[1], velocity_2d[0])
                else:
                    velocity_2d = np.array([0, 0])
                    yaw_rad = 0
                
                # 垂直速度控制
                dz = goal[2] - current_pos[2]
                vz = np.clip(dz * 0.5, -1.0, 1.0)
                
                # 发送速度命令（非阻塞）
                client.moveByVelocityAsync(
                    float(velocity_2d[0]),
                    float(velocity_2d[1]),
                    float(vz),
                    duration=0.5,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(False, math.degrees(yaw_rad)),
                    vehicle_name=vehicle_name
                )
                
                # 调试输出
                if loop_count % 3 == 0:
                    print(f"  [{loop_count:04d}] 位置:({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) "
                          f"距目标:{distance_to_goal:.1f}m 障碍物:{obstacle_dist:.2f}m "
                          f"速度:{actual_speed:.2f}m/s {speed_status}")
            
            loop_count += 1
            
            # 控制循环频率
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.1 - elapsed)
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
        navigate_with_proactive_avoidance(
            client=client,
            waypoints=waypoints,
            vehicle_name='UAV0',
            max_speed=2.0,
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
