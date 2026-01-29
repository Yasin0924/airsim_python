"""
airsim_avoid.py - V4
智能避障导航系统 - V4版本（向量混合法）

核心思想：
1. 不使用状态机，每帧动态计算飞行方向
2. 飞行方向 = 目标方向 × (1-权重) + 避障方向 × 权重
3. 距离越近，避障权重越大
4. 只有极近(<1米)才后退
"""

import airsim
import numpy as np
import time
import math

def get_obstacle_info_v4(client, vehicle_name='UAV0'):
    """
    V4: 获取障碍物信息，返回距离和相对位置
    :return: (最近障碍物距离, 障碍物水平偏移[-1到1], 左侧安全距离, 右侧安全距离)
    """
    try:
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, 
                              pixels_as_float=True, compress=False)
        ], vehicle_name=vehicle_name)
        
        if len(responses) == 0:
            return 999.0, 0, 999.0, 999.0
        
        depth_img = airsim.get_pfm_array(responses[0])
        height, width = depth_img.shape
        
        # 分区域检测
        center_h_start = int(height * 0.25)
        center_h_end = int(height * 0.75)
        
        # 中心区域（正前方）
        center_start = int(width * 0.35)
        center_end = int(width * 0.65)
        center_region = depth_img[center_h_start:center_h_end, center_start:center_end]
        
        # 左侧区域
        left_region = depth_img[center_h_start:center_h_end, int(width*0.1):center_start]
        
        # 右侧区域
        right_region = depth_img[center_h_start:center_h_end, center_end:int(width*0.9)]
        
        def get_min_dist(region):
            valid = region[np.isfinite(region) & (region > 0.5) & (region < 100)]
            return float(np.min(valid)) if len(valid) > 0 else 999.0
        
        center_dist = get_min_dist(center_region)
        left_dist = get_min_dist(left_region)
        right_dist = get_min_dist(right_region)
        
        # 计算最近障碍物的水平位置
        min_dist = min(center_dist, left_dist, right_dist)
        if min_dist == left_dist:
            h_offset = -0.5  # 障碍物在左边
        elif min_dist == right_dist:
            h_offset = 0.5   # 障碍物在右边
        else:
            # 中心区域，计算精确位置
            valid_mask = np.isfinite(center_region) & (center_region > 0.5) & (center_region < 100)
            if np.any(valid_mask):
                depths = center_region.copy()
                depths[~valid_mask] = 999
                min_idx = np.unravel_index(np.argmin(depths), depths.shape)
                h_offset = (min_idx[1] / (center_end - center_start) - 0.5)
            else:
                h_offset = 0
        
        return center_dist, h_offset, left_dist, right_dist
    
    except Exception as e:
        return 999.0, 0, 999.0, 999.0


def navigate_v4(client, waypoints, vehicle_name='UAV0', 
                max_speed=3.0, waypoint_tolerance=1.5):
    """
    
    """
    
    print("\n" + "=" * 60)
    print("开始V4智能避障导航")
    print("=" * 60)
    
    for wp_idx, waypoint in enumerate(waypoints):
        goal = np.array([waypoint.x_val, waypoint.y_val, waypoint.z_val])
        
        print(f"\n>>> 前往航点 {wp_idx + 1}/{len(waypoints)}: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})")
        
        loop_count = 0
        last_path_point = None
        stuck_count = 0
        last_pos = None
        
        # V5.3: 多阶段脱困
        escape_attempts = 0  # 脱困尝试次数
        
        # V4.3: 方向保持机制 - 大偏转后保持方向一段时间
        hold_direction = None  # 保持的方向
        hold_until = 0  # 保持截止时间
        
        while True:
            # 获取当前状态
            state = client.getMultirotorState(vehicle_name=vehicle_name)
            pos = state.kinematics_estimated.position
            current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
            
            # 检查是否到达
            distance_to_goal = np.linalg.norm(current_pos[:2] - goal[:2])
            if distance_to_goal < waypoint_tolerance:
                print(f"  ✅ 到达航点，距离: {distance_to_goal:.2f}m")
                break
            
            # 获取障碍物信息
            center_dist, h_offset, left_dist, right_dist = get_obstacle_info_v4(client, vehicle_name)
            min_dist = min(center_dist, left_dist, right_dist)
            
            # ============================================
            # V4.5 核心：向量混合法 + 碰撞检测
            # ============================================
            
            # V4.9: 碰撞检测 - 后退+侧移逃离
            collision_info = client.simGetCollisionInfo(vehicle_name=vehicle_name)
            if collision_info.has_collided:
                print(f"  ⚠️ 检测到危险！后退+侧移")
                # 先短距后退0.5秒
                for _ in range(5):
                    client.moveByVelocityAsync(
                        float(-math.cos(current_yaw) * 1.5),
                        float(-math.sin(current_yaw) * 1.5),
                        0,
                        duration=0.2,
                        vehicle_name=vehicle_name
                    )
                    time.sleep(0.1)
                
                # 再向空旷方向侧移1.5秒
                if left_dist > right_dist:
                    escape_angle = current_yaw - math.radians(90)
                else:
                    escape_angle = current_yaw + math.radians(90)
                
                for _ in range(15):
                    client.moveByVelocityAsync(
                        float(math.cos(escape_angle) * 2.0),
                        float(math.sin(escape_angle) * 2.0),
                        0,
                        duration=0.2,
                        vehicle_name=vehicle_name
                    )
                    time.sleep(0.1)
                continue
            
            # 1. 计算目标方向
            target_dir = goal[:2] - current_pos[:2]
            target_dist = np.linalg.norm(target_dir)
            if target_dist > 0.1:
                target_dir = target_dir / target_dist
            else:
                target_dir = np.array([1, 0])
            
            # 获取当前朝向
            orientation = state.kinematics_estimated.orientation
            current_yaw = airsim.to_eularian_angles(orientation)[2]
            
            # 2. 计算避障方向和权重
            avoidance_weight = 0
            avoidance_dir = np.array([0, 0])
            status = "正常"
            
            # V5.6: 渐进式避障 - 增加保持时间确保绕过障碍物
            if center_dist < 10.0:
                # 计算渐进式避障角度和权重
                if center_dist >= 6.0:
                    # 6-10米：微调 (10-20度)
                    base_angle = 10 + (10.0 - center_dist) * 2.5
                    avoidance_weight = 0.2 + (10.0 - center_dist) * 0.05
                    hold_time = 0.5  # 增加：保持0.5秒
                elif center_dist >= 4.0:
                    # 4-6米：中等 (25-40度)
                    base_angle = 25 + (6.0 - center_dist) * 7.5
                    avoidance_weight = 0.4 + (6.0 - center_dist) * 0.1
                    hold_time = 1.0  # 增加：保持1秒
                elif center_dist >= 2.5:
                    # 2.5-4米：较大 (45-65度)
                    base_angle = 45 + (4.0 - center_dist) * 13.3
                    avoidance_weight = 0.7
                    hold_time = 1.5  # 增加：保持1.5秒
                else:
                    # <2.5米：紧急 (75度)
                    base_angle = 75
                    avoidance_weight = 0.95
                    hold_time = 2.0  # 增加：保持2秒
                
                # V5.4: 智能角度调整 - 根据侧向障碍物距离调整角度
                min_safe_dist = 3.0
                
                # 选择空间大的方向
                if left_dist > right_dist:
                    chosen_dist = left_dist
                    is_left = True
                else:
                    chosen_dist = right_dist
                    is_left = False
                
                # 根据侧向距离调整角度
                if chosen_dist < min_safe_dist:
                    # 侧向也有障碍，减小角度穿过去
                    angle_factor = chosen_dist / min_safe_dist  # 0-1
                    adjusted_angle = base_angle * angle_factor * 0.5  # 减半
                    adjusted_angle = max(15, adjusted_angle)  # 最小15度
                    side_note = f"(窄{chosen_dist:.1f}m)"
                else:
                    # 侧向空间足够，正常角度穿过
                    adjusted_angle = base_angle
                    side_note = ""
                
                # 计算避障方向
                if is_left:
                    avoid_angle = current_yaw - math.radians(adjusted_angle)
                    side = f"左{side_note}"
                else:
                    avoid_angle = current_yaw + math.radians(adjusted_angle)
                    side = f"右{side_note}"
                
                # 如果两边都太近，后退
                if left_dist < 2.0 and right_dist < 2.0:
                    avoid_angle = current_yaw + math.radians(180)
                    adjusted_angle = 180
                    side = "后退"
                    avoidance_weight = 1.0
                    hold_time = 0.5
                
                avoidance_dir = np.array([math.cos(avoid_angle), math.sin(avoid_angle)])
                status = f"避障{adjusted_angle:.0f}°→{side}"
                
                # V4.3: 如果角度大，设置方向保持
                current_time = time.time()
                if hold_time > 0 and current_time > hold_until:
                    hold_direction = avoidance_dir.copy()
                    hold_until = current_time + hold_time
            
            # 3. 混合方向（考虑方向保持）
            current_time = time.time()
            if hold_direction is not None and current_time < hold_until:
                # V4.4: 方向保持期内仍检查前方障碍物
                if center_dist < 4.0:
                    # 前方有障碍，中断保持，重新避障
                    hold_direction = None
                    hold_until = 0
                    final_dir = avoidance_dir if avoidance_weight > 0 else target_dir
                    status = f"中断保持，重新避障"
                else:
                    # 前方安全，继续保持方向
                    final_dir = hold_direction
                    status = f"保持方向({hold_until - current_time:.1f}s)"
            elif avoidance_weight > 0:
                final_dir = target_dir * (1 - avoidance_weight) + avoidance_dir * avoidance_weight
                norm = np.linalg.norm(final_dir)
                if norm > 0.01:
                    final_dir = final_dir / norm
                else:
                    final_dir = avoidance_dir
            else:
                final_dir = target_dir
                hold_direction = None  # 清除保持状态
            
            # 4. 计算速度 - V4.8近距离更大减速
            if center_dist < 1.5:
                # 紧急后退
                back_dir = np.array([-math.cos(current_yaw), -math.sin(current_yaw)])
                velocity_2d = back_dir * 1.5
                status = "后退！"
            elif center_dist < 3.0:
                # <3米：大幅减速到50%
                velocity_2d = final_dir * max_speed * 0.5
            elif center_dist < 5.0:
                # 3-5米：减速到60%
                velocity_2d = final_dir * max_speed * 0.6
            elif center_dist < 12.0:
                # 5-12米：线性减速
                speed_factor = 0.6 + (center_dist - 5.0) * 0.057  # 60%->100%
                velocity_2d = final_dir * max_speed * speed_factor
            else:
                velocity_2d = final_dir * max_speed
            
            # 5. 计算yaw（朝向飞行方向）
            yaw_rad = math.atan2(velocity_2d[1], velocity_2d[0])
            
            # 6. 计算垂直速度
            dz = goal[2] - current_pos[2]
            vz = np.clip(dz * 0.5, -1.0, 1.0)
            
            # 7. 执行飞行
            client.moveByVelocityAsync(
                float(velocity_2d[0]),
                float(velocity_2d[1]),
                float(vz),
                duration=0.5,  # 增加duration避免抖动
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, math.degrees(yaw_rad)),
                vehicle_name=vehicle_name
            )
            
            # 日志
            if loop_count % 3 == 0:
                speed = np.linalg.norm(velocity_2d)
                print(f"  [{loop_count:04d}] 位置:({current_pos[0]:.1f},{current_pos[1]:.1f},{current_pos[2]:.1f}) "
                      f"距目标:{distance_to_goal:.1f}m 障碍:中{center_dist:.1f}m/左{left_dist:.1f}m/右{right_dist:.1f}m "
                      f"速度:{speed:.2f}m/s {status}")
            
            # V4.4: 去掉路径绘制，避免被识别为障碍物
            
            # 卡住检测 - V5.3多阶段脱困
            if last_pos is not None:
                movement = np.linalg.norm(current_pos - last_pos)
                if movement < 0.05:
                    stuck_count += 1
                else:
                    stuck_count = 0
                    escape_attempts = 0  # 移动了，重置脱困计数
                
                if stuck_count > 20:  # 2秒没动
                    escape_attempts += 1
                    stuck_count = 0
                    
                    if escape_attempts <= 2:
                        # 第1-2次：左/右侧移
                        if escape_attempts == 1:
                            print(f"  ⚠️ 脱困尝试{escape_attempts}: 左侧移")
                            escape_angle = current_yaw - math.radians(90)
                        else:
                            print(f"  ⚠️ 脱困尝试{escape_attempts}: 右侧移")
                            escape_angle = current_yaw + math.radians(90)
                        
                        for _ in range(20):
                            client.moveByVelocityAsync(
                                float(math.cos(escape_angle) * 2.0),
                                float(math.sin(escape_angle) * 2.0),
                                0,
                                duration=0.2,
                                vehicle_name=vehicle_name
                            )
                            time.sleep(0.1)
                    
                    elif escape_attempts <= 4:
                        # 第3-4次：左后/右后
                        if escape_attempts == 3:
                            print(f"  ⚠️ 脱困尝试{escape_attempts}: 左后方")
                            escape_angle = current_yaw - math.radians(135)
                        else:
                            print(f"  ⚠️ 脱困尝试{escape_attempts}: 右后方")
                            escape_angle = current_yaw + math.radians(135)
                        
                        for _ in range(20):
                            client.moveByVelocityAsync(
                                float(math.cos(escape_angle) * 2.0),
                                float(math.sin(escape_angle) * 2.0),
                                0,
                                duration=0.2,
                                vehicle_name=vehicle_name
                            )
                            time.sleep(0.1)
                    
                    elif escape_attempts <= 6:
                        # 第5-6次：后退
                        print(f"  ⚠️ 脱困尝试{escape_attempts}: 后退")
                        for _ in range(20):
                            client.moveByVelocityAsync(
                                float(-math.cos(current_yaw) * 2.0),
                                float(-math.sin(current_yaw) * 2.0),
                                0,
                                duration=0.2,
                                vehicle_name=vehicle_name
                            )
                            time.sleep(0.1)
                    
                    else:
                        # 第7次+：循环各方向
                        directions = [90, -90, 135, -135, 180, 45, -45]
                        idx = (escape_attempts - 7) % len(directions)
                        angle_deg = directions[idx]
                        print(f"  ⚠️ 脱困尝试{escape_attempts}: {angle_deg}度")
                        escape_angle = current_yaw + math.radians(angle_deg)
                        
                        for _ in range(25):
                            client.moveByVelocityAsync(
                                float(math.cos(escape_angle) * 2.0),
                                float(math.sin(escape_angle) * 2.0),
                                0,
                                duration=0.2,
                                vehicle_name=vehicle_name
                            )
                            time.sleep(0.1)
            
            last_pos = current_pos.copy()
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
    
    # 定义航点
    waypoints = [
        airsim.Vector3r(60, 0, -3),
        airsim.Vector3r(70, -80, -3),
        airsim.Vector3r(55, -120, -3),
        airsim.Vector3r(0, 0, -3)
    ]
    
    try:
        # 清除之前的标记
        client.simFlushPersistentMarkers()
        
        navigate_v4(
            client=client,
            waypoints=waypoints,
            vehicle_name='UAV0',
            max_speed=3.0,
            waypoint_tolerance=3.0
        )
        
        print("\n任务完成！悬停稳定中...")
        client.hoverAsync(vehicle_name='UAV0').join()
        time.sleep(2)  # 悬停2秒稳定
        print("平稳降落中...")
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
