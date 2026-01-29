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
        
        # V5.15: 分区域检测 - 左35%，中30%，右35%
        center_h_start = int(height * 0.25)
        center_h_end = int(height * 0.75)
        
        # 中心区域（正前方30%）
        center_start = int(width * 0.35)
        center_end = int(width * 0.65)
        center_region = depth_img[center_h_start:center_h_end, center_start:center_end]
        
        # 左侧区域（35%）
        left_region = depth_img[center_h_start:center_h_end, 0:center_start]
        
        # 右侧区域（35%）
        right_region = depth_img[center_h_start:center_h_end, center_end:width]
        
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
    print("开始智能避障导航")
    print("=" * 60)
    
    for wp_idx, waypoint in enumerate(waypoints):
        goal = np.array([waypoint.x_val, waypoint.y_val, waypoint.z_val])
        
        # V5.19: 悬停并转向下一航点，确认安全后再飞
        if wp_idx > 0:
            print(f"  ⏸️ 悬停转向下一航点...")
            # 悬停
            client.hoverAsync(vehicle_name=vehicle_name).join()
            time.sleep(0.5)
            
            # 计算下一航点方向
            state = client.getMultirotorState(vehicle_name=vehicle_name)
            pos = state.kinematics_estimated.position
            current_pos = np.array([pos.x_val, pos.y_val])
            target_dir = goal[:2] - current_pos
            target_yaw = math.atan2(target_dir[1], target_dir[0])
            
            # 逐步转向
            current_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
            for _ in range(20):
                yaw_diff = target_yaw - current_yaw
                while yaw_diff > math.pi: yaw_diff -= 2 * math.pi
                while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
                if abs(yaw_diff) < math.radians(5):
                    break
                yaw_step = max(-math.radians(15), min(math.radians(15), yaw_diff))
                current_yaw += yaw_step
                client.moveByVelocityAsync(0, 0, 0, duration=0.2,
                    yaw_mode=airsim.YawMode(False, math.degrees(current_yaw)),
                    vehicle_name=vehicle_name)
                time.sleep(0.1)
            
            # 确认前方安全
            center_dist, _, _, _ = get_obstacle_info_v4(client, vehicle_name)
            if center_dist < 4.0:
                print(f"  ⚠️ 前方有障碍物({center_dist:.1f}m)，等待避障系统处理")
            else:
                print(f"  ✅ 前方安全({center_dist:.1f}m)，开始飞行")
        
        print(f"\n>>> 前往航点 {wp_idx + 1}/{len(waypoints)}: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})")
        
        loop_count = 0
        last_path_point = None
        stuck_count = 0
        last_pos = None
        
        # V5.3: 多阶段脱困
        escape_attempts = 0  # 脱困尝试次数
        
        # V4.3: 方向保持机制
        hold_direction = None  # 保持的方向
        hold_until = 0  # 保持截止时间
        
        # V5.9: 记住避障触发距离
        trigger_distance = 999  # 避障触发时的障碍物距离
        
        # V5.33: 锁定方向标记（旋转选路后不再扫描）
        direction_locked = False
        
        while True:
            # 获取当前状态
            state = client.getMultirotorState(vehicle_name=vehicle_name)
            pos = state.kinematics_estimated.position
            current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
            
            # 检查是否到达
            distance_to_goal = np.linalg.norm(current_pos[:2] - goal[:2])
            if distance_to_goal < waypoint_tolerance:
                print(f"  ✅ 到达航点 {wp_idx + 1}/{len(waypoints)}: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})")
                break
            
            # 获取障碍物信息
            center_dist, h_offset, left_dist, right_dist = get_obstacle_info_v4(client, vehicle_name)
            min_dist = min(center_dist, left_dist, right_dist)
            
            # ============================================
            # V4.5 核心：向量混合法 + 碰撞检测
            # ============================================
            
            # V5.18: 碰撞检测 - 适中的逃离幅度
            collision_info = client.simGetCollisionInfo(vehicle_name=vehicle_name)
            if collision_info.has_collided:
                print(f"  ⚠️ 检测到危险！小幅后退+侧移")
                
                # 后退0.5秒
                for _ in range(5):
                    client.moveByVelocityAsync(
                        float(-math.cos(current_yaw) * 1.0),
                        float(-math.sin(current_yaw) * 1.0),
                        0,
                        duration=0.2,
                        vehicle_name=vehicle_name
                    )
                    time.sleep(0.1)
                
                # 侧移1秒
                if left_dist > right_dist:
                    escape_angle = current_yaw - math.radians(90)
                else:
                    escape_angle = current_yaw + math.radians(90)
                
                for _ in range(10):
                    client.moveByVelocityAsync(
                        float(math.cos(escape_angle) * 1.0),
                        float(math.sin(escape_angle) * 1.0),
                        0,
                        duration=0.2,
                        vehicle_name=vehicle_name
                    )
                    time.sleep(0.1)
                
                # 重置碰撞状态
                client.simSetVehiclePose(
                    client.simGetVehiclePose(vehicle_name=vehicle_name),
                    ignore_collision=True,
                    vehicle_name=vehicle_name
                )
                
                # V5.34: 碰撞后清除所有状态，重新开始
                hold_direction = None
                hold_until = 0
                direction_locked = False
                time.sleep(0.3)
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
            
            # V5.35: 紧急悬停避障 - 任何方向<2米
            min_all_dist = min(center_dist, left_dist, right_dist)
            if min_all_dist < 2.0 and not direction_locked:
                print(f"  ⚠️ 紧急！障碍物仅{min_all_dist:.1f}m，悬停扫描")
                client.hoverAsync(vehicle_name=vehicle_name)
                time.sleep(0.5)
                
                # 直接进入旋转扫描
                target_yaw_scan = math.atan2(target_dir[1], target_dir[0])
                yaw_diff = target_yaw_scan - current_yaw
                while yaw_diff > math.pi: yaw_diff -= 2 * math.pi
                while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
                rotate_step = math.radians(20) if yaw_diff > 0 else math.radians(-20)
                
                scan_yaw = current_yaw
                best_dir = None
                best_min_dist = 0
                
                for i in range(18):
                    scan_yaw += rotate_step
                    client.moveByVelocityAsync(0, 0, 0, duration=0.5,
                        yaw_mode=airsim.YawMode(False, math.degrees(scan_yaw)),
                        vehicle_name=vehicle_name)
                    time.sleep(0.45)
                    client.hoverAsync(vehicle_name=vehicle_name)
                    time.sleep(0.3)
                    
                    scan_center, _, scan_left, scan_right = get_obstacle_info_v4(client, vehicle_name)
                    scan_min = min(scan_center, scan_left, scan_right)
                    print(f"    扫描{i+1}/18: 中{scan_center:.1f}m 左{scan_left:.1f}m 右{scan_right:.1f}m")
                    
                    if scan_min > best_min_dist:
                        best_min_dist = scan_min
                        best_dir = scan_yaw
                    
                    if scan_center > 5.0 and scan_left > 2.5 and scan_right > 2.5:
                        print(f"    ✅ 找到安全方向")
                        best_dir = scan_yaw
                        client.hoverAsync(vehicle_name=vehicle_name)
                        time.sleep(0.5)
                        break
                
                if best_dir is not None:
                    fly_dir = np.array([math.cos(best_dir), math.sin(best_dir)])
                    hold_direction = fly_dir.copy()
                    hold_until = time.time() + 2.0
                    direction_locked = True
                continue
            
            # 2. 计算避障方向和权重
            avoidance_weight = 0
            avoidance_dir = np.array([0, 0])
            status = "正常"
            
            # V5.15: 渐进式避障 - 减少保持时间
            if center_dist < 6.0:
                if center_dist >= 4.0:
                    # 4-6米：中等 (35-50度)
                    base_angle = 35 + (6.0 - center_dist) * 7.5
                    avoidance_weight = 0.5 + (6.0 - center_dist) * 0.1
                    hold_time = 1.0
                elif center_dist >= 2.5:
                    # 2.5-4米：较大 (55-80度)
                    base_angle = 55 + (4.0 - center_dist) * 16.7
                    avoidance_weight = 0.8
                    hold_time = 1.5
                else:
                    # <2.5米：紧急 (90度)
                    base_angle = 90
                    avoidance_weight = 0.98
                    hold_time = 2.0
                
                # V5.22: 障碍物越正中，角度和时间越大
                # h_offset范围: -0.5(最左) ~ 0(正中) ~ 0.5(最右)
                center_factor = 1.0 - abs(h_offset) * 2  # 0(边缘) ~ 1(正中)
                center_factor = max(0, min(1, center_factor))  # 限制0-1
                
                # 正中障碍物增加20%角度和30%时间
                base_angle = base_angle * (1 + center_factor * 0.2)
                hold_time = hold_time * (1 + center_factor * 0.3)
                
                # V5.16: 智能方向选择 - 结合障碍物位置和侧向距离
                min_safe_dist = 3.0
                
                # 根据中间障碍物位置计算偏好方向
                # h_offset < 0 表示障碍物偏左，应该向右避
                # h_offset > 0 表示障碍物偏右，应该向左避
                if h_offset < -0.1:
                    # 障碍物偏左，优先向右避
                    prefer_right = True
                elif h_offset > 0.1:
                    # 障碍物偏右，优先向左避
                    prefer_right = False
                else:
                    # 障碍物在正中，选择空间大的一侧
                    prefer_right = (right_dist > left_dist)
                
                # 结合侧向距离确认选择
                if prefer_right:
                    if right_dist > min_safe_dist:
                        # 右侧安全，向右避
                        is_left = False
                        chosen_dist = right_dist
                    elif left_dist > right_dist:
                        # 右侧不安全但左侧更好，改向左
                        is_left = True
                        chosen_dist = left_dist
                    else:
                        # 右侧虽不理想但仍选右
                        is_left = False
                        chosen_dist = right_dist
                else:
                    if left_dist > min_safe_dist:
                        # 左侧安全，向左避
                        is_left = True
                        chosen_dist = left_dist
                    elif right_dist > left_dist:
                        # 左侧不安全但右侧更好，改向右
                        is_left = False
                        chosen_dist = right_dist
                    else:
                        # 左侧虽不理想但仍选左
                        is_left = True
                        chosen_dist = left_dist
                
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
                
                # V5.9: 记住避障触发距离
                current_time = time.time()
                if hold_time > 0 and current_time > hold_until:
                    hold_direction = avoidance_dir.copy()
                    hold_until = current_time + hold_time
                    trigger_distance = center_dist  # 保存触发距离
            
            # 3. V5.33: 安全回归策略
            current_time = time.time()
            if hold_direction is not None and current_time < hold_until:
                # V5.33: 如果方向已锁定，跳过扫描，直接保持方向飞行
                if direction_locked:
                    final_dir = hold_direction
                    status = f"保持飞行({hold_until - current_time:.1f}s)"
                    # 锁定期结束后解锁
                    if current_time >= hold_until:
                        direction_locked = False
                # 保持期内检查前方障碍物（未锁定时）
                elif center_dist < 4.0:
                    # V5.26: 前方有障碍，悬停并旋转扫描寻找安全路径
                    print(f"  ⏸️ 保持期发现障碍({center_dist:.1f}m)，悬停扫描安全路径")
                    client.hoverAsync(vehicle_name=vehicle_name)
                    time.sleep(0.5)
                    
                    # 计算向途径点方向旋转(确定是左转还是右转)
                    target_yaw = math.atan2(target_dir[1], target_dir[0])
                    yaw_diff = target_yaw - current_yaw
                    while yaw_diff > math.pi: yaw_diff -= 2 * math.pi
                    while yaw_diff < -math.pi: yaw_diff += 2 * math.pi
                    rotate_step = math.radians(20) if yaw_diff > 0 else math.radians(-20)
                    
                    # V5.28: 慢速旋转扫描，每个角度充分停留
                    scan_yaw = current_yaw
                    best_dir = None
                    best_min_dist = 0
                    found_safe = False
                    
                    for i in range(18):  # 扫描18个方向(每20度)
                        # 先转向
                        scan_yaw += rotate_step
                        client.moveByVelocityAsync(0, 0, 0, duration=0.8,
                            yaw_mode=airsim.YawMode(False, math.degrees(scan_yaw)),
                            vehicle_name=vehicle_name)
                        time.sleep(0.6)  # 等待转向稳定
                        
                        # 悬停稳定后再检测
                        client.hoverAsync(vehicle_name=vehicle_name)
                        time.sleep(0.3)  # 稳定
                        
                        # 检测左中右距离
                        scan_center, _, scan_left, scan_right = get_obstacle_info_v4(client, vehicle_name)
                        min_dist = min(scan_center, scan_left, scan_right)
                        print(f"    扫描{i+1}/18: 中{scan_center:.1f}m 左{scan_left:.1f}m 右{scan_right:.1f}m")
                        
                        # 记录最佳方向（在检测完成后）
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_dir = scan_yaw
                        
                        # V5.32: 安全条件 - 中间>5m，左右>2.5m
                        if scan_center > 5.0 and scan_left > 2.5 and scan_right > 2.5:
                            print(f"    ✅ 找到安全方向: 中>5m 左右>2.5m，立即固定")
                            found_safe = True
                            best_dir = scan_yaw
                            # 立即悬停固定当前朝向
                            client.hoverAsync(vehicle_name=vehicle_name)
                            time.sleep(0.5)
                            break
                    
                    # 转向最佳方向并飞行
                    if best_dir is not None:
                        # V5.29: 先完全悬停
                        client.hoverAsync(vehicle_name=vehicle_name)
                        time.sleep(0.3)
                        
                        # 明确转到最佳方向
                        print(f"    🎯 转向安全方向: {math.degrees(best_dir):.0f}°")
                        client.moveByVelocityAsync(0, 0, 0, duration=1.0,
                            yaw_mode=airsim.YawMode(False, math.degrees(best_dir)),
                            vehicle_name=vehicle_name)
                        time.sleep(0.8)
                        
                        # 悬停稳定
                        client.hoverAsync(vehicle_name=vehicle_name)
                        time.sleep(0.3)
                        
                        # 再次确认安全
                        confirm_center, _, confirm_left, confirm_right = get_obstacle_info_v4(client, vehicle_name)
                        print(f"    📍 确认: 中{confirm_center:.1f}m 左{confirm_left:.1f}m 右{confirm_right:.1f}m")
                        
                        if confirm_center > 4.0 and confirm_left > 2.5 and confirm_right > 2.5:
                            # 渐进加速启动（使用best_dir方向）
                            print(f"    🚀 安全确认，渐进加速...")
                            fly_dir = np.array([math.cos(best_dir), math.sin(best_dir)])
                            for speed in [0.3, 0.6, 1.0]:
                                client.moveByVelocityAsync(
                                    float(fly_dir[0] * speed),
                                    float(fly_dir[1] * speed),
                                    0, duration=0.5,
                                    yaw_mode=airsim.YawMode(False, math.degrees(best_dir)),
                                    vehicle_name=vehicle_name)
                                time.sleep(0.4)
                            
                            # V5.33: 锁定选择的方向2秒
                            hold_direction = fly_dir.copy()
                            hold_until = time.time() + 2.0
                            direction_locked = True
                            continue
                        else:
                            print(f"    ⚠️ 不安全，重新扫描...")
                    
                    # 未找到安全方向，重新避障
                    hold_direction = None
                    hold_until = 0
                    continue
                else:
                    # 前方安全，继续保持方向
                    final_dir = hold_direction
                    status = f"保持方向({hold_until - current_time:.1f}s)"
            
            elif hold_direction is not None and current_time >= hold_until:
                # V5.20: 保持期结束，进入安全确认阶段
                if center_dist < 6.0:
                    # 前方仍有障碍，延长保持1秒
                    hold_until = current_time + 1.0
                    final_dir = hold_direction
                    status = f"延长保持(前方{center_dist:.1f}m)"
                else:
                    # 前方安全，渐进回归
                    final_dir = target_dir * 0.5 + hold_direction * 0.5
                    norm = np.linalg.norm(final_dir)
                    if norm > 0.01:
                        final_dir = final_dir / norm
                    hold_direction = None
                    # V5.20: 设置安全确认标记
                    safety_check_until = current_time + 1.0  # 1秒安全确认期
                    status = f"渐进回归"
            
            # V5.23: 安全确认阶段 - 发现障碍物时悬停重新选择路径
            elif 'safety_check_until' in dir() and current_time < safety_check_until:
                if center_dist < 5.0:
                    # 前方有新障碍，悬停重新评估
                    print(f"  ⏸️ 安全确认发现障碍({center_dist:.1f}m)，悬停重新避障")
                    client.hoverAsync(vehicle_name=vehicle_name)
                    time.sleep(0.3)
                    
                    # 清除安全确认，让避障逻辑重新接管
                    del safety_check_until
                    
                    # 强制进入避障状态
                    avoidance_weight = 0.8
                    status = f"回归后检测到障碍({center_dist:.1f}m)"
                    continue  # 重新开始循环，执行避障
                else:
                    # 前方安全，继续低速
                    final_dir = target_dir
                    status = f"安全确认中({safety_check_until - current_time:.1f}s)"
            
            elif avoidance_weight > 0:
                final_dir = target_dir * (1 - avoidance_weight) + avoidance_dir * avoidance_weight
                norm = np.linalg.norm(final_dir)
                if norm > 0.01:
                    final_dir = final_dir / norm
                else:
                    final_dir = avoidance_dir
            else:
                final_dir = target_dir
                # 清除安全确认标记
                if 'safety_check_until' in dir():
                    del safety_check_until
            
            # 4. V5.20: 速度控制
            # 标记状态
            is_returning = (status == "渐进回归")
            is_safety_check = ("安全确认" in status)
            in_avoidance = (hold_direction is not None) or ("避障" in status) or ("保持" in status)
            
            if center_dist < 1.5:
                # 紧急后退
                back_dir = np.array([-math.cos(current_yaw), -math.sin(current_yaw)])
                velocity_2d = back_dir * 1.5
                status = "后退！"
            elif is_returning:
                # V5.9: 根据触发距离决定回归速度
                if trigger_distance < 3.0:
                    velocity_2d = final_dir * max_speed * 0.15
                elif trigger_distance < 5.0:
                    velocity_2d = final_dir * max_speed * 0.2
                else:
                    velocity_2d = final_dir * max_speed * 0.3
                trigger_distance = 999
            elif is_safety_check:
                # V5.21: 安全确认阶段根据距离调速
                if center_dist < 3.0:
                    velocity_2d = final_dir * 0.3  # 0.3m/s
                elif center_dist < 6.0:
                    velocity_2d = final_dir * 1.0  # 1.0m/s
                else:
                    velocity_2d = final_dir * max_speed * 0.5  # 确认安全，提速
            elif in_avoidance:
                # V5.15: 避障过程中提高速度
                if center_dist < 2.5:
                    velocity_2d = final_dir * 0.8  # 0.8m/s
                elif center_dist < 4.0:
                    velocity_2d = final_dir * max_speed * 0.5  # 1.5m/s
                else:
                    velocity_2d = final_dir * max_speed * 0.6  # 1.8m/s
            elif center_dist < 2.5:
                velocity_2d = final_dir * 0.5  # 0.5m/s
            elif center_dist < 4.0:
                velocity_2d = final_dir * max_speed * 0.35
            elif center_dist < 6.0:
                velocity_2d = final_dir * max_speed * 0.5
            elif center_dist < 10.0:
                speed_factor = 0.5 + (center_dist - 6.0) * 0.125
                velocity_2d = final_dir * max_speed * speed_factor
            else:
                velocity_2d = final_dir * max_speed
            
            # 5. V5.14: 计算yaw并限制旋转速度
            target_yaw = math.atan2(velocity_2d[1], velocity_2d[0])
            yaw_diff = target_yaw - current_yaw
            # 归一化到-pi到pi
            while yaw_diff > math.pi:
                yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi:
                yaw_diff += 2 * math.pi
            # V5.14: 回归时更慢的转向
            if is_returning:
                max_yaw_change = math.radians(10)  # 回归时每次最多10度
            else:
                max_yaw_change = math.radians(25)  # 正常每次最多25度
            yaw_diff = max(-max_yaw_change, min(max_yaw_change, yaw_diff))
            yaw_rad = current_yaw + yaw_diff
            
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
