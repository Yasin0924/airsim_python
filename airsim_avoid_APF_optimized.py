"""
airsim_avoid_APF_optimized.py
CarrotChasing轨迹跟踪算法与APF避障算法融合 - 优化版本 V9

关键改进：
- 卡顿检测与自动逃逸策略 (Stuck Recovery)
- 解决APF陷入局部极小值的问题
- 距离没动且有障碍时，强制向侧向飞行
"""

import math
import numpy as np
import airsim
import time
import threading
from ObstacleDetection.obstacles_detect import obstacles_detect
from UavAgent import get_state
from mymath import distance, myatan, isClockwise


class VelocityFilter:
    """低通滤波器，平滑速度命令"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_v = None
    
    def filter(self, v_new):
        if self.prev_v is None:
            self.prev_v = v_new
            return v_new
        v_filtered = self.alpha * np.array(v_new) + (1 - self.alpha) * self.prev_v
        self.prev_v = v_filtered
        return v_filtered


class BackgroundObstacleDetector:
    """后台障碍物检测器 - 使用独立的AirSim连接，避免IOLoop冲突"""
    def __init__(self, Q_search, vehicle_name='', port=41451):
        self.Q_search = Q_search
        self.vehicle_name = vehicle_name
        self.port = port
        self._lock = threading.Lock()
        self._obstacles = []
        self._running = False
        self._thread = None
        self._detection_time = 0
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[后台检测] 已启动")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[后台检测] 已停止")
    
    def get_obstacles(self):
        with self._lock:
            return list(self._obstacles), self._detection_time
    
    def _loop(self):
        # 在后台线程中创建独立的AirSim客户端连接
        try:
            detector_client = airsim.MultirotorClient(port=self.port)
            detector_client.confirmConnection()
            print(f"[后台检测] 独立连接已建立 (port={self.port})")
        except Exception as e:
            print(f"[后台检测] 无法建立独立连接: {e}")
            return
        
        while self._running:
            try:
                t_start = time.time()
                obstacles = obstacles_detect(detector_client, self.Q_search, vehicle_name=self.vehicle_name)
                detection_time = time.time() - t_start
                with self._lock:
                    self._obstacles = obstacles
                    self._detection_time = detection_time
            except Exception as e:
                print(f"[后台检测] 错误: {e}")
            time.sleep(0.1)  # 检测间隔100ms


def move_by_path_and_avoid_APF(client, Path, K_track=None, delta=1, K_avoid=None,
                               Q_search=10, epsilon=2, Ul=None, dt=0.3, vehicle_name='UAV0',
                               enable_plot=False, debug=True):
    """优化后的APF避障飞行 V7"""
    [K0, K1, K2] = K_track
    [Kg, Kr] = K_avoid
    [Ul_avoid, Ul_track] = Ul

    state = get_state(client, vehicle_name=vehicle_name)
    P_start = np.array(state['position'])
    V_start = np.array(state['linear_velocity'])[0:2]
    pos_record = [P_start]
    I2 = np.matrix([[1, 0], [0, 1]])

    count = 0
    P_curr = P_start
    V_curr = V_start
    V_last = np.array([0, 0])
    height = -3
    Wb = P_curr[0:2]
    nowtime, lasttime = time.time(), time.time()

    velocity_filter = VelocityFilter(alpha=0.4)

    # 启动后台障碍物检测器（使用独立连接避免IOLoop冲突）
    detector = BackgroundObstacleDetector(Q_search, vehicle_name, port=41451)
    detector.start()
    time.sleep(1.0)  # 等待后台连接建立和首次检测

    if enable_plot:
        plot_p2 = [airsim.Vector3r(P_curr[0], P_curr[1], P_curr[2])]

    # V9: 卡顿检测变量
    last_check_pos = P_curr
    last_check_time = time.time()
    stuck_count = 0
    is_escaping = False
    escape_start_time = 0
    escape_dir = np.array([0, 0])

    try:
        for path_num in range(len(Path)):
            Wa = Wb
            Wb = np.array([Path[path_num].x_val, Path[path_num].y_val])
            Wa_sub_Wb_matrix = np.matrix((Wa - Wb)).T
            dist_wa_wb = distance(Wa, Wb)
            if dist_wa_wb < 0.01:
                continue
            A = I2 - Wa_sub_Wb_matrix.dot(Wa_sub_Wb_matrix.T) / (dist_wa_wb ** 2)
            Pt_matrix = np.matrix(P_curr[0:2] - Wb).T
            d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))

            print(f"\n=== 前往航路点 {path_num+1}/{len(Path)}: ({Wb[0]:.1f}, {Wb[1]:.1f}) ===")

            # V8改进：只根据距离判断是否到达，移除 d_to_Wb > delta 条件防止意外切换
            while distance(Wb, P_curr[0:2]) > epsilon:

                loop_start = time.time()

                # 1. 获取状态
                state = get_state(client, vehicle_name=vehicle_name)
                P_curr = np.array(state['position'])
                V_curr = np.array(state['linear_velocity'])
                yaw = -state['orientation'][2]

                Pt_matrix = np.matrix(P_curr[0:2] - Wb).T
                Frep = np.array([0, 0])

                # 2. 获取障碍物（从后台线程读取，不阻塞！）
                info_obstacles, time_detection = detector.get_obstacles()

                # 3. 计算斥力
                num_obstacles = 0
                Rz = np.array([[math.cos(yaw), math.sin(yaw)],
                               [-math.sin(yaw), math.cos(yaw)]])

                for obstacle in info_obstacles:
                    pos_obstacle_min = obstacle.min
                    pos_obstacle_max = obstacle.max
                    P_search_list = np.array([
                        np.array([pos_obstacle_min.x_val, pos_obstacle_min.y_val]),
                        np.array([pos_obstacle_min.x_val, pos_obstacle_max.y_val]),
                        np.array([pos_obstacle_max.x_val, pos_obstacle_min.y_val]),
                        np.array([pos_obstacle_max.x_val, pos_obstacle_max.y_val])
                    ]).dot(Rz.T)
                    for P_search in P_search_list:
                        d_search = np.linalg.norm(P_search)
                        if 0.1 < d_search <= Q_search:
                            num_obstacles += 1
                            Frep = Frep + Kr * (1 / d_search - 1 / Q_search) / (d_search ** 3) * \
                                   (-P_search) * (distance(P_search + P_curr[0:2], Wb) ** 2)

                # 4. 计算控制量 (正常APF)
                if num_obstacles == 0:
                    U1 = np.array((K0 * Pt_matrix + K1 * A.dot(Pt_matrix)).T)[0]
                    if np.linalg.norm(U1, ord=np.inf) > Ul_track:
                        U1 = U1 * Ul_track / np.linalg.norm(U1, ord=np.inf)
                    U = -(U1 + V_curr[0:2]) / K2
                else:
                    # 原有APF逻辑
                    Frep = Frep / num_obstacles
                    Fatt = -Kg * (P_curr[0:2] - Wb)
                    # (省略中间的Carrot Chasing逻辑保留原样)
                    
                    if count >= 2:
                        # ... 原有Carrot Chasing逻辑保持 ...
                        p0 = pos_record[-1]
                        p1 = pos_record[-2]
                        nowtime = time.time()
                        time_diff = nowtime - lasttime
                        Vra = (distance(p0, Wb) - distance(p1, Wb)) / time_diff if time_diff > 0.01 else 0
                        lasttime = nowtime
                        
                        if abs(Vra) < 0.95 * Ul_avoid and len(info_obstacles) != 0 \
                                and np.linalg.norm(V_curr[0:2], ord=np.inf) < np.linalg.norm(V_last, ord=np.inf):
                            # ... 原有旋转逻辑 ...
                            angle_g = myatan([0, 0], [Fatt[0], Fatt[1]])
                            angle_g = 0 if angle_g is None else angle_g
                            angle_r = myatan([0, 0], [Frep[0], Frep[1]])
                            angle_r = 0 if angle_r is None else angle_r
                            theta = 60 * math.pi / 180 if isClockwise(angle_g, angle_r) else -60 * math.pi / 180
                            Frep = [math.cos(theta) * Frep[0] - math.sin(theta) * Frep[1],
                                    math.sin(theta) * Frep[0] + math.cos(theta) * Frep[1]]

                        l = Ul_avoid
                        Kv = 3 * l / (2 * l + abs(Vra))
                        Kd = 15 * math.exp(-(distance(P_curr[0:2], Wb) - 1.5) ** 2 / 2) + 1
                        Ke = 5
                        Fatt = Kv * Kd * Ke * Fatt
                        
                    U = Fatt + Frep
                    if np.linalg.norm(U, ord=np.inf) > Ul_avoid:
                        U = Ul_avoid * U / np.linalg.norm(U, ord=np.inf)
                    U = (U - V_curr[0:2]) / K2

                # ===== V9: 卡顿检测与逃逸覆盖 =====
                current_time = time.time()
                
                if is_escaping:
                    # 逃逸模式：强制侧向飞行2秒
                    if current_time - escape_start_time > 2.0:
                        is_escaping = False
                        stuck_count = 0
                        print(">>> 逃逸完成，恢复正常控制")
                    else:
                        U = escape_dir * 3.0  # 3m/s 侧向强力飞行
                        if count % 10 == 0:
                            print(f">>> 正在逃逸... {current_time - escape_start_time:.1f}s")
                else:
                    # 正常模式：检查是否卡顿
                    check_interval = 2.0
                    if current_time - last_check_time > check_interval:
                        dist_moved = np.linalg.norm(P_curr[0:2] - last_check_pos[0:2])
                        if dist_moved < 0.5 and num_obstacles > 0:
                            stuck_count += 1
                            print(f"!!! 检测到卡顿 ({stuck_count}/2) 2秒才动了{dist_moved:.2f}m")
                        else:
                            stuck_count = 0 # 只要动了就重置
                        
                        last_check_pos = P_curr
                        last_check_time = current_time

                        if stuck_count >= 2:
                            # 触发逃逸
                            is_escaping = True
                            escape_start_time = current_time
                            dir_to_target = Wb - P_curr[0:2]
                            norm_dir = np.linalg.norm(dir_to_target)
                            if norm_dir > 0.1:
                                dir_to_target = dir_to_target / norm_dir
                                # 默认向右逃逸 (顺时针90度): (x,y) -> (y, -x)
                                escape_dir = np.array([dir_to_target[1], -dir_to_target[0]])
                            else:
                                escape_dir = np.array([1, 0]) # 默认向东
                            print(f"!!! 确认卡住，触发侧向逃逸策略！方向: {escape_dir}")

                V_next = V_curr[0:2] + U * dt
                speed = np.linalg.norm(V_next)
                max_speed = 3.0  # V8: 降低最大速度到3m/s，适应1秒的检测延迟
                if speed > max_speed:
                    V_next = V_next * max_speed / speed

                # 速度平滑滤波
                V_smooth = velocity_filter.filter(V_next)

                # 发送速度命令
                client.moveByVelocityAsync(
                    float(V_smooth[0]), 
                    float(V_smooth[1]), 
                    0,
                    duration=2.0,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(False, 0),
                    vehicle_name=vehicle_name
                )

                # 绘图
                if enable_plot and count % 3 == 0:
                    plot_p1 = plot_p2
                    plot_p2 = [airsim.Vector3r(P_curr[0], P_curr[1], P_curr[2])]
                    client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)

                # 更新记录
                V_last = V_curr
                d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))
                pos_record.append(P_curr)
                count += 1

                loop_total = time.time() - loop_start

                # 调试输出
                if debug and count % 10 == 0:
                    dist_to_target = distance(Wb, P_curr[0:2])
                    actual_speed = np.linalg.norm(V_curr[0:2])
                    print(f"[{count:04d}] 位置:({P_curr[0]:.1f},{P_curr[1]:.1f}) "
                          f"距目标:{dist_to_target:.1f}m 速度:{actual_speed:.2f}m/s "
                          f"障碍:{num_obstacles} 后台检测:{time_detection*1000:.0f}ms 循环:{loop_total*1000:.0f}ms")

                # 最小间隔避免过载
                elapsed = time.time() - loop_start
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)
    finally:
        detector.stop()


if __name__ == "__main__":
    client = airsim.MultirotorClient(port=41451)
    client.confirmConnection()
    client.reset()
    client.enableApiControl(True, vehicle_name='UAV0')
    client.armDisarm(True, vehicle_name='UAV0')
    client.simFlushPersistentMarkers()
    client.takeoffAsync(vehicle_name='UAV0').join()
    client.moveToZAsync(-3, 1, vehicle_name='UAV0').join()

    points = [airsim.Vector3r(60, 0, -3),
              airsim.Vector3r(70, -80, -3),
              airsim.Vector3r(55, -120, -3),
              airsim.Vector3r(0, 0, -3)]

    print("=" * 50)
    print("开始避障飞行 (V9 - 卡顿检测 & 自动逃逸)")
    print("=" * 50)
    
    move_by_path_and_avoid_APF(
        client, points,
        K_track=[6.0, 10, 0.4],  # 略微增大K0
        delta=5,
        K_avoid=[15, 80],  # V8: 大幅增大Kg(引力)从6->15，防止跑偏
        Q_search=12,  # 略微增大搜索范围
        epsilon=1.5,  #  到达判定距离宽松一点
        Ul=[3, 4],    #  降低速度限制
        dt=0.05,
        vehicle_name='UAV0',
        enable_plot=True,
        debug=True
    )

    print("\n任务完成，降落中...")
    client.landAsync(vehicle_name='UAV0').join()
    client.armDisarm(False, vehicle_name='UAV0')
    client.enableApiControl(False, vehicle_name='UAV0')
