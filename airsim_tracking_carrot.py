"""
tracking_carrot.py
airsim 四旋翼轨迹跟踪 Carrot控制算法
"""

import airsim
import math
import time
import numpy as np
from UavAgent import get_state, move_by_acceleration
from mymath import distance, myatan, distance_3d


def move_by_path(client, Va, Path, Pz, delta=1, K=0.5, dt=0.05, epsilon=0.2, plot_interval=5):
    """
    :param client: AirSim连接客户端
    :param Va: 速率上限
    :param Path: 被跟踪航路点
    :param Pz: 飞行高度
    :param delta: 向前搜索下一个航路点的距离
    :param K: 控制器参数
    :param dt: 迭代时间（已优化为0.05s）
    :param epsilon: 误差上限
    :param plot_interval: 绘图间隔（每隔多少次循环绘制一次）
    :return: 无
    """

    state = client.simGetGroundTruthKinematics()
    psi = airsim.to_eularian_angles(state.orientation)[2]
    Px = state.position.x_val
    Py = state.position.y_val
    Wb = [Px, Py]
    
    plot_counter = 0  # 绘图计数器

    for i in range(len(Path)):
        Wa = Wb
        Wb = [Path[i].x_val, Path[i].y_val]
        theta = myatan(Wa, Wb)
        xt = Wa[0]
        yt = Wa[1]
        while (((xt - Wb[0]) * (Wb[0] - Wa[0]) < 0 and (yt - Wb[1]) * (Wb[1] - Wa[1]) < 0) and
               ((i != (len(Path) - 1)) or
                distance([xt, yt], Wb)) > epsilon):
            loop_start = time.time()  # 记录循环开始时间
            
            theta_u = myatan(Wa, [Px, Py])
            beta = 0 if theta_u is None else theta - theta_u
            Ru = distance(Wa, [Px, Py])
            R = Ru * math.cos(beta)
            e = Ru * math.sin(beta)
            xt = Wa[0] + (R + delta) * math.cos(theta)
            yt = Wa[1] + (R + delta) * math.sin(theta)
            psi_d = myatan([Px, Py], [xt, yt])
            u = K * (psi_d - psi) * Va
            u = (u / abs(u)) if abs(u) > 1 else u
            psi = psi_d
            Vy = Va * math.sin(psi) + u * dt
            if abs(Vy) >= Va:
                Vy = Va
            Vx = np.sign(math.cos(psi)) * math.sqrt(Va ** 2 - Vy ** 2)
            
            # 【优化】移除.join()，使用非阻塞调用
            client.moveByVelocityZAsync(Vx, Vy, Pz, dt)
            
            # 获取状态（仅一次）
            state = client.simGetGroundTruthKinematics()
            Px_new = state.position.x_val
            Py_new = state.position.y_val
            
            # 【优化】降低绘图频率，减少API调用开销
            plot_counter += 1
            if plot_counter % plot_interval == 0:
                plot_p1 = [airsim.Vector3r(Px, Py, Pz)]
                plot_p2 = [airsim.Vector3r(Px_new, Py_new, Pz)]
                client.simPlotArrows(plot_p1, plot_p2, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
                client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)
            
            Px = Px_new
            Py = Py_new
            
            # 【优化】固定控制频率，确保循环时间稳定
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


def move_by_path_3d(client, Path, K0=1.5, K1=4, K2=0.6, dt=0.12, a0=1, delta=0.7, plot_interval=3):
    """
    :param client: AirSim连接客户端
    :param Path: 被跟踪航路点
    :param K0: 控制器参数
    :param K1: 控制器参数
    :param K2: 控制器参数
    :param dt: 迭代时间（已优化为0.12s）
    :param a0: 控制器输出上限
    :param delta: 向前搜索下一个航路点的距离
    :param plot_interval: 绘图间隔（每隔多少次循环绘制一次）
    :return: 无
    """

    state = get_state(client)
    P = state['position']
    V = state['linear_velocity']
    Wb = P
    Wb_m = np.matrix(Wb).T
    P_m = np.matrix(P).T
    V_m = np.matrix(V).T
    I3 = np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    plot_counter = 0  # 绘图计数器
    
    for i in range(len(Path)):
        Wa = Wb
        Wb = [Path[i].x_val, Path[i].y_val, Path[i].z_val]
        Wa_m = Wb_m
        Wb_m = np.matrix(Wb).T
        A = I3 - (Wa_m - Wb_m).dot((Wa_m - Wb_m).T) / (distance_3d(Wa_m, Wb_m) ** 2)
        Pt = P_m - Wb_m
        d = np.linalg.norm(Pt - A.dot(Pt))
        while d >= delta or \
                (i == len(Path) - 1
                 and ((P[0] - Wb[0]) * (Wb[0] - Wa[0]) < 0
                      or (P[1] - Wb[1]) * (Wb[1] - Wa[1]) < 0
                      or (P[2] - Wb[2]) * (Wb[2] - Wa[2]) < 0)):
            loop_start = time.time()  # 记录循环开始时间
            
            Pt = P_m - Wb_m
            U1 = K0 * Pt + K1 * A.dot(Pt)
            if np.linalg.norm(U1, ord=np.inf) > a0:
                U1 = U1 * a0 / np.linalg.norm(U1, ord=np.inf)
            U = -(U1 + V_m) / K2
            U_cmd = np.array(U)[:, 0]
            
            # 【优化】使用非阻塞调用
            move_by_acceleration(client, U_cmd[0], U_cmd[1], U_cmd[2], dt * 10, waited=False)
            
            d = np.linalg.norm(Pt - A.dot(Pt))
            
            # 获取状态（仅一次）
            P_old = P.copy() if isinstance(P, list) else list(P)
            state = get_state(client)
            P = state['position']
            V = state['linear_velocity']
            P_m = np.matrix(P).T
            V_m = np.matrix(V).T
            
            # 【优化】降低绘图频率
            plot_counter += 1
            if plot_counter % plot_interval == 0:
                plot_p1 = [airsim.Vector3r(P_old[0], P_old[1], P_old[2])]
                plot_p2 = [airsim.Vector3r(P[0], P[1], P[2])]
                client.simPlotArrows(plot_p1, plot_p2, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
                client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)
            
            # 【优化】固定控制频率
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    client = airsim.MultirotorClient()  # 创建连接
    client.confirmConnection()  # 检查连接
    client.reset()
    client.enableApiControl(True)  # 获取控制权
    client.armDisarm(True)  # 电机启转
    client.takeoffAsync().join()  # 起飞
    client.moveToZAsync(-3, 1).join()  # 上升到3米高度
    client.simSetTraceLine([1, 0, 0, 1], thickness=5)
    client.simFlushPersistentMarkers()  # 清空画图

    # 二维航路点跟踪
    points = [airsim.Vector3r(5, 0, -3),
              airsim.Vector3r(5, 8, -3),
              airsim.Vector3r(8, 12, -3),
              airsim.Vector3r(4, 9, -3)]
    client.simPlotPoints(points, color_rgba=[0, 1, 0, 1], size=30, is_persistent=True)
    move_by_path(client, 3, points, -3, delta=0.5, epsilon=0.05)

    # 三维航路点跟踪（也可执行二维）
    # points = [airsim.Vector3r(5, 0, -3),
    #           airsim.Vector3r(3, 10, -1),
    #           airsim.Vector3r(8, 12, -7),
    #           airsim.Vector3r(-5, 9, -2)]
    # client.simPlotPoints(points, color_rgba=[0, 1, 0, 1], size=30, is_persistent=True)
    # client.simPlotLineStrip(points, color_rgba=[0, 1, 0, 1], thickness=5, is_persistent=True)
    # move_by_path_3d(client, points, delta=0.5, a0=2, dt=0.12, K0=1, K1=2, K2=0.8)

    # 仿真结束
    client.landAsync().join()  # 降落
    client.armDisarm(False)  # 电机上锁
    client.enableApiControl(False)  # 释放控制权
