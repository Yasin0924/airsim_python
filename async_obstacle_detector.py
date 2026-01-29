"""
async_obstacle_detector.py
异步障碍物检测器 - 在后台线程运行图像处理，避免阻塞主控制循环

修复说明：
为检测线程创建独立的AirSim客户端，避免IOLoop冲突
"""

import threading
import time
import airsim
from ObstacleDetection.obstacles_detect import obstacles_detect


class AsyncObstacleDetector:
    """
    异步障碍物检测器
    在独立线程中运行障碍物检测，主循环可以直接读取最新结果
    注意：检测线程使用独立的AirSim客户端连接，避免IOLoop冲突
    """

    def __init__(self, Q_search, vehicle_name='', detection_interval=0.1, port=41451):
        """
        初始化异步检测器
        :param Q_search: 搜索障碍物距离
        :param vehicle_name: 无人机名称
        :param detection_interval: 检测间隔（秒），控制检测频率
        :param port: AirSim端口号
        """
        self.Q_search = Q_search
        self.vehicle_name = vehicle_name
        self.detection_interval = detection_interval
        self.port = port

        # 线程安全的共享数据
        self._lock = threading.Lock()
        self._obstacles = []  # 最新检测到的障碍物
        self._last_detection_time = 0  # 上次检测时间

        # 线程控制
        self._running = False
        self._thread = None
        self._detector_client = None  # 检测线程专用客户端

    def start(self):
        """启动异步检测线程"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止异步检测线程"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_obstacles(self):
        """
        获取最新检测到的障碍物（线程安全）
        :return: 障碍物列表
        """
        with self._lock:
            return list(self._obstacles)

    def get_last_detection_time(self):
        """获取上次检测的时间戳"""
        with self._lock:
            return self._last_detection_time

    def _detection_loop(self):
        """检测线程的主循环"""
        # 在检测线程中创建独立的AirSim客户端连接
        try:
            self._detector_client = airsim.MultirotorClient(port=self.port)
            self._detector_client.confirmConnection()
            print(f"[AsyncObstacleDetector] Detection thread connected to AirSim (port={self.port})")
        except Exception as e:
            print(f"[AsyncObstacleDetector] Failed to connect: {e}")
            return

        while self._running:
            try:
                # 使用检测线程专用的客户端执行障碍物检测
                obstacles = obstacles_detect(
                    self._detector_client,
                    self.Q_search,
                    vehicle_name=self.vehicle_name
                )

                # 更新共享数据（线程安全）
                with self._lock:
                    self._obstacles = obstacles
                    self._last_detection_time = time.time()

            except Exception as e:
                # 出错时不中断检测循环，打印错误继续
                print(f"[AsyncObstacleDetector] Detection error: {e}")

            # 控制检测频率
            time.sleep(self.detection_interval)

    def __enter__(self):
        """支持 with 语句"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.stop()
        return False

