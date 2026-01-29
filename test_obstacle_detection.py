"""
test_obstacle_detection.py
测试障碍物检测是否正常工作
"""

import airsim
import time
from async_obstacle_detector import AsyncObstacleDetector

# 连接AirSim
client = airsim.MultirotorClient(port=41451)
client.confirmConnection()
print("已连接到AirSim")

# 创建异步障碍物检测器
detector = AsyncObstacleDetector(
    Q_search=15.0,
    vehicle_name='UAV0',
    detection_interval=0.5,
    port=41451
)

print("\n启动障碍物检测器...")
detector.start()

# 等待几秒让检测器运行
for i in range(10):
    time.sleep(1)
    obstacles = detector.get_obstacles()
    print(f"\n[{i+1}秒] 障碍物检测结果:")
    print(f"  类型: {type(obstacles)}")
    print(f"  数量: {len(obstacles) if isinstance(obstacles, list) else 'N/A'}")
    
    if isinstance(obstacles, list):
        if len(obstacles) > 0:
            print(f"  第一个障碍物类型: {type(obstacles[0])}")
            print(f"  第一个障碍物: min=({obstacles[0].min.x_val:.2f}, {obstacles[0].min.y_val:.2f}), "
                  f"max=({obstacles[0].max.x_val:.2f}, {obstacles[0].max.y_val:.2f})")
        else:
            print("  ✅ 返回空列表（正常，可能没有障碍物）")
    else:
        print(f"  ❌ 错误！返回的不是列表: {obstacles}")

print("\n停止检测器...")
detector.stop()
print("测试完成")
