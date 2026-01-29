"""
test_depth_camera.py
测试深度相机数据，查看为什么只能检测到近距离障碍物
"""

import airsim
import numpy as np
import time

# 连接AirSim
client = airsim.MultirotorClient(port=41451)
client.confirmConnection()
print("已连接到AirSim\n")

# 获取深度图像
print("获取深度图像...")
image_requests = [
    airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
    airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
]

responses = client.simGetImages(image_requests, vehicle_name='UAV0')

# 分析深度图
print("\n深度图分析:")
print("=" * 60)

for idx, response in enumerate(responses):
    img_type = "DepthPerspective" if idx == 0 else "DepthPlanar"
    depth_img = airsim.get_pfm_array(response)
    
    print(f"\n{img_type}:")
    print(f"  图像尺寸: {depth_img.shape}")
    print(f"  数据类型: {depth_img.dtype}")
    
    # 过滤掉无效值
    valid_depths = depth_img[np.isfinite(depth_img) & (depth_img > 0)]
    
    if len(valid_depths) > 0:
        print(f"  有效深度值数量: {len(valid_depths)}/{depth_img.size}")
        print(f"  最小深度: {np.min(valid_depths):.2f}m")
        print(f"  最大深度: {np.max(valid_depths):.2f}m")
        print(f"  平均深度: {np.mean(valid_depths):.2f}m")
        print(f"  中位数深度: {np.median(valid_depths):.2f}m")
        
        # 统计不同距离范围的像素数
        ranges = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 100)]
        print(f"\n  深度分布:")
        for r_min, r_max in ranges:
            count = np.sum((valid_depths >= r_min) & (valid_depths < r_max))
            percentage = count / len(valid_depths) * 100
            print(f"    {r_min:3d}-{r_max:3d}m: {count:6d} 像素 ({percentage:5.1f}%)")
    else:
        print(f"  ⚠️ 没有有效的深度值！")
        print(f"  NaN数量: {np.sum(np.isnan(depth_img))}")
        print(f"  Inf数量: {np.sum(np.isinf(depth_img))}")
        print(f"  零值数量: {np.sum(depth_img == 0)}")

print("\n" + "=" * 60)
print("\n💡 提示:")
print("  - 如果最大深度很小（<5m），说明深度相机配置有问题")
print("  - 如果大部分像素都是NaN/Inf，说明场景中没有障碍物或相机设置错误")
print("  - 正常情况下应该能看到10-100米范围的深度值")
