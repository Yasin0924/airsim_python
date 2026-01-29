import numpy as np


def calculate_pose(depth_planar, depth_perspective):
    # get the width and height of img
    height = depth_planar.shape[0]
    width = depth_planar.shape[1]

    # get orientation
    y_orientation = np.arange(0, width).reshape(1, -1)
    y_orientation = np.repeat(y_orientation, height, axis=0)
    y_orientation = y_orientation - (width - 1) / 2

    z_orientation = np.arange(0, height).reshape(-1, 1)
    z_orientation = np.repeat(z_orientation, width, axis=1)
    z_orientation = z_orientation - (height - 1) / 2

    scale_orientation = np.sqrt(np.square(y_orientation) + np.square(z_orientation))
    # 防止除以零
    scale_orientation = np.where(scale_orientation == 0, 1.0, scale_orientation)

    # calculate position
    x_pos = depth_planar.copy()
    
    # 修复：防止负数开方产生NaN
    # 确保 depth_perspective^2 >= depth_planar^2
    depth_diff_squared = np.square(depth_perspective) - np.square(depth_planar)
    depth_diff_squared = np.maximum(depth_diff_squared, 0)  # 负数置为0
    dis_from_center = np.sqrt(depth_diff_squared)
    
    # 处理可能的NaN值
    dis_from_center = np.nan_to_num(dis_from_center, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_pos = dis_from_center * y_orientation / scale_orientation
    z_pos = dis_from_center * z_orientation / scale_orientation
    
    # 最终检查：确保没有NaN或Inf
    y_pos = np.nan_to_num(y_pos, nan=0.0, posinf=0.0, neginf=0.0)
    z_pos = np.nan_to_num(z_pos, nan=0.0, posinf=0.0, neginf=0.0)

    return x_pos, y_pos, z_pos

