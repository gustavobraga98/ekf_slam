import numpy as np


def euler_from_quaternion(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def quaternion_from_euler(yaw):
    qx = float(0)
    qy = float(0)
    qz = float(np.sin(yaw / 2.0))
    qw = float(np.cos(yaw / 2.0))
    return [qx, qy, qz, qw]
