import numpy as np

def euler_from_quaternion(quaternion):
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return yaw  # Retorna apenas o ângulo yaw

def quaternion_from_euler(yaw):
    # Cálculo do quaternion a partir do ângulo de Euler (yaw)
    qx = 0
    qy = 0
    qz = np.sin(yaw / 2.0)
    qw = np.cos(yaw / 2.0)

    return [qx, qy, qz, qw]