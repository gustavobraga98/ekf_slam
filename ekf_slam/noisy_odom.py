import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math
import random


def euler_from_quaternion(quaternion):
    """
    Converte um quaternion em ângulos de Euler (roll, pitch, yaw).
    """
    x, y, z, w = quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))  # Clampe para evitar valores fora do intervalo
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    """
    Converte ângulos de Euler (roll, pitch, yaw) em um quaternion.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w


class NoisyOdomPublisher(Node):
    def __init__(self):
        super().__init__('noisy_odom_publisher')
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.noisy_odom_publisher = self.create_publisher(
            Odometry, '/noisy_odom', 10)

        # Parâmetros de ruído (\(\alpha_1, \alpha_2, \alpha_3, \alpha_4\))
        self.alpha1 = 0.001  # Ruído angular inicial
        self.alpha2 = 0.005  # Ruído angular proporcional à translação
        self.alpha3 = 0.2    # Ruído de translação
        self.alpha4 = 0.001  # Ruído de translação proporcional às rotações


        # Última odometria recebida
        self.last_odom = None
        self.pose = [0.0, 0.0, 0.0]  # x, y, yaw

    def odom_callback(self, msg):
        noisy_odom = self.add_noise_to_odom(msg)
        self.noisy_odom_publisher.publish(noisy_odom)

    def add_noise_to_odom(self, odom_msg):
        if self.last_odom is None:
            # Inicializa pose no primeiro callback
            self.last_odom = odom_msg
            self.pose[0] = odom_msg.pose.pose.position.x
            self.pose[1] = odom_msg.pose.pose.position.y
            orientation = odom_msg.pose.pose.orientation
            _, _, self.pose[2] = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w])
            return odom_msg

        # Calcula translação e rotação incremental
        dx = odom_msg.pose.pose.position.x - self.last_odom.pose.pose.position.x
        dy = odom_msg.pose.pose.position.y - self.last_odom.pose.pose.position.y
        trans = np.sqrt(dx ** 2 + dy ** 2)

        last_orientation = self.last_odom.pose.pose.orientation
        _, _, theta1 = euler_from_quaternion([
            last_orientation.x, last_orientation.y,
            last_orientation.z, last_orientation.w])

        curr_orientation = odom_msg.pose.pose.orientation
        _, _, theta2 = euler_from_quaternion([
            curr_orientation.x, curr_orientation.y,
            curr_orientation.z, curr_orientation.w])

        rot1 = np.arctan2(dy, dx) - theta1
        rot2 = theta2 - theta1 - rot1

        # Adiciona ruído baseado nos parâmetros \(\alpha\)
        sd_rot1 = self.alpha1 * abs(rot1) + self.alpha2 * trans
        sd_trans = self.alpha3 * trans + self.alpha4 * (abs(rot1) + abs(rot2))
        sd_rot2 = self.alpha1 * abs(rot2) + self.alpha2 * trans

        trans_noisy = trans + np.random.normal(0, sd_trans)
        rot1_noisy = rot1 + np.random.normal(0, sd_rot1)
        rot2_noisy = rot2 + np.random.normal(0, sd_rot2)

        # Atualiza pose estimada
        self.pose[0] += trans_noisy * np.cos(self.pose[2] + rot1_noisy)  # Correção para X
        self.pose[1] += trans_noisy * np.sin(self.pose[2] + rot1_noisy)  # Correção para Y
        self.pose[2] += rot1_noisy + rot2_noisy

        # Cria mensagem de odometria ruidosa
        noisy_odom = Odometry()
        noisy_odom.header = odom_msg.header
        noisy_odom.child_frame_id = odom_msg.child_frame_id

        noisy_odom.pose.pose.position.x = self.pose[0]
        noisy_odom.pose.pose.position.y = self.pose[1]
        noisy_quat = quaternion_from_euler(0, 0, self.pose[2])
        noisy_odom.pose.pose.orientation = Quaternion(
            x=noisy_quat[0], y=noisy_quat[1], z=noisy_quat[2], w=noisy_quat[3])

        # Copia as velocidades (sem ruído)
        noisy_odom.twist.twist = odom_msg.twist.twist

        self.last_odom = odom_msg
        return noisy_odom



def main(args=None):
    rclpy.init(args=args)
    node = NoisyOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
