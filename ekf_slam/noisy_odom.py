from ekf_slam.utils import euler_from_quaternion, quaternion_from_euler
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import random
import numpy as np


class OdomNoiseNode(Node):
    def __init__(self):
        super().__init__('odom_noise_node')

        # Publisher para odometria com ruído
        self.noisy_odom_pub = self.create_publisher(Odometry, '/noisy_odom', 10)

        # Subscriber para odometria original
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Inicializa o drift (erro sistemático) nas variáveis de posição e orientação
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.drift_theta = 0.0

        # Parâmetros para ruído e drift
        self.noise_std_dev = [0.1, 0.1, 0.05]  # Desvio padrão do ruído: x, y, e theta
        self.drift_rate = [0.0001, 0.0001, 0.00005]  # Taxa de drift: x, y, e theta

    def odom_callback(self, msg):
        # Aplica ruído e drift na odometria
        noisy_odom = self.add_noise_to_odom(msg)
        noisy_odom = self.apply_drift(noisy_odom)

        # Publica odometria com ruído
        self.noisy_odom_pub.publish(noisy_odom)

    def add_noise_to_odom(self, odom_msg):
        # Cria uma nova mensagem de odometria com ruído
        noisy_odom = Odometry()
        noisy_odom.header = odom_msg.header

        # Adiciona ruído gaussiano nas posições
        noisy_odom.pose.pose.position.x = odom_msg.pose.pose.position.x + random.gauss(0, self.noise_std_dev[0])
        noisy_odom.pose.pose.position.y = odom_msg.pose.pose.position.y + random.gauss(0, self.noise_std_dev[1])

        # Adiciona ruído na orientação (yaw)
        orientation = odom_msg.pose.pose.orientation
        euler = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        noisy_euler = [euler[0], euler[1], euler[2] + random.gauss(0, self.noise_std_dev[2])]
        noisy_orientation = quaternion_from_euler(*noisy_euler)

        noisy_odom.pose.pose.orientation.x = noisy_orientation[0]
        noisy_odom.pose.pose.orientation.y = noisy_orientation[1]
        noisy_odom.pose.pose.orientation.z = noisy_orientation[2]
        noisy_odom.pose.pose.orientation.w = noisy_orientation[3]

        # Copia as velocidades
        noisy_odom.twist = odom_msg.twist

        return noisy_odom

    def apply_drift(self, odom_msg):
        # Incrementa o drift nas posições e na orientação
        self.drift_x += self.drift_rate[0]
        self.drift_y += self.drift_rate[1]
        self.drift_theta += self.drift_rate[2]

        # Aplica o drift nas posições
        odom_msg.pose.pose.position.x += self.drift_x
        odom_msg.pose.pose.position.y += self.drift_y

        # Aplica o drift na orientação
        orientation = odom_msg.pose.pose.orientation
        euler = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        euler[2] += self.drift_theta
        euler[2] = self.normalize_angle(euler[2])  # Normaliza o ângulo
        drifted_orientation = quaternion_from_euler(*euler)

        odom_msg.pose.pose.orientation.x = drifted_orientation[0]
        odom_msg.pose.pose.orientation.y = drifted_orientation[1]
        odom_msg.pose.pose.orientation.z = drifted_orientation[2]
        odom_msg.pose.pose.orientation.w = drifted_orientation[3]

        return odom_msg

    @staticmethod
    def normalize_angle(angle):
        """Normaliza o ângulo para o intervalo [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = OdomNoiseNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
