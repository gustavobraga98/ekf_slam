from ekf_slam.utils import euler_from_quaternion, quaternion_from_euler
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node


class EKFNode(Node):
    def __init__(self):
        super().__init__('EKF_node')

        # Subscrição para odometria ruidosa
        self.odom_sub = self.create_subscription(Odometry, "/noisy_odom", self.odom_callback, 10)
        # Publicação para posição corrigida pelo EKF
        self.ekf_pub = self.create_publisher(Pose, "/ekf_pos", 10)

        # Estado inicial do filtro (posição e orientação)
        self.prev_position = [0.0, 0.0, 0.0]
        self.position = [0.0, 0.0, 0.0]

        # Covariância inicial
        self.cov_matrix = np.diag([0.1, 0.1, 0.1])
        self.error_matrix = np.diag([0.01, 0.01, 0.001])
        
        # Velocidades e tempo inicial
        self.velocity = [0.0, 0.0]
        self.prev_time = self.get_clock().now()

    def normalize_angle(self, angle):
        """Normaliza um ângulo para o intervalo [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def odom_callback(self, msg):
        # Obtém a posição e orientação a partir da mensagem
        self.position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            euler_from_quaternion(msg.pose.pose.orientation)
        ]
        self.position[2] = self.normalize_angle(self.position[2])

        # Calcula o delta de tempo
        current_time = self.get_clock().now()
        delta_time = (current_time - self.prev_time).nanoseconds / 1e9

        # Calcula deslocamentos e velocidades
        delta_x = self.position[0] - self.prev_position[0]
        delta_y = self.position[1] - self.prev_position[1]
        delta_theta = self.position[2] - self.prev_position[2]
        delta_theta = self.normalize_angle(delta_theta)

        if delta_time > 0.05:
            self.velocity = [
                np.sqrt(delta_x**2 + delta_y**2) / delta_time,
                delta_theta / delta_time
            ]

        # Atualiza posição e tempo anterior
        self.prev_position = self.position
        self.prev_time = current_time

        # Etapas do EKF
        self.prediction(delta_time)
        self.update_state()
        self.publish_ekf_position()

    def prediction(self, delta_time):
        v = self.velocity[0]
        w = self.velocity[1]
        theta = self.position[2]

        # Predição de movimento
        delta_x = v * np.cos(theta) * delta_time
        delta_y = v * np.sin(theta) * delta_time
        delta_theta = w * delta_time

        self.predicted_position = np.array([
            self.prev_position[0] + delta_x,
            self.prev_position[1] + delta_y,
            self.normalize_angle(self.prev_position[2] + delta_theta)
        ])

        # Atualiza a covariância
        F = np.array([
            [1, 0, -v * np.sin(theta) * delta_time],
            [0, 1, v * np.cos(theta) * delta_time],
            [0, 0, 1]
        ])
        self.cov_matrix = F @ self.cov_matrix @ F.T + self.error_matrix

    def update_state(self):
        observed_position = np.array(self.position)
        innovation = observed_position - self.predicted_position
        innovation[2] = self.normalize_angle(innovation[2])

        H = np.eye(3)
        R = np.diag([0.05, 0.05, 0.001])
        S = H @ self.cov_matrix @ H.T + R
        K = self.cov_matrix @ H.T @ np.linalg.inv(S)

        # Atualiza estado e covariância
        self.position = self.predicted_position + K @ innovation
        self.position[2] = self.normalize_angle(self.position[2])
        self.cov_matrix = (np.eye(3) - K @ H) @ self.cov_matrix

    def publish_ekf_position(self):
        pose_msg = Pose()
        pose_msg.position.x = self.position[0]
        pose_msg.position.y = self.position[1]
        quat = quaternion_from_euler(0, 0, self.position[2])
        pose_msg.orientation.x = quat[0]
        pose_msg.orientation.y = quat[1]
        pose_msg.orientation.z = quat[2]
        pose_msg.orientation.w = quat[3]
        self.ekf_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
