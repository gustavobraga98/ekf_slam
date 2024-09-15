import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

class OdomNoiseNode(Node):
    def __init__(self):
        super().__init__('odom_noise_node')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',  # Tópico original da odometria
            self.odom_callback,
            10
        )
        self.publisher = self.create_publisher(
            Odometry,
            '/odom_noisy',  # Novo tópico de odometria com ruído
            10
        )

        # Parâmetros de ruído inicial e drift
        self.position_noise_stddev = 0.05  # Aumentar o ruído padrão da posição
        self.orientation_noise_stddev = np.radians(2)  # Aumentar o ruído padrão da orientação

        # Drift acumulativo
        self.drift_rate_position = 0.01    # Aumentar a taxa de drift por atualização
        self.drift_rate_orientation = np.radians(0.5)  # Aumentar a taxa de drift por atualização

        # Drift acumulativo inicial
        self.cumulative_drift_position = np.array([0.0, 0.0, 0.0])
        self.cumulative_drift_orientation = np.array([0.0, 0.0, 0.0])

    def odom_callback(self, msg):
        noisy_msg = Odometry()
        noisy_msg.header = msg.header
        noisy_msg.child_frame_id = msg.child_frame_id

        # Adicione ruído à posição com drift acumulativo
        self.cumulative_drift_position += np.random.normal(0, self.drift_rate_position, 3)
        noisy_msg.pose.pose.position.x = msg.pose.pose.position.x + np.random.normal(0, self.position_noise_stddev) + self.cumulative_drift_position[0]
        noisy_msg.pose.pose.position.y = msg.pose.pose.position.y + np.random.normal(0, self.position_noise_stddev) + self.cumulative_drift_position[1]
        noisy_msg.pose.pose.position.z = msg.pose.pose.position.z + np.random.normal(0, self.position_noise_stddev) + self.cumulative_drift_position[2]
        
        # Adicione ruído à orientação com drift acumulativo (conversão quaternion -> euler -> quaternion)
        q = msg.pose.pose.orientation
        euler = self.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.cumulative_drift_orientation += np.random.normal(0, self.drift_rate_orientation, 3)
        noisy_euler = [
            euler[0] + np.random.normal(0, self.orientation_noise_stddev) + self.cumulative_drift_orientation[0],
            euler[1] + np.random.normal(0, self.orientation_noise_stddev) + self.cumulative_drift_orientation[1],
            euler[2] + np.random.normal(0, self.orientation_noise_stddev) + self.cumulative_drift_orientation[2]
        ]
        noisy_quaternion = self.quaternion_from_euler(*noisy_euler)
        noisy_msg.pose.pose.orientation.x = noisy_quaternion[0]
        noisy_msg.pose.pose.orientation.y = noisy_quaternion[1]
        noisy_msg.pose.pose.orientation.z = noisy_quaternion[2]
        noisy_msg.pose.pose.orientation.w = noisy_quaternion[3]

        # Copiar informações de velocidade
        noisy_msg.twist.twist = msg.twist.twist

        # Publicar mensagem com ruído
        self.publisher.publish(noisy_msg)

    def euler_from_quaternion(self, quaternion):
        x, y, z, w = quaternion
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def quaternion_from_euler(self, roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    odom_noise_node = OdomNoiseNode()
    rclpy.spin(odom_noise_node)
    odom_noise_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
