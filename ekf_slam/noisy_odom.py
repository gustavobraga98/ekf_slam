import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import random
from utils import (quaternion_from_euler,
                    euler_from_quaternion)


class NoisyOdomPublisher(Node):
    def __init__(self):
        super().__init__('noisy_odom_publisher')
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.noisy_odom_publisher = self.create_publisher(
            Odometry, '/noisy_odom', 10)
        
        self.drift_x = 0.0
        self.drift_y = 0.0

    def odom_callback(self, msg):
        noisy_odom = self.add_noise_to_odom(msg)
        self.noisy_odom_publisher.publish(noisy_odom)

    def add_noise_to_odom(self, odom_msg):
        noisy_odom = Odometry()
        noisy_odom.header = odom_msg.header
        noisy_odom.child_frame_id = odom_msg.child_frame_id

        # Posição com ruído e drift
        noise_std_dev = 0.05
        self.drift_x += random.uniform(-0.01, 0.01)  # Incremento de drift X
        self.drift_y += random.uniform(-0.01, 0.01)  # Incremento de drift Y
        noisy_odom.pose.pose.position.x = odom_msg.pose.pose.position.x + random.gauss(0, noise_std_dev) + self.drift_x
        noisy_odom.pose.pose.position.y = odom_msg.pose.pose.position.y + random.gauss(0, noise_std_dev) + self.drift_y

        # Orientação com ruído
        original_orientation = odom_msg.pose.pose.orientation

        # Passa o objeto `original_orientation` diretamente
        original_yaw = euler_from_quaternion(original_orientation)

        noise_yaw = random.gauss(0, 0.01)  # Ruído no yaw
        noisy_yaw = original_yaw + noise_yaw
        noisy_quat = quaternion_from_euler(noisy_yaw)

        noisy_odom.pose.pose.orientation = Quaternion(
            x=float(noisy_quat[0]), y=float(noisy_quat[1]),
            z=float(noisy_quat[2]), w=float(noisy_quat[3]))

        # Copy linear and angular velocities
        noisy_odom.twist.twist = odom_msg.twist.twist
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
