import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import time


class DataSaver(Node):
    def __init__(self):
        super().__init__('data_saver')

        # Subscrições para os tópicos de odometria
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/noisy_odom', self.noisy_odom_callback, 10)
        self.create_subscription(Odometry, '/ekf_odom', self.ekf_odom_callback, 10)

        # Inicializa o arquivo CSV
        self.csv_file = open('odometry_data.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'source', 'x', 'y'])

        # Tempo inicial para normalizar o tempo
        self.start_time = time.time()

    def save_data(self, source, msg):
        current_time = time.time() - self.start_time
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.csv_writer.writerow([current_time, source, x, y])
        self.get_logger().info(f"Saved data from {source}: time={current_time:.2f}, x={x:.2f}, y={y:.2f}")

    def odom_callback(self, msg):
        self.save_data('/odom', msg)

    def noisy_odom_callback(self, msg):
        self.save_data('/noisy_odom', msg)

    def ekf_odom_callback(self, msg):
        self.save_data('/ekf_odom', msg)

    def destroy_node(self):
        super().destroy_node()
        self.csv_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = DataSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
