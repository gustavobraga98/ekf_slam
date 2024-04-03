import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        with open('lidar_data.txt', 'a') as f:
            f.write(str(msg.header) + '\n')
            f.write(str(msg.angle_min) + '\n')
            f.write(str(msg.angle_max) + '\n')
            f.write(str(msg.angle_increment) + '\n')
            f.write(str(msg.time_increment) + '\n')
            f.write(str(msg.scan_time) + '\n')
            f.write(str(msg.range_min) + '\n')
            f.write(str(msg.range_max) + '\n')
            f.write(str(msg.ranges) + '\n')
            f.write(str(msg.intensities) + '\n')



def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()