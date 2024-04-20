import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time



class MapPlotter(Node):

    def __init__(self):
        super().__init__('map_plotter')
        self.scan_sub = Subscriber(self, LaserScan, "/scan")
        self.odom_sub = Subscriber(self, Odometry, "/odom")
        self.ts = ApproximateTimeSynchronizer(
            [self.scan_sub, self.odom_sub],
            30,
            0.01,  # defines the delay (in seconds) with which messages can be synchronized
        )
        self.map = []  # Inicializa o mapa como uma lista vazia
        self.current_x = 0.0
        self.current_y = 0.0
        self.fig, self.ax = plt.subplots()  # Prepara a figura para plotagem
        self.line, = self.ax.plot([], [], 'bo')  # Cria um objeto Line2D vazio
        self.ts.registerCallback(self.callback)

    def callback(self, scan_msg, odom_msg):
        quaternion = (
        odom_msg.pose.pose.orientation.x,
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w
        )
        roll, pitch, yaw = self.euler_from_quaternion(quaternion)
        angle_increment = scan_msg.angle_increment
        ranges = scan_msg.ranges
        angles = yaw + (np.arange(len(ranges)) * angle_increment)
        # Converte os dados polares para coordenadas cartesianas
        points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
        points = points[~np.isnan(points).any(axis=1)]
        points = points[~np.isinf(points).any(axis=1)]
        self.points = np.array([-points[:,1], points[:,0]]).T
        self.robot_position = np.array([-odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.x])
        print(self.robot_position)
        # Transforma os pontos para o referencial do robô
        transformed_points = self.points + self.robot_position
        # Adiciona os pontos transformados ao mapa
        self.map.extend(transformed_points.tolist())
        self.update_map()


    def euler_from_quaternion(self,quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def update_map(self):
        tic = time.time()
        # Atualiza os dados do objeto Line2D
        self.line.set_data(*zip(*self.map))
        # Ajusta os limites dos eixos para incluir todos os pontos
        self.ax.relim()
        self.ax.autoscale_view()
        # Redesenha a figura
        self.fig.canvas.draw()
        plt.pause(0.00001)
        tac = time.time()
        # print(f"Tempo de renderização: {tac - tic}")
        # print(f"Tamanho do mapa: {len(self.map)}")

def main(args=None):
    rclpy.init(args=args)
    map_plotter = MapPlotter()
    rclpy.spin(map_plotter)
    map_plotter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
