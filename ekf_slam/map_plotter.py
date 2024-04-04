import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry

class MapPlotter(Node):

    def __init__(self):
        super().__init__('map_plotter')
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odometry_callback,
            10)
        self.map = []  # Inicializa o mapa como uma lista vazia
        self.current_x = 0.0
        self.current_y = 0.0
        self.fig, self.ax = plt.subplots()  # Prepara a figura para plotagem

    def scan_callback(self, msg):
        angle_increment = msg.angle_increment
        ranges = msg.ranges
        angles = np.arange(len(ranges)) * angle_increment
        # Converte os dados polares para coordenadas cartesianas
        points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
        points = points[~np.isnan(points).any(axis=1)]
        points = points[~np.isinf(points).any(axis=1)]
        self.points = np.array([-points[:,1], points[:,0]]).T
        

    def odometry_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.robot_position = np.array([self.current_y, self.current_x])
        # Transforma os pontos para o referencial do robô
        transformed_points = self.points + self.robot_position
        # Adiciona os pontos transformados ao mapa
        self.map.extend(transformed_points.tolist())
        self.update_map()

    def update_map(self):
        # Limpa a figura anterior
        self.ax.clear()
        # Plota os pontos acumulados
        for point in self.map:
            self.ax.plot(point[0], point[1], 'bo')
        # Redesenha a figura
        self.fig.canvas.draw()
        plt.pause(0.00001)

def main(args=None):
    rclpy.init(args=args)
    map_plotter = MapPlotter()
    rclpy.spin(map_plotter)
    map_plotter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
