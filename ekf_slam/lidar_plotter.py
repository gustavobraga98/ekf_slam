import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from message_filters import ApproximateTimeSynchronizer, Subscriber

class LidarRansac(Node):

    def __init__(self, degree_range=5, consensus_threshold=40, 
                 max_iterations=1000, distance_threshold=0.01, delete_threshold=0.05,
                 line_similarity_threshold=1, intersection_proximity_threshold=1,
                 map_resolution=0.1):
        super().__init__('minimal_subscriber')
        
        self.scan_sub = Subscriber(self, LaserScan, "/scan")
        self.odom_sub = Subscriber(self, Odometry, "/odom")

        self.ts = ApproximateTimeSynchronizer(
            [self.scan_sub, self.odom_sub],
            30,
            0.01,
        )
        self.ts.registerCallback(self.listener_callback)
        
        self.degree_range = degree_range
        self.consensus_threshold = consensus_threshold
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.delete_threshold = delete_threshold
        self.line_similarity_threshold = line_similarity_threshold
        self.intersection_proximity_threshold = intersection_proximity_threshold
        self.map_resolution = map_resolution  # Resolution for discretization
        self.intersection_points = []
        self.all_points = set()  # Using a set to store unique discretized points
        self.temp_lines = []  # Temporary lines for current iteration
        self.min_x, self.max_x = float('inf'), float('-inf')
        self.min_y, self.max_y = float('inf'), float('-inf')
        self.robot_position = np.array([0.0, 0.0])
        self.robot_orientation = 0.0

        # Setup plot
        self.fig, self.ax = plt.subplots()
        
    def listener_callback(self, scan_msg, odom_msg):
        quaternion = (
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        )
        roll, pitch, yaw = self.euler_from_quaternion(quaternion)
        self.robot_position = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        self.robot_orientation = yaw

        angle_increment = scan_msg.angle_increment
        ranges = scan_msg.ranges

        angles = yaw + (np.arange(len(ranges)) * angle_increment)
        
        points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
        points = points[~np.isnan(points).any(axis=1)]
        points = points[~np.isinf(points).any(axis=1)]
        self.points = np.array([-points[:,1], points[:,0]]).T
        
        # Transform points to global coordinates
        self.robot_position = np.array([-odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.x])
        transformed_points = self.points + self.robot_position

        # Atualiza os pontos acumulados com discretização
        discretized_points = set(tuple(self.discretize_point(point)) for point in transformed_points)
        self.all_points.update(discretized_points)
        
        # Converter pontos discretizados de volta para pontos contínuos para cálculo de limites
        all_points_continuous = np.array([self.continuous_point(p) for p in self.all_points])
        
        # Atualiza limites do mapa
        if all_points_continuous.size > 0:
            self.min_x, self.max_x = min(self.min_x, np.min(all_points_continuous[:, 0])), max(self.max_x, np.max(all_points_continuous[:, 0]))
            self.min_y, self.max_y = min(self.min_y, np.min(all_points_continuous[:, 1])), max(self.max_y, np.max(all_points_continuous[:, 1]))

        # Encontra novas linhas com RANSAC e armazena em temp_lines
        self.temp_lines = self.ransac(transformed_points)
        self.temp_lines = self.group_similar_lines(self.temp_lines)

        # Atualiza interseções baseadas nas novas linhas temporárias
        self.update_intersections(self.find_intersections(self.temp_lines))
        
        # Plotar o mapa completo com pontos e linhas temporárias
        self.plot_lines_and_points()

    def euler_from_quaternion(self, quaternion):
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

    def ransac(self, points):
        lines = []
        epoch = 0
        while len(points) > 0 and len(points) > self.consensus_threshold and epoch < self.max_iterations:
            random_index = randint(0, len(points)-1)
            random_point = points[random_index]
            random_point_angle = np.arctan2(random_point[1], random_point[0])
            lower_angle = random_point_angle - np.radians(self.degree_range)
            upper_angle = random_point_angle + np.radians(self.degree_range)
            angles = np.arctan2(points[:,1], points[:,0])
            mask = (angles >= lower_angle) & (angles <= upper_angle)
            points_in_range = points[mask]

            if len(points_in_range) >= 6:
                sample = points_in_range[np.random.choice(points_in_range.shape[0], 6, replace=False)]
                m, b = self.calculate_best_fit_line(np.array(sample))
                consensus = self.find_points_close_to_line(points, m, b, self.distance_threshold)

                if len(consensus) >= self.consensus_threshold:
                    m, b = self.calculate_best_fit_line(consensus)
                    best_line = [m, b]
                    lines.append(best_line)
                    consensus = self.find_points_close_to_line(points, m, b, self.delete_threshold)
                    mask = ~np.isin(points, consensus).all(axis=1)
                    points = points[mask]

            epoch += 1
        return lines

    def group_similar_lines(self, lines):
        unique_lines = []
        while lines:
            line = lines.pop(0)
            m1, b1 = line
            similar_lines = [line]
            for other_line in lines[:]:
                m2, b2 = other_line
                if abs(m1 - m2) < self.line_similarity_threshold and abs(b1 - b2) < self.line_similarity_threshold:
                    similar_lines.append(other_line)
                    lines.remove(other_line)

            # Average the slope and intercept of similar lines
            if similar_lines:
                avg_m = np.mean([l[0] for l in similar_lines])
                avg_b = np.mean([l[1] for l in similar_lines])
                unique_lines.append([avg_m, avg_b])

        return unique_lines

    def find_intersections(self, lines):
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                m1, b1 = lines[i]
                m2, b2 = lines[j]
                
                # Evita retas paralelas
                if np.isclose(m1, m2):
                    continue
                
                # Calcula a interseção
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1
                
                # Verifica se a interseção está dentro dos limites conhecidos dos pontos
                if self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y:
                    intersections.append((x, y))
                else:
                    self.get_logger().info(f"Intersection out of bounds: ({x}, {y})")

        return intersections

    def update_intersections(self, new_intersections):
        for point in new_intersections:
            too_close = False
            for saved_point in self.intersection_points:
                if np.linalg.norm(np.array(point) - np.array(saved_point)) < self.intersection_proximity_threshold:
                    too_close = True
                    break  # No need to check the rest if we already found a close point
            if not too_close:
                self.intersection_points.append(point)

    def plot_lines_and_points(self):
        all_points_continuous = np.array([self.continuous_point(p) for p in self.all_points])
        
        self.ax.clear()  # Clear the previous plot
        
        # Plot accumulated points
        self.ax.scatter(all_points_continuous[:, 0], all_points_continuous[:, 1], label='Points')
        
        x_values = np.linspace(self.min_x - 0.5, self.max_x + 0.5, 400)
        
        for i, line in enumerate(self.temp_lines):
            m, b = line
            y_values = m * x_values + b
            self.ax.plot(x_values, y_values, label=f'Line {i+1}')
        
        for intersection in self.intersection_points:
            self.ax.scatter(*intersection, color='red', s=100, zorder=5, label='Intersection Point')
        
        self.ax.set_xlim(self.min_x - 0.5, self.max_x + 0.5)
        self.ax.set_ylim(self.min_y - 0.5, self.max_y + 0.5)
        self.ax.legend()
        self.fig.canvas.draw()
        plt.pause(0.1)

    def calculate_best_fit_line(self, points):
        x = points[:,0]
        y = points[:,1]
        coefficients = np.polyfit(x, y, 1)
        return coefficients[0], coefficients[1]

    def calculate_distance_to_line(self, points, m, b):
        distances = np.abs(-m * points[:,0] + points[:,1] - b) / np.sqrt(m**2 + 1)
        return distances

    def find_points_close_to_line(self, points, m, b, distance_threshold):
        distances = self.calculate_distance_to_line(points, m, b)
        mask = distances <= distance_threshold
        return points[mask]

    def discretize_point(self, point):
        """Converte um ponto para a célula de grade correspondente com base na resolução."""
        return (int(point[0] // self.map_resolution), int(point[1] // self.map_resolution))

    def continuous_point(self, discretized_point):
        """Converte um ponto discretizado de volta para coordenadas contínuas."""
        return (discretized_point[0] * self.map_resolution, discretized_point[1] * self.map_resolution)

def main(args=None):
    rclpy.init(args=args)
    lidar_ransac = LidarRansac()
    rclpy.spin(lidar_ransac)
    lidar_ransac.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()