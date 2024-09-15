import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from message_filters import ApproximateTimeSynchronizer, Subscriber

class LidarRansac(Node):
    def __init__(self, degree_range=5, consensus_threshold=40, 
                 max_iterations=1000, distance_threshold=0.01, delete_threshold=0.05,
                 line_similarity_threshold=1, intersection_proximity_threshold=1,
                 map_resolution=0.1):
        
        super().__init__('minimal_subscriber')
        
        # Subscribers
        self.scan_sub = Subscriber(self, LaserScan, "/scan")
        self.odom_sub = Subscriber(self, Odometry, "/odom_noisy")  # Alterado para consumir odometria com ruído
        self.gt_odom_sub = self.create_subscription(Odometry, "/odom", self.gt_odom_callback, 10)  # Ground truth odometry
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        self.ts = ApproximateTimeSynchronizer([self.scan_sub, self.odom_sub], 30, 0.01)
        self.ts.registerCallback(self.listener_callback)
        
        self.degree_range = degree_range
        self.consensus_threshold = consensus_threshold
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.delete_threshold = delete_threshold
        self.line_similarity_threshold = line_similarity_threshold
        self.intersection_proximity_threshold = intersection_proximity_threshold
        self.map_resolution = map_resolution

        self.intersection_points = []
        self.all_points = set()
        self.temp_lines = []
        self.min_x, self.max_x = float('inf'), float('-inf')
        self.min_y, self.max_y = float('inf'), float('-inf')
        self.robot_position = np.array([0.0, 0.0])
        self.robot_orientation = 0.0

        self.current_control = np.array([0.0, 0.0])
        self.state_initialized = False
        self.state = np.array([0.0, 0.0, 0.0])
        self.P = np.eye(3)
        self.last_odom_time = None

        self.gt_trajectory = []  # Guarda a trajetória ground truth
        self.noisy_trajectory = []  # Guarda a trajetória ruidosa
        self.ekf_trajectory = []  # Guarda a trajetória EKF

        # Setup plot
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def listener_callback(self, scan_msg, odom_msg):
        quaternion = (odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
                      odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w)
        roll, pitch, yaw = self.euler_from_quaternion(quaternion)
        self.robot_position = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        self.robot_orientation = yaw

        if not self.state_initialized:
            self.state = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw])
            self.state_initialized = True

        current_odom_time = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
        
        if self.last_odom_time is not None:
            dt = current_odom_time - self.last_odom_time
            if dt > 0:
                self.state = self.predict_state(self.state, self.current_control, dt)
                A = self.calculate_jacobian_A(self.state, self.current_control, dt)
                Q = np.diag([0.1, 0.1, np.radians(1)])
                self.P = self.predict_covariance(self.P, A, Q)

                z = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw])
                H = np.eye(3)
                R = np.diag([0.1, 0.1, np.radians(1)])
                h_x = self.state
                y = z - h_x
                K = self.compute_kalman_gain(self.P, H, R)
                self.state = self.state + K @ y
                I = np.eye(len(self.P))
                self.P = (I - K @ H) @ self.P

                self.noisy_trajectory.append(self.robot_position)
                self.ekf_trajectory.append(self.state[:2].copy())

                self.get_logger().info(f"Predicted State: [x: {self.state[0]:.6f}, y: {self.state[1]:.6f}, theta: {self.state[2]:.6f}]")
                self.get_logger().info(f"Odometry State: [x: {self.robot_position[0]:.6f}, y: {self.robot_position[1]:.6f}, theta: {self.robot_orientation:.6f}]")
        
        self.last_odom_time = current_odom_time

        # Processamento atual do LIDAR e Odometria
        angle_increment = scan_msg.angle_increment
        ranges = scan_msg.ranges
        angles = yaw + (np.arange(len(ranges)) * angle_increment)
        points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
        points = points[~np.isnan(points).any(axis=1)]
        self.points = points[~np.isinf(points).any(axis=1)]
        transformed_points = self.points + self.robot_position
        discretized_points = set(tuple(self.discretize_point(point)) for point in transformed_points)
        self.all_points.update(discretized_points)

        all_points_continuous = np.array([self.continuous_point(p) for p in self.all_points])
        if all_points_continuous.size > 0:
            self.min_x = min(self.min_x, np.min(all_points_continuous[:, 0]))
            self.max_x = max(self.max_x, np.max(all_points_continuous[:, 0]))
            self.min_y = min(self.min_y, np.min(all_points_continuous[:, 1]))
            self.max_y = max(self.max_y, np.max(all_points_continuous[:, 1]))

        self.temp_lines = self.ransac(transformed_points)
        self.temp_lines = self.group_similar_lines(self.temp_lines)
        self.update_intersections(self.find_intersections(self.temp_lines))

        self.plot_lines_and_points()

    def gt_odom_callback(self, msg):
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.gt_trajectory.append(pos)

    def compute_kalman_gain(self, P, H, R):
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        return K

    def predict_covariance(self, P, A, Q):
        return A @ P @ A.T + Q

    def calculate_jacobian_A(self, state, control, dt):
        x, y, theta = state
        v, omega = control
        A = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1, v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
        return A
    
    def cmd_vel_callback(self, msg):
        self.current_control = np.array([msg.linear.x, msg.angular.z])

    def predict_state(self, state, control, dt):
        x, y, theta = state
        v, omega = control
        new_x = x + v * np.cos(theta) * dt
        new_y = y + v * np.sin(theta) * dt
        new_theta = theta + omega * dt

        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        return np.array([new_x, new_y, new_theta])

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

                if np.isclose(m1, m2):
                    continue

                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1

                if self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y:
                    intersections.append((x, y))
                else:
                    pass

        return intersections

    def update_intersections(self, new_intersections):
        for point in new_intersections:
            too_close = False
            for saved_point in self.intersection_points:
                if np.linalg.norm(np.array(point) - np.array(saved_point)) < self.intersection_proximity_threshold:
                    too_close = True
                    break
            if not too_close:
                self.intersection_points.append(point)

    def plot_lines_and_points(self):
        all_points_continuous = np.array([self.continuous_point(p) for p in self.all_points])

        self.ax.clear()
        self.ax.scatter(all_points_continuous[:, 0], all_points_continuous[:, 1], label='Points', s=2)

        x_values = np.linspace(self.min_x - 0.5, self.max_x + 0.5, 400)

        for i, line in enumerate(self.temp_lines):
            m, b = line
            y_values = m * x_values + b
            self.ax.plot(x_values, y_values, label=f'Line {i+1}')
        
        for intersection in self.intersection_points:
            self.ax.scatter(*intersection, color='red', s=100, zorder=5, label='Intersection Point')

        self.ax.scatter(self.state[0], self.state[1], color='blue', marker=(3, 0, self.state[2] * 180 / np.pi - 90), s=100, label='Robot')

        self.ax.set_xlim(self.min_x - 0.5, self.max_x + 0.5)
        self.ax.set_ylim(self.min_y - 0.5, self.max_y + 0.5)
        if len(self.gt_trajectory) > 0:
            gt_trajectory = np.array(self.gt_trajectory)
            self.ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label='Ground Truth', color='green')

        if len(self.noisy_trajectory) > 0:
            noisy_trajectory = np.array(self.noisy_trajectory)
            self.ax.plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], label='Noisy Odometry', color='red')

        if len(self.ekf_trajectory) > 0:
            ekf_trajectory = np.array(self.ekf_trajectory)
            self.ax.plot(ekf_trajectory[:, 0], ekf_trajectory[:, 1], label='EKF Odometry', color='blue')
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)
    

    def calculate_best_fit_line(self, points):
        x, y = points[:,0], points[:,1]
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
        return (int(point[0] // self.map_resolution), int(point[1] // self.map_resolution))

    def continuous_point(self, discretized_point):
        return (discretized_point[0] * self.map_resolution, discretized_point[1] * self.map_resolution)

def main(args=None):
    rclpy.init(args=args)
    lidar_ransac = LidarRansac()
    try:
        rclpy.spin(lidar_ransac)
    except KeyboardInterrupt:
        pass

    lidar_ransac.destroy_node()
    rclpy.shutdown()

    # Optional: save trajectories to file or plot again outside the loop if needed
    # ...

if __name__ == '__main__':
    main()
