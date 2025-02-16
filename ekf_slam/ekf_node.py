import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class EKFNode(Node):
    def __init__(self):
        super().__init__('EKF_node')

        # Subscrições e Publicações
        self.lidar_sub = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/noisy_odom", self.odom_callback, 10)
        self.ekf_pub = self.create_publisher(Odometry, "/ekf_odom", 10)

        # Estado inicial [x, y, theta]
        self.initial_state = np.array([-1.9999, -0.5, 0.0])
        self.ekf_state = self.initial_state.copy()

        # Covariância inicial
        self.P = np.diag([0.1, 0.1, np.deg2rad(5)])

        # Lista de marcos conhecidos
        self.known_landmarks = []
        self.last_odom_time = None
        plt.ion()  # Modo interativo
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.lidar_points_plot, = self.ax.plot([], [], 'b.', label='LIDAR Points')
        self.landmarks_plot, = self.ax.plot([], [], 'ro', label='Landmarks')
        self.robot_position_plot, = self.ax.plot([], [], 'go', label='Robot Position')  # Ponto verde para a posição do robô
        self.ax.legend()

    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Converter medições LIDAR para coordenadas cartesianas
        points = np.array([
            (ranges[i] * np.cos(angles[i]), ranges[i] * np.sin(angles[i]))
            for i in range(len(ranges)) if msg.range_min < ranges[i] < msg.range_max
        ])

        # Transformar pontos do LIDAR para o quadro global
        global_points = self.lidar_to_global(points, self.ekf_state)

        # Aplicar DBSCAN para encontrar clusters
        db = DBSCAN(eps=0.2, min_samples=5).fit(global_points)
        labels = db.labels_

        # Calcular a posição média de cada cluster
        unique_labels = set(labels)
        clusters = []
        for label in unique_labels:
            if label == -1:
                continue  # Ignorar ruído
            cluster_points = global_points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            clusters.append(cluster_center)

        # Associar clusters a marcos conhecidos
        self.associate_landmarks_and_estimate_position(clusters)
        self.visualize_lidar_and_landmarks(global_points, clusters)

    def lidar_to_global(self, points, robot_pose):
        # Extrair a posição e orientação do robô
        x, y, theta = robot_pose

        # Transformar para o quadro global
        global_points = []
        for px, py in points:
            x_global = x + px * np.cos(theta) - py * np.sin(theta)
            y_global = y + px * np.sin(theta) + py * np.cos(theta)
            global_points.append((x_global, y_global))
        
        # Log para debugging
        # self.get_logger().info(f"Global LIDAR Points: {global_points}")
        
        return np.array(global_points)

    def visualize_lidar_and_landmarks(self, points, clusters):
        # Atualizar pontos do LIDAR
        self.lidar_points_plot.set_data(points[:, 0], points[:, 1])

        # Atualizar marcos conhecidos com cores diferentes
        if clusters:
            clusters = np.array(clusters)
            num_clusters = len(clusters)
            colors = cm.rainbow(np.linspace(0, 1, num_clusters))  # Gerar cores diferentes para cada cluster

            for i, cluster in enumerate(clusters):
                self.ax.plot(cluster[0], cluster[1], 'o', color=colors[i], label=f'Cluster {i}')

        # Atualizar a posição do robô
        self.robot_position_plot.set_data([self.ekf_state[0]], [self.ekf_state[1]])  # Passar como listas de um único elemento

        # Atualizar o plot
        self.ax.set_title("LIDAR and Landmarks")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    def associate_landmarks_and_estimate_position(self, clusters):
        for cluster in clusters:
            dx, dy = cluster
            theta = self.ekf_state[2]
            global_x = self.ekf_state[0] + (dx * np.cos(theta) - dy * np.sin(theta))
            global_y = self.ekf_state[1] + (dx * np.sin(theta) + dy * np.cos(theta))
            global_cluster = np.array([global_x, global_y])

            # Log para debugging
            self.get_logger().info(f"Global Cluster: {global_cluster}")

            if not self.known_landmarks:
                self.known_landmarks.append(global_cluster)
                continue

            distances = [np.linalg.norm(global_cluster - known) for known in self.known_landmarks]
            min_distance = min(distances)
            if min_distance < 0.5:
                closest_landmark = self.known_landmarks[np.argmin(distances)]
                self.ekf_state[:2] = closest_landmark - cluster  # Update position directly
            else:
                self.known_landmarks.append(global_cluster)

        self.update(self.ekf_state[:2])
        self.get_logger().info(f"Number of landmarks: {len(self.known_landmarks)}")
        self.get_logger().info(f"Estimated position from LIDAR: {self.ekf_state}")

    def odom_callback(self, msg: Odometry):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        # Calcular delta_t
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_odom_time is None:
            self.last_odom_time = current_time
            self.last_position = np.array([position.x, position.y])
            self.last_yaw = yaw
            return

        delta_t = current_time - self.last_odom_time
        self.last_odom_time = current_time

        # Calcular velocidades linear e angular
        delta_pos = np.array([position.x, position.y]) - self.last_position
        self.last_position = np.array([position.x, position.y])
        v = np.linalg.norm(delta_pos) / delta_t

        delta_yaw = yaw - self.last_yaw
        self.last_yaw = yaw
        omega = delta_yaw / delta_t

        # Passar dados de odometria e delta_t para a previsão
        self.get_logger().info(f"Received odometry data: position=({position.x}, {position.y}), yaw={yaw}, v={v}, omega={omega}, delta_t={delta_t}")
        self.predict(v, omega, delta_t)

    def predict(self, v, omega, delta_t):
        # Extrair dados de odometria
        x, y, theta = self.ekf_state

        # Modelo de movimento
        x_pred = x + v * delta_t * np.cos(theta)
        y_pred = y + v * delta_t * np.sin(theta)
        theta_pred = theta + omega * delta_t

        # Atualizar o estado previsto
        self.ekf_state = np.array([x_pred, y_pred, theta_pred])

        # Matriz Jacobiana do modelo de movimento
        F = np.array([
            [1, 0, -v * delta_t * np.sin(theta)],
            [0, 1, v * delta_t * np.cos(theta)],
            [0, 0, 1]
        ])

        # Covariância do processo
        Q = np.diag([0.1, 0.1, np.deg2rad(1)])  # Ajuste conforme necessário

        # Atualizar a covariância do estado
        self.P = F @ self.P @ F.T + Q

    def update(self, lidar_measurement):
        z = lidar_measurement
        z_hat = self.ekf_state[:2]
        y = z - z_hat

        H = np.array([[1, 0, 0],
                      [0, 1, 0]])
        R = np.diag([0.05, 0.05])
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.ekf_state = self.ekf_state + K @ y
        I = np.eye(len(self.ekf_state))
        self.P = (I - K @ H) @ self.P

# Função auxiliar para converter quaternion para ângulos de Euler
def euler_from_quaternion(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()