from ekf_slam.utils import euler_from_quaternion, quaternion_from_euler
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from scipy.spatial import KDTree


class EKFNode(Node):
    def __init__(self):
        super().__init__('EKF_node')

        # Subscrições
        self.lidar_sub = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/noisy_odom", self.noisy_odom_callback, 10)

        # Publicação
        self.ekf_pub = self.create_publisher(Odometry, "/ekf_odom", 10)

        # Landmarks
        self.landmarks = []
        self.landmark_threshold = 0.2
        self.landmark_tree = None

        # Estado inicial
        self.position = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.velocity = np.array([0.0, 0.0])  # [linear, angular]
        self.prev_time = self.get_clock().now()

        # Controle de inicialização do EKF
        self.initialized = False
        self.initial_positions = []  # Armazena posições para cálculo da média inicial

        # Covariâncias
        self.cov_matrix = np.diag([0.1, 0.1, 0.1])
        self.error_matrix = np.diag([0.1, 0.1, 0.05])
        self.measurement_noise = np.diag([0.02, 0.02, 0.005])

        # Suavização
        self.smoothing_factor = 0.1
        self.robot_stopped = False  # Estado do robô baseado no /cmd_vel

        self.header = None

    def noisy_odom_callback(self, msg):
        """Captura a odometria inicial para iniciar o EKF."""
        if not self.initialized:
            # Extrai posição e orientação
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            orientation = msg.pose.pose.orientation
            self.header = msg.header

            # Chama a função para calcular o yaw
            yaw = euler_from_quaternion(orientation)  # Passa o quaternion diretamente

            # Armazena os valores para calcular a média
            self.initial_positions.append([x, y, 0])

            if len(self.initial_positions) >= 5:  # Média dos 5 primeiros pontos
                initial_mean = np.mean(self.initial_positions, axis=0)
                self.position = np.array(initial_mean)
                self.initialized = True  # Evita alterações futuras
                self.get_logger().info(f"Posição inicial definida: {self.position}")
        else:
            self.header = msg.header

    def cmd_vel_callback(self, msg):
        """Atualiza a velocidade com base no /cmd_vel."""
        self.velocity = np.array([msg.linear.x, msg.angular.z])
        self.robot_stopped = np.isclose(msg.linear.x, 0.0) and np.isclose(msg.angular.z, 0.0)

    def lidar_callback(self, msg):
        """Processa dados do LiDAR para identificar landmarks."""
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points = [
            [r * np.cos(angle), r * np.sin(angle)]
            for r, angle in zip(msg.ranges, angles)
            if msg.range_min < r < msg.range_max
        ]

        if points:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=self.landmark_threshold, min_samples=3).fit(points)
            unique_labels = set(clustering.labels_)
            self.landmarks = [
                np.mean(np.array(points)[clustering.labels_ == label], axis=0)
                for label in unique_labels if label != -1
            ]
            self.landmark_tree = KDTree(self.landmarks)

    def normalize_angle(self, angle):
        """Normaliza ângulo para [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def prediction(self, delta_time):
        """Executa a etapa de predição do EKF."""
        v, w = self.velocity
        theta = self.position[2]

        if not self.robot_stopped:
            # Se o robô estiver se movendo, realiza a predição
            delta_x = v * np.cos(theta) * delta_time
            delta_y = v * np.sin(theta) * delta_time
            delta_theta = w * delta_time
            self.position += np.array([delta_x, delta_y, self.normalize_angle(delta_theta)])

        # Matriz de transição de estado (F) considerando movimento
        F = np.array([
            [1, 0, -v * np.sin(theta) * delta_time],
            [0, 1, v * np.cos(theta) * delta_time],
            [0, 0, 1]
        ])

        # Se o robô está parado, evitamos qualquer mudança na posição e mantemos a covariância.
        if self.robot_stopped:
            F = np.eye(3)  # Nenhuma mudança no estado

        # Atualiza a covariância da predição
        self.cov_matrix = F @ self.cov_matrix @ F.T + self.error_matrix

    def update_state(self):
        """Atualiza o estado com base nas observações do LiDAR."""
        if not self.landmarks or self.landmark_tree is None:
            return

        for landmark in self.landmarks:
            # Coordenadas do landmark no mundo
            lx, ly = self.transform_landmark_to_world(landmark)

            # Predição da posição do landmark no mundo
            dx = lx - self.position[0]
            dy = ly - self.position[1]
            predicted_distance = np.sqrt(dx**2 + dy**2)
            predicted_angle = self.normalize_angle(np.arctan2(dy, dx) - self.position[2])

            # Observação esperada
            z_pred = np.array([predicted_distance, predicted_angle])

            # Observação real (simulada com ruído, se necessário)
            z_obs = z_pred  # Substitua por leituras reais do sensor, se aplicável.

            # Inovação (diferença entre observação e predição)
            innovation = z_obs - z_pred

            # Jacobiana H
            H = np.array([
                [-dx / predicted_distance, -dy / predicted_distance, 0],
                [dy / (predicted_distance**2), -dx / (predicted_distance**2), -1]
            ])

            # Se o robô está parado, aumente a incerteza para a odometria
            R = np.diag([0.02, 0.005])  # Confiança maior no LIDAR

            # Covariância da inovação
            S = H @ self.cov_matrix @ H.T + R

            # Ganhando de Kalman
            K = self.cov_matrix @ H.T @ np.linalg.inv(S)

            # Atualiza o estado
            delta_state = K @ innovation
            self.position += delta_state

            # Normaliza o ângulo
            self.position[2] = self.normalize_angle(self.position[2])

            # Atualiza a covariância
            self.cov_matrix = (np.eye(3) - K @ H) @ self.cov_matrix

    def transform_landmark_to_world(self, landmark):
        """Transforma coordenadas do robô para o mundo."""
        x_r, y_r = landmark
        theta = self.position[2]
        x_w = self.position[0] + x_r * np.cos(theta) - y_r * np.sin(theta)
        y_w = self.position[1] + x_r * np.sin(theta) + y_r * np.cos(theta)
        return x_w, y_w

    def publish_ekf_odom(self):
        """Publica a odometria corrigida."""
        if self.header:
            odom_msg = Odometry()
            odom_msg.header = self.header
            odom_msg.pose.pose.position.x = self.position[0]
            odom_msg.pose.pose.position.y = self.position[1]
            quat = quaternion_from_euler(self.position[2])
            odom_msg.pose.pose.orientation.x = quat[0]
            odom_msg.pose.pose.orientation.y = quat[1]
            odom_msg.pose.pose.orientation.z = quat[2]
            odom_msg.pose.pose.orientation.w = quat[3]
            self.ekf_pub.publish(odom_msg)
            self.header = None

    def update(self):
        """Chama a predição, atualização e publicação a cada ciclo"""
        current_time = self.get_clock().now()
        delta_time = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time

        # Predição e atualização de estado
        if delta_time > 0:
            self.prediction(delta_time)
            self.update_state()

        # Publica a odometria corrigida
        self.publish_ekf_odom()


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()

    # Define o timer para a atualização periódica
    timer_period = 0.1  # 10 Hz
    node.create_timer(timer_period, node.update)

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()