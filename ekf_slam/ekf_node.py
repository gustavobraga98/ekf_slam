import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.linalg import block_diag
from geometry_msgs.msg import Quaternion, PoseWithCovariance, TwistWithCovariance

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.linalg import block_diag
from geometry_msgs.msg import Quaternion, PoseWithCovariance, TwistWithCovariance

class EKFSLAM(Node):
    def __init__(self):
        super().__init__('ekf_slam')
        
        # Configuração do EKF
        self.state = [-1.999939, -0.5, 0]  # [x, y, theta]
        self.P = np.eye(3) * 0.1  # Covariância inicial
        self.landmarks = []       # Lista de marcos [x_global, y_global]
        self.lm_cov = []          # Covariâncias dos marcos
        
        # Parâmetros (CORRIGIDO: Q agora 2x2)
        self.process_noise = np.diag([0.1, 0.05])    # Q (ruído em v e w)
        self.measurement_noise = np.diag([0.1, 0.1]) # R
        self.max_association_distance = 0.5
        
        # Subscrever tópicos sincronizados
        self.scan_sub = Subscriber(self, LaserScan, '/scan')
        self.odom_sub = Subscriber(self, Odometry, '/noisy_odom')
        
        self.ts = ApproximateTimeSynchronizer(
            [self.scan_sub, self.odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sensor_callback)
        
        # Publicadores
        self.odom_pub = self.create_publisher(Odometry, '/ekf_odom', 10)
        
        # Visualização
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.setup_plots()

    def setup_plots(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.scatter_lidar = self.ax.scatter([], [], c='b', s=5, label='LIDAR')
        self.scatter_landmarks = self.ax.scatter([], [], c='r', marker='x', label='Landmarks')
        self.robot_pose = self.ax.scatter([], [], c='g', marker='o', s=100, label='Robot')
        self.ax.legend()
        self.fig.canvas.draw()

    def sensor_callback(self, scan_msg, odom_msg):
        self.process_odometry(odom_msg)
        self.process_lidar(scan_msg)
        self.publish_odometry(odom_msg.header.stamp)
        self.update_visualization()

    def process_odometry(self, msg):
        dt = 0.1  # Período de amostragem aproximado
        
        # Extrair velocidades do twist
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        # Predição do EKF
        self.state_prediction(v, w, dt)
        self.covariance_prediction(v, w, dt)

    def state_prediction(self, v, w, dt):
        theta = self.state[2]
        self.state += [
            v * np.cos(theta) * dt,
            v * np.sin(theta) * dt,
            w * dt
        ]
        self.state[2] = self.normalize_angle(self.state[2])

    def covariance_prediction(self, v, w, dt):
        theta = self.state[2]
        
        # Matriz F do robô (3x3)
        F_robot = np.eye(3)
        F_robot[0,2] = -v * np.sin(theta) * dt
        F_robot[1,2] = v * np.cos(theta) * dt
        
        # Matriz G do robô (3x2)
        G_robot = np.array([
            [np.cos(theta) * dt, 0],
            [np.sin(theta) * dt, 0],
            [0, dt]
        ])
        
        # Expansão para marcos existentes
        n_landmarks = len(self.landmarks) * 2
        F = block_diag(F_robot, np.eye(n_landmarks))  # (3+2n x 3+2n)
        G = np.vstack([G_robot, np.zeros((n_landmarks, 2))])  # (3+2n x 2)
        
        # Atualizar covariância (CORRIGIDO: usar Q 2x2)
        self.P = F @ self.P @ F.T + G @ self.process_noise @ G.T

    def process_lidar(self, msg):
        clusters = self.cluster_lidar(msg)
        if not clusters:
            return
        
        # Processar medições
        for z in clusters:
            self.data_association(z)
        
        # Adicionar novos marcos
        self.add_new_landmarks(clusters)

    def cluster_lidar(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Converter para coordenadas cartesianas
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        points = np.vstack((
            ranges[valid] * np.cos(angles[valid]),
            ranges[valid] * np.sin(angles[valid])
        )).T
        
        # Clusterização
        if len(points) < 10:
            return []
        
        dbscan = DBSCAN(eps=0.2, min_samples=5).fit(points)
        clusters = []
        
        for label in set(dbscan.labels_):
            if label == -1:
                continue
            cluster = points[dbscan.labels_ == label]
            clusters.append(cluster.mean(axis=0))
        
        return clusters

    def data_association(self, z):
        min_dist = float('inf')
        best_idx = -1
        
        for i, lm in enumerate(self.landmarks):
            # Converter marco para coordenadas do robô
            dx = lm[0] - self.state[0]
            dy = lm[1] - self.state[1]
            theta = self.state[2]
            
            # Transformação esperada
            z_hat = np.array([
                dx * np.cos(theta) + dy * np.sin(theta),
                -dx * np.sin(theta) + dy * np.cos(theta)
            ])
            
            # Calcular distância
            innovation = z - z_hat
            distance = np.linalg.norm(innovation)
            
            if distance < min_dist and distance < self.max_association_distance:
                min_dist = distance
                best_idx = i
        
        if best_idx != -1:
            self.ekf_update(z, best_idx)

    def ekf_update(self, z, lm_idx):
        # Obter posição do marco
        lm = self.landmarks[lm_idx]
        
        # Calcular Jacobiano
        dx = lm[0] - self.state[0]
        dy = lm[1] - self.state[1]
        theta = self.state[2]
        
        H = np.array([
            [-np.cos(theta), -np.sin(theta), -dx * np.sin(theta) + dy * np.cos(theta)],
            [np.sin(theta), -np.cos(theta), -dx * np.cos(theta) - dy * np.sin(theta)]
        ])
        
        # Calcular inovação
        z_hat = np.array([
            dx * np.cos(theta) + dy * np.sin(theta),
            -dx * np.sin(theta) + dy * np.cos(theta)
        ])
        innovation = z - z_hat
        
        # Atualização do EKF
        S = H @ self.P[:3,:3] @ H.T + self.measurement_noise
        K = self.P[:3,:3] @ H.T @ np.linalg.inv(S)
        
        self.state[:3] += K @ innovation
        self.P[:3,:3] = (np.eye(3) - K @ H) @ self.P[:3,:3]
        
        # Garantir consistência angular
        self.state[2] = self.normalize_angle(self.state[2])

    def add_new_landmarks(self, clusters):
        for z in clusters:
            # Converter para coordenadas globais
            x = self.state[0] + z[0] * np.cos(self.state[2]) - z[1] * np.sin(self.state[2])
            y = self.state[1] + z[0] * np.sin(self.state[2]) + z[1] * np.cos(self.state[2])
            
            # Verificar se já existe
            exists = False
            for lm in self.landmarks:
                if np.linalg.norm([x - lm[0], y - lm[1]]) < self.max_association_distance:
                    exists = True
                    break
            
            if not exists:
                self.landmarks.append([x, y])
                # Expandir covariância corretamente
                new_cov = np.eye(2) * 0.1
                self.P = block_diag(self.P, new_cov)

    def publish_odometry(self, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.orientation = self.yaw_to_quaternion(self.state[2])
        
        self.odom_pub.publish(odom)

    def update_visualization(self):
        # Atualizar posição do robô
        self.robot_pose.set_offsets([self.state[0], self.state[1]])
        
        # Atualizar marcos
        if self.landmarks:
            landmarks = np.array(self.landmarks)
            self.scatter_landmarks.set_offsets(landmarks)
        
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    @staticmethod
    def yaw_to_quaternion(yaw):
        q = Quaternion()
        q.w = np.cos(yaw / 2)
        q.z = np.sin(yaw / 2)
        return q

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        plt.close('all')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()