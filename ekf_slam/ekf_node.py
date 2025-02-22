import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.linalg import block_diag
from geometry_msgs.msg import Quaternion
import math

class EKFSLAM(Node):
    def __init__(self):
        super().__init__('ekf_slam')
        
        # EKF Configuration
        self.state = np.array([-1.999939, -0.5, 0.0])  # [x, y, theta]
        self.P = np.eye(3) * 0.1
        self.landmarks = []
        self.lm_cov = []
        
        # Parameters
        self.process_noise = np.diag([0.1, 0.5])  # Mais ruído angular       # Q (v, w)
        self.measurement_noise = np.diag([0.1, 0.1])  # R
        self.max_association_distance = 1.0            # Mahalanobis threshold
        self.dt = 0.1                                  # Sampling time
        self.is_rotating = False
        
        # ROS Subscribers
        self.scan_sub = Subscriber(self, LaserScan, '/scan')
        self.odom_sub = Subscriber(self, Odometry, '/noisy_odom')
        
        self.ts = ApproximateTimeSynchronizer(
            [self.scan_sub, self.odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sensor_callback)
        
        # ROS Publisher
        self.odom_pub = self.create_publisher(Odometry, '/ekf_odom', 10)
        
        # Visualization
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
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.predict(v, w)

    def predict(self, v, w):
        theta = self.state[2]
        self.is_rotating = abs(v) < 0.01 and abs(w) > 0.01

        if self.is_rotating:
            # Modelo de rotação pura
            dtheta = w * self.dt
            self.state[2] += dtheta
            
            # Jacobiana simplificada
            F = np.eye(len(self.state))
            F[2,2] = 1.0  # Mantém covariância angular
            
            # Atualização da covariância
            self.P = F @ self.P @ F.T + np.eye(len(self.state)) * 0.01
            
            self.get_logger().info(f"Pure rotation: dθ = {np.degrees(dtheta):.1f}°")
        else:
            # Improved motion model
            if abs(w) < 1e-6:
                # Movimento linear
                self.state[0] += v * np.cos(theta) * self.dt
                self.state[1] += v * np.sin(theta) * self.dt
                F = np.eye(3)
                F[0,2] = -v * np.sin(theta) * self.dt
                F[1,2] = v * np.cos(theta) * self.dt
            else:
                # Movimento circular
                radius = v / w
                dtheta = w * self.dt
                self.state[0] += radius * (np.sin(theta + dtheta) - np.sin(theta))
                self.state[1] += radius * (np.cos(theta) - np.cos(theta + dtheta))
                self.state[2] += dtheta
                F = np.eye(3)
                F[0,2] = radius * (np.cos(theta + dtheta) - np.cos(theta))
                F[1,2] = radius * (-np.sin(theta + dtheta) + np.sin(theta))
            
            self.state[2] = self.normalize_angle(self.state[2])
            
            # Jacobian for covariance prediction
            G = np.zeros((3, 2))
            if self.is_rotating:
                G[2,1] = self.dt  # Só afeta a orientação
            else:
                G[0,0] = np.cos(theta) * self.dt
                G[1,0] = np.sin(theta) * self.dt
                G[2,1] = self.dt
            
            n = len(self.landmarks) * 2
            F_full = block_diag(F, np.eye(n))
            G_full = np.vstack([G, np.zeros((n, 2))])
            
            # Covariance prediction
            self.P = F_full @ self.P @ F_full.T + G_full @ self.process_noise @ G_full.T

    def process_lidar(self, msg):
        clusters = self.cluster_lidar(msg)
        if not clusters:
            return
        
        for z in clusters:
            self.data_association(z)
        self.add_new_landmarks(clusters)

    def cluster_lidar(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        points = np.vstack((
            ranges[valid] * np.cos(angles[valid]),
            ranges[valid] * np.sin(angles[valid])
        )).T
        
        if len(points) < 10:
            return []
        
        dbscan = DBSCAN(eps=0.2, min_samples=5).fit(points)
        return [points[dbscan.labels_ == label].mean(axis=0) 
                for label in set(dbscan.labels_) if label != -1]

    def data_association(self, z):
        min_mahal = float('inf')
        best_idx = -1
        
        # Priorizar distância euclidiana durante rotação
        if self.is_rotating:
            for i, lm in enumerate(self.landmarks):
                dist = np.linalg.norm(z - self.compute_expected_measurement(i)[0])
                if dist < 0.3:  # Threshold reduzido
                    return self.update(z, i)
        
        # Caso contrário usar Mahalanobis normal
        else:
            for i, lm in enumerate(self.landmarks):
                z_hat, H, S = self.compute_expected_measurement(i)
                innovation = z - z_hat
                mahal = innovation.T @ np.linalg.inv(S) @ innovation
                
                if mahal < min_mahal and mahal < self.max_association_distance:
                    min_mahal = mahal
                    best_idx = i
            
            if best_idx != -1:
                self.update(z, best_idx)

    def compute_expected_measurement(self, lm_idx):
        lm = self.landmarks[lm_idx]
        dx = lm[0] - self.state[0]
        dy = lm[1] - self.state[1]
        theta = self.state[2]
        
        # Expected measurement
        z_hat = np.array([
            dx * np.cos(theta) + dy * np.sin(theta),
            -dx * np.sin(theta) + dy * np.cos(theta)
        ])
        
        # Jacobian
        H = np.zeros((2, 3 + 2 * len(self.landmarks)))
        H[0,0] = -np.cos(theta)
        H[0,1] = -np.sin(theta)
        H[0,2] = -dx * np.sin(theta) + dy * np.cos(theta)
        
        H[1,0] = np.sin(theta)
        H[1,1] = -np.cos(theta)
        H[1,2] = -dx * np.cos(theta) - dy * np.sin(theta)
        
        lm_start = 3 + 2 * lm_idx
        H[0, lm_start] = np.cos(theta)
        H[0, lm_start+1] = np.sin(theta)
        H[1, lm_start] = -np.sin(theta)
        H[1, lm_start+1] = np.cos(theta)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.measurement_noise
        return z_hat, H, S

    def update(self, z, lm_idx):
        z_hat, H, S = self.compute_expected_measurement(lm_idx)
        innovation = z - z_hat
        
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ innovation
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P
        self.state[2] = self.normalize_angle(self.state[2])

    def add_new_landmarks(self, clusters):
        theta = self.state[2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        for z in clusters:
            # Cálculo da posição global do landmark
            x = self.state[0] + z[0] * cos_theta - z[1] * sin_theta
            y = self.state[1] + z[0] * sin_theta + z[1] * cos_theta

            # Verificação de existência
            exists = False
            if self.landmarks:
                landmarks_array = np.array(self.landmarks)
                dists = np.linalg.norm(landmarks_array - [x,y], axis=1)
                if np.any(dists < 0.5):
                    exists = True

            if not exists:
                # Tamanho atual do estado antes de adicionar o novo landmark
                current_state_size = len(self.state)
                
                # Jacobiano em relação ao estado atual + medição
                J = np.zeros((2, current_state_size + 2))
                
                # Derivadas em relação ao estado do robô
                J[0, 0] = 1.0  # dx/drobot_x
                J[0, 2] = -z[0] * sin_theta - z[1] * cos_theta  # dx/dtheta
                
                J[1, 1] = 1.0  # dy/drobot_y
                J[1, 2] = z[0] * cos_theta - z[1] * sin_theta  # dy/dtheta
                
                # Derivadas em relação à medição (z0, z1)
                J[0, current_state_size] = cos_theta  # dx/dz0
                J[0, current_state_size + 1] = -sin_theta  # dx/dz1
                
                J[1, current_state_size] = sin_theta  # dy/dz0
                J[1, current_state_size + 1] = cos_theta  # dy/dz1

                # Cálculo da covariância aumentada
                P_aug = block_diag(self.P, self.measurement_noise)
                P_new = J @ P_aug @ J.T

                # Expansão da matriz de covariância
                new_P = np.zeros((current_state_size + 2, current_state_size + 2))
                new_P[:current_state_size, :current_state_size] = self.P
                new_P[current_state_size:, current_state_size:] = P_new
                
                # Cross-correlação
                cross_corr = self.P @ J[:, :current_state_size].T
                new_P[:current_state_size, current_state_size:] = cross_corr
                new_P[current_state_size:, :current_state_size] = cross_corr.T

                self.P = new_P
                self.landmarks.append([x, y])
                self.state = np.concatenate([self.state, [x, y]])

    def publish_odometry(self, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = 'odom'
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.orientation = self.yaw_to_quaternion(self.state[2])
        self.odom_pub.publish(odom)

    def update_visualization(self):
        if self.landmarks:
            landmarks = np.array(self.landmarks)
            self.scatter_landmarks.set_offsets(landmarks)
        self.robot_pose.set_offsets([self.state[0], self.state[1]])
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