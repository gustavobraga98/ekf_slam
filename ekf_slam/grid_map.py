import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math

class GridMappingNode(Node):
    def __init__(self):
        super().__init__('grid_mapping_node')

        # Parâmetros do mapa
        self.grid_res = 0.05  # Resolução do grid (metros por célula)
        self.grid_width = 100  # Largura inicial do grid (células)
        self.grid_height = 100  # Altura inicial do grid (células)
        self.origin_x = self.grid_width // 2  # Origem do robô no grid
        self.origin_y = self.grid_height // 2  # Origem do robô no grid

        # Inicializa o grid com valores desconhecidos (0.5)
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), 0.5)

        # Lógicas de evidência (log-odds)
        self.l_free = -0.4  # Evidência para livre
        self.l_occ = 0.85  # Evidência para ocupado

        # Publishers e subscribers
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        self.lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.pose_subscriber = self.create_subscription(Odometry, '/ekf_odom', self.pose_callback, 10)

        # Broadcaster de transformações TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Estado do robô [x, y, theta]
        self.state = [-1.9999, -0.5, 0.0]

    def pose_callback(self, msg):
        """Atualiza a pose estimada do robô com base no tópico de odometria."""
        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y

        # Extrai yaw da orientação (em 2D, considerando apenas a rotação em torno do eixo Z)
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.state[2] = math.atan2(siny_cosp, cosy_cosp)

        # Publica a transformação entre "map" e "base_link"
        self.publicar_transformacao()

        # Verifica se é necessário expandir o grid
        self.verificar_expansao_grid()

    def lidar_callback(self, msg):
        """Atualiza o mapa com os dados do LIDAR."""
        angulo_inicial = msg.angle_min
        incremento_angular = msg.angle_increment

        # Inicializar um conjunto de células alteradas para otimizar a publicação
        cells_to_update = set()

        # Processa os dados do LiDAR
        for i, distancia in enumerate(msg.ranges):
            if distancia < msg.range_min or distancia > msg.range_max:
                continue

            # Calcula a posição do obstáculo em coordenadas cartesianas
            angulo_atual = angulo_inicial + i * incremento_angular
            x_obstaculo = self.state[0] + distancia * np.cos(self.state[2] + angulo_atual)
            y_obstaculo = self.state[1] + distancia * np.sin(self.state[2] + angulo_atual)

            # Converte as coordenadas do mundo real para coordenadas de grid
            ix_grid, iy_grid = self.transformar_coordenadas_para_grid(x_obstaculo, y_obstaculo)

            # Ignora pontos fora do grid
            if ix_grid < 0 or ix_grid >= self.grid_width or iy_grid < 0 or iy_grid >= self.grid_height:
                continue

            # Marca a célula como ocupada (log-odds)
            self.occupancy_grid[iy_grid, ix_grid] += self.l_occ
            cells_to_update.add((ix_grid, iy_grid))

            # Traça uma linha do robô até o obstáculo para marcar células livres
            pos_x, pos_y = self.transformar_coordenadas_para_grid(self.state[0], self.state[1])
            pontos = self.bresenham(pos_x, pos_y, ix_grid, iy_grid)

            for (px, py) in pontos:
                if 0 <= px < self.grid_width and 0 <= py < self.grid_height:
                    self.occupancy_grid[py, px] += self.l_free
                    cells_to_update.add((px, py))

        # Se houver células modificadas, publique o mapa
        if cells_to_update:
            self.publicar_occupancy_grid()

    def verificar_expansao_grid(self):
        """Expande o grid se o robô estiver próximo das bordas."""
        # Margem em metros para a borda antes de expandir
        margem_metros = 1.0  # Margem de 1 metro

        # Calcula as margens em termos de células
        margem_celulas = int(margem_metros / self.grid_res)

        # Verifica se o robô está perto das bordas do grid
        pos_x, pos_y = self.transformar_coordenadas_para_grid(self.state[0], self.state[1])

        expandir_direita = pos_x >= self.grid_width - margem_celulas
        expandir_esquerda = pos_x <= margem_celulas
        expandir_cima = pos_y >= self.grid_height - margem_celulas
        expandir_baixo = pos_y <= margem_celulas

        if expandir_direita or expandir_esquerda or expandir_cima or expandir_baixo:
            self.expandir_grid(expandir_direita, expandir_esquerda, expandir_cima, expandir_baixo)

    def expandir_grid(self, direita, esquerda, cima, baixo):
        """Expande o grid quando necessário."""
        incremento = 25  # Número de células a adicionar

        if direita:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, 0), (0, incremento)), constant_values=0.5)
            self.grid_width += incremento

        if esquerda:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, 0), (incremento, 0)), constant_values=0.5)
            self.grid_width += incremento
            self.origin_x += incremento  # Ajusta a origem

        if cima:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, incremento), (0, 0)), constant_values=0.5)
            self.grid_height += incremento

        if baixo:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((incremento, 0), (0, 0)), constant_values=0.5)
            self.grid_height += incremento
            self.origin_y += incremento  # Ajusta a origem

    def publicar_occupancy_grid(self):
        """Publica o mapa como um OccupancyGrid para visualização no Rviz."""
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"
        grid_msg.info.resolution = self.grid_res
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height
        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = -(self.origin_x * self.grid_res)
        grid_msg.info.origin.position.y = -(self.origin_y * self.grid_res)
        grid_msg.info.origin.orientation.w = 1.0

        prob_grid = 1 - 1 / (1 + np.exp(self.occupancy_grid))
        grid_data = (prob_grid * 100).astype(np.int8)
        grid_data[self.occupancy_grid == 0.5] = -1  # Desconhecido

        grid_msg.data = grid_data.flatten().tolist()
        self.map_publisher.publish(grid_msg)

    def publicar_transformacao(self):
        """Publica a transformação entre o quadro 'map' e o quadro do robô."""
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"  # Quadro fixo
        t.child_frame_id = "base_link"  # Quadro do robô

        t.transform.translation.x = self.state[0]
        t.transform.translation.y = self.state[1]
        t.transform.translation.z = 0.0

        q = self.euler_para_quaternion(0, 0, self.state[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def euler_para_quaternion(roll, pitch, yaw):
        """Converte ângulos de Euler (roll, pitch, yaw) para quaternion."""
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return [qx, qy, qz, qw]

    def transformar_coordenadas_para_grid(self, x_mundo_real, y_mundo_real):
        """Converte coordenadas do mundo real para coordenadas de grid."""
        x_grid = int(x_mundo_real / self.grid_res) + self.origin_x
        y_grid = int(y_mundo_real / self.grid_res) + self.origin_y
        return x_grid, y_grid

    @staticmethod
    def bresenham(x1, y1, x2, y2):
        """Algoritmo de Bresenham para rasterizar linhas."""
        pontos = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            pontos.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return pontos


def main(args=None):
    rclpy.init(args=args)
    node = GridMappingNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()