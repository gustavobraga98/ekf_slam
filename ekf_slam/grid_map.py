import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
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

        # Inicializa grid com valores desconhecidos
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), 0.5)

        # Lógicas de evidência (log-odds)
        self.l_free = -0.4  # Evidência para livre
        self.l_occ = 0.85  # Evidência para ocupado

        # Publishers e subscribers
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        self.lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.pose_subscriber = self.create_subscription(Odometry, '/ekf_pos', self.pose_callback, 10)

        # Estado do robô
        self.state = [0.0, 0.0, 0.0]  # [x, y, theta]

    def pose_callback(self, msg):
        """Atualiza a pose estimada do robô com base no tópico de pose filtrada."""
        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y

        # Extrai yaw da orientação
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.state[2] = math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        """Atualiza o mapa baseado nos dados do LiDAR."""
        self.verificar_expansao_grid()
        angulo_inicial = msg.angle_min
        incremento_angular = msg.angle_increment

        for i, distancia in enumerate(msg.ranges):
            if distancia < msg.range_min or distancia > msg.range_max:
                continue

            # Calcula posição do obstáculo em coordenadas reais
            angulo_atual = angulo_inicial + i * incremento_angular
            x_obstaculo = self.state[0] + distancia * np.cos(self.state[2] + angulo_atual)
            y_obstaculo = self.state[1] + distancia * np.sin(self.state[2] + angulo_atual)

            # Converte para coordenadas de grid
            ix_grid, iy_grid = self.transformar_coordenadas_para_grid(x_obstaculo, y_obstaculo)

            # Ignora pontos fora do grid
            if ix_grid < 0 or ix_grid >= self.grid_width or iy_grid < 0 or iy_grid >= self.grid_height:
                continue

            # Traça linha do robô até o obstáculo (células livres)
            pos_x, pos_y = self.transformar_coordenadas_para_grid(self.state[0], self.state[1])
            pontos = self.bresenham(pos_x, pos_y, ix_grid, iy_grid)

            for (px, py) in pontos:
                if 0 <= px < self.grid_width and 0 <= py < self.grid_height:
                    self.occupancy_grid[py, px] += self.l_free

            # Marca célula como ocupada
            self.occupancy_grid[iy_grid, ix_grid] += self.l_occ

        self.publicar_occupancy_grid()

    def publicar_occupancy_grid(self):
        """Publica o mapa em formato OccupancyGrid para visualização no RViz."""
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

        # Converte log-odds para probabilidade
        prob_grid = 1 - 1 / (1 + np.exp(self.occupancy_grid))
        grid_data = (prob_grid * 100).astype(np.int8)
        grid_data[prob_grid == 0.5] = -1  # Desconhecido

        grid_msg.data = grid_data.flatten().tolist()
        self.map_publisher.publish(grid_msg)

    def verificar_expansao_grid(self):
        """Expande a grid se o robô estiver próximo das bordas."""
        margem = 10  # Células antes de expandir
        expandir_direita = self.state[0] + margem >= self.grid_width
        expandir_esquerda = self.state[0] < margem
        expandir_cima = self.state[1] + margem >= self.grid_height
        expandir_baixo = self.state[1] < margem

        if expandir_direita or expandir_esquerda or expandir_cima or expandir_baixo:
            self.expandir_grid(expandir_direita, expandir_esquerda, expandir_cima, expandir_baixo)

    def expandir_grid(self, direita, esquerda, cima, baixo):
        """Expande o grid quando necessário."""
        incremento = 25

        if direita:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, 0), (0, incremento)), constant_values=0.5)
            self.grid_width += incremento
        if esquerda:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, 0), (incremento, 0)), constant_values=0.5)
            self.grid_width += incremento
            self.origin_x += incremento
        if cima:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, incremento), (0, 0)), constant_values=0.5)
            self.grid_height += incremento
        if baixo:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((incremento, 0), (0, 0)), constant_values=0.5)
            self.grid_height += incremento
            self.origin_y += incremento

    def transformar_coordenadas_para_grid(self, x_mundo_real, y_mundo_real):
        x_grid = int(x_mundo_real / self.grid_res) + self.origin_x
        y_grid = int(y_mundo_real / self.grid_res) + self.origin_y
        return x_grid, y_grid

    @staticmethod
    def bresenham(x1, y1, x2, y2):
        """Implementação do algoritmo de Bresenham para rasterização de linha."""
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
