import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


def bresenham(x0, y0, x1, y1):
    """
    Executa o algoritmo de Bresenham para determinar os pontos entre
    duas coordenadas (x0, y0) -> (x1, y1).
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        print(f"Bresenham pontos: x={x0}, y={y0}")
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

class OdomNoiseNode(Node):
    
    def __init__(self):
        super().__init__('odom_lidar_node')

        # ---- ASSINATURAS PARA LIDAR E ODOMETRIA ---- #
        self.scan_sub = Subscriber(self, LaserScan, "/scan")
        self.odom_sub = Subscriber(self, Odometry, "/odom")

        # ---- SINCRONIZAÇÃO DE MENSAGENS ---- #
        self.ts = ApproximateTimeSynchronizer(
            [self.scan_sub, self.odom_sub],
            queue_size=30,
            slop=0.01
        )
        self.ts.registerCallback(self.sync_callback)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ---- PARÂMETROS DA OCCUPANCY GRID ---- #
        self.grid_res = 0.1  # Resolução: 10 cm por célula
        self.grid_width = 50  # Inicial: 5 metros de largura (50 células)
        self.grid_height = 50  # Inicial: 5 metros de altura (50 células)

        # Inicializa a Occupancy Grid com valores desconhecidos: 0.5 (50% chances livres ou ocupadas)
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), 0.5)

        # ---- POSIÇÃO DO ROBÔ NO GRID ---- #
        # O robô começa no centro da grade inicial
        self.origin_x = self.grid_width // 2
        self.origin_y = self.grid_height // 2
        self.posicao_robô_x = self.origin_x
        self.posicao_robô_y = self.origin_y

        self.x = 0.0  # Posição inicial no eixo x
        self.y = 0.0  # Posição inicial no eixo y
        self.theta = 0.0  # Orientação (yaw)

        # ---- PUBLICADOR DO MAPA PARA O RVIZ ---- #
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        self.map_pub_rate = 2.0  # Taxa de publicação do mapa
        self.timer = self.create_timer(1.0 / self.map_pub_rate, self.publicar_occupancy_grid)
        self.timer2 = self.create_timer(0.1, self.publicar_transformacao)

        # Log
        self.get_logger().info(f"Occupancy Grid iniciada com {self.grid_width}x{self.grid_height} células.")

    def publicar_transformacao(self):
        """
        Publica a transformação estática de odom->map
        """
        # Criamos uma transformação estática entre o `map` e o `odom`
        t = TransformStamped()

        # Preencha com timestamp
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"  # Frame de referência mapeada será o 'map'
        t.child_frame_id = "odom"  # 'odom' será a child frame

        # A transformação (Translations e Rotations)
        # Aqui nós definimos uma transformação neutra (sem deslocamento e sem rotação)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Publicar a transformação
        self.tf_broadcaster.sendTransform(t)

    
    def sync_callback(self, scan_msg, odom_msg):
        """
        Callback que é chamado com dados sincronizados de LIDAR e Odometria.
        Atualizará a posição do robô com base na odometria.
        """

        # ---- PREDIÇÃO USANDO ODOMETRIA ---- #
        
        # 1. Extrair a posição do robô a partir da odometria (pose.pose.position)
        self.x = odom_msg.pose.pose.position.x
        self.y = odom_msg.pose.pose.position.y
        
        # 2. Converter a orientação do robô (quaternion -> yaw) usando a função apropriada
        self.theta = self.get_yaw_from_quaternion(odom_msg.pose.pose.orientation)

        # Log de saída para verificar se estamos recebendo corretamente os dados
        self.get_logger().info(f"Pose Atualizada com Odometria - x: {self.x}, y: {self.y}, theta: {self.theta} graus")
        
        # ---- ATUALIZE O MAPA USANDO OS DADOS DO LIDAR ---- #
        # Vamos converter a posição do robô para a grade e garantir que estamos chamando esta função
        self.posicao_robô_x, self.posicao_robô_y = self.transformar_coordenadas_para_grid(self.x, self.y)
        self.atualizar_mapa_lidar(scan_msg, self.posicao_robô_x, self.posicao_robô_y)

    def transformar_coordenadas_para_grid(self, x_mundo_real, y_mundo_real):
        """
        Converte coordenadas do 'mundo real' (odometria) para células do Occupancy Grid.
        """
        x_grid = int(x_mundo_real / self.grid_res) + self.origin_x
        y_grid = int(y_mundo_real / self.grid_res) + self.origin_y
        return x_grid, y_grid

    def verificar_expansao_grid(self):
        """
        Verifica se o robô está próximo das bordas da grade e expande se necessário.
        """
        margem_de_expansao = 5  # Margem de segurança antes de chegar à borda do grid
        expandir = False

        if self.posicao_robô_x + margem_de_expansao >= self.grid_width:
            self.get_logger().debug("Expandir à direita.")
            expandir_direita = True
            expandir = True
        else:
            expandir_direita = False

        if self.posicao_robô_x < margem_de_expansao:
            self.get_logger().debug("Expandir à esquerda.")
            expandir_esquerda = True
            expandir = True
        else:
            expandir_esquerda = False

        if self.posicao_robô_y + margem_de_expansao >= self.grid_height:
            self.get_logger().debug("Expandir para cima.")
            expandir_cima = True
            expandir = True
        else:
            expandir_cima = False

        if self.posicao_robô_y < margem_de_expansao:
            self.get_logger().debug("Expandir para baixo.")
            expandir_baixo = True
            expandir = True
        else:
            expandir_baixo = False

        if expandir:
            self.expandir_grid(expandir_direita, expandir_esquerda, expandir_cima, expandir_baixo)

    def expandir_grid(self, direita, esquerda, cima, baixo):
        """
        Expande a Occupancy Grid conforme a necessidade baseada na posição atual do robô.
        """
        self.get_logger().info("Expandindo a Occupancy Grid...")

        incremento = 25
        grid_expandida = False

        if direita:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, 0), (0, incremento)), constant_values=0.5)
            self.grid_width += incremento
            grid_expandida = True

        if esquerda:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, 0), (incremento, 0)), constant_values=0.5)
            self.grid_width += incremento
            self.origin_x += incremento
            grid_expandida = True

        if cima:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((0, incremento), (0, 0)), constant_values=0.5)
            self.grid_height += incremento
            grid_expandida = True

        if baixo:
            self.occupancy_grid = np.pad(self.occupancy_grid, ((incremento, 0), (0, 0)), constant_values=0.5)
            self.grid_height += incremento
            self.origin_y += incremento
            grid_expandida = True

        if grid_expandida:
            self.get_logger().info(f"Grid expandida para {self.grid_width} x {self.grid_height}. Nova origem: ({self.origin_x}, {self.origin_y})")

    def atualizar_mapa_lidar(self, scan_msg, posicao_robô_x, posicao_robô_y):
        """
        Projeta as leituras do LIDAR no mapa (Occupancy Grid),
        atualizando as células e expandindo a grid se necessário.
        """
        self.verificar_expansao_grid()

        angulo_inicial = scan_msg.angle_min
        incremento_angular = scan_msg.angle_increment

        self.get_logger().info(f"LIDAR ranges count: {len(scan_msg.ranges)}, range_min: {scan_msg.range_min}, range_max: {scan_msg.range_max}, angle_min: {scan_msg.angle_min}, angle_increment: {scan_msg.angle_increment}")

        ocupadas = 0  # Contador para saber quantas células estamos marcando

        for i, distancia in enumerate(scan_msg.ranges):
            if distancia <= scan_msg.range_min or distancia >= scan_msg.range_max:
                self.get_logger().debug(f"Dado LIDAR inválido - distância: {distancia}")
                continue  # Se o dado for inválido, ignoramos

            angulo_atual = angulo_inicial + i * incremento_angular
            x_obstaculo = self.x + distancia * np.cos(self.theta + angulo_atual)
            y_obstaculo = self.y + distancia * np.sin(self.theta + angulo_atual)

            self.get_logger().debug(f"LIDAR coordenadas no mundo: x_obstaculo = {x_obstaculo}, y_obstaculo = {y_obstaculo}")

            ix_grid = int((x_obstaculo / self.grid_res) + self.origin_x)
            iy_grid = int((y_obstaculo / self.grid_res) + self.origin_y)

            self.get_logger().debug(f"Coordenadas no grid: ix_grid = {ix_grid}, iy_grid = {iy_grid}")

            if ix_grid < 0 or ix_grid >= self.grid_width or iy_grid < 0 or iy_grid >= self.grid_height:
                self.get_logger().debug(f"Coordenadas fora do grid: ix_grid = {ix_grid}, iy_grid = {iy_grid}")
                continue  # Pula se os valores projetados estiverem fora do grid atual

            pontos = bresenham(self.posicao_robô_x, self.posicao_robô_y, ix_grid, iy_grid)

            for (px, py) in pontos:
                if 0 <= px < self.grid_width and 0 <= py < self.grid_height:
                    self.occupancy_grid[py, px] = 0  # Marca como livre
                    self.get_logger().debug(f"Ponto livre marcado: px = {px}, py = {py}")

            if 0 <= ix_grid < self.grid_width and 0 <= iy_grid < self.grid_height:
                self.occupancy_grid[iy_grid, ix_grid] = 1  # Marca o obstáculo como ocupado
                ocupadas += 1
                self.get_logger().debug(f"Ponto ocupado marcado: ix_grid = {ix_grid}, iy_grid = {iy_grid}")

        self.get_logger().info(f"Leituras do LIDAR projetadas: {ocupadas} células marcadas.")

    def publicar_occupancy_grid(self):
        """
        Publica a Occupancy Grid no formato correto para ser visualizada no RViz.
        """
        grid_msg = OccupancyGrid()
        
        # Configurando o cabeçalho da mensagem
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"

        # Configurando a resolução, a largura e a altura (informações sobre o mapa)
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = self.grid_res
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height

        # Definir a origem do mapa (posição de referência do grid no mundo real)
        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = -(self.origin_x * self.grid_res)
        grid_msg.info.origin.position.y = -(self.origin_y * self.grid_res)
        grid_msg.info.origin.orientation.w = 1.0  # Sem rotação

        # Normalizar os dados da Occupancy Grid
        # Inicialmente, uma cópia do grid_data
        grid_data = np.copy(self.occupancy_grid)
        
        # Passo 1: Converter valores desconhecidos (0.5) para -1
        grid_data[grid_data == 0.5] = -1  # Define -1 para as células "desconhecidas"

        # Passo 2: Multiplicar os valores normais por 100 (ocupação)
        grid_data[grid_data == 1] = 100  # Define 100 para as células "ocupadas"
        grid_data[grid_data == 0] = 0   # Define 0 para as células "livres"

        # Passo 3: Limitar os valores, garantindo que nenhum esteja fora da faixa de ocupação ROS.
        grid_data = np.clip(grid_data, -1, 100)  # Garantir valores entre -1 (desconhecido) e 100 (ocupado)

        # Passo 4: **Aqui está o ajuste crítico** — garantir que as células estão no formato `int8`
        grid_data = grid_data.astype(np.int8)  # Conversão direta para o tipo `int8`
        
        # Passo 5: Converter o array NumPy para uma sequência de inteiros (pois é isso que o ROS OccupancyGrid espera)
        grid_msg.data = grid_data.flatten().tolist()  # Sequência de valores "int" necessária pelo ROS

        # Publica o mapa no tópico '/map'
        self.map_publisher.publish(grid_msg)

        # Logger para saber que o mapa está sendo publicado corretamente
        self.get_logger().info(f"Mapa publicado no tópico '/map'. Tamanho: {len(grid_msg.data)} células.")


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

    def get_yaw_from_quaternion(self, orientation_q):
        """
        Converte uma mensagem Quaternion (utilizada na odometria) para o ângulo yaw (theta).
        """
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = self.euler_from_quaternion(orientation_list)
        return yaw
    
    

def main(args=None):
    rclpy.init(args=args)
    node = OdomNoiseNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
