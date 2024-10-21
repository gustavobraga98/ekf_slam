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

        # ---- EKF SLAM ---- #
        self.state = np.array([self.x, self.y, self.theta])
        self.covariance = np.eye(3) * 0.1
        self.motion_noise = np.diag([0.1, 0.1, np.deg2rad(1)])**2
        self.sensor_noise = np.diag([0.1, 0.1])**2

        # Contador para ignorar as primeiras medições
        self.leituras_ignoradas = 0
        self.leituras_para_ignorar = 10  # Número de leituras que você quer ignorar

        # Log
        self.get_logger().info(f"Occupancy Grid iniciada com {self.grid_width}x{self.grid_height} células.")

    def publicar_transformacao(self):
        """
        Publica a transformação dinâmica de map -> odom,
        baseada na estimativa corrigida do robô.
        """
        t = TransformStamped()

        # Preencha com timestamp
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        # Use a posição estimada pelo EKF para a transformação
        t.transform.translation.x = self.state[0]
        t.transform.translation.y = self.state[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = np.sin(self.state[2] / 2)
        t.transform.rotation.w = np.cos(self.state[2] / 2)

        # Publicar a transformação
        self.tf_broadcaster.sendTransform(t)

    
    def sync_callback(self, scan_msg, odom_msg):
        self.leituras_ignoradas += 1

        # Extrair posição global usando odometria
        global_x, global_y = self.localizar_posicao_global(odom_msg)

        # PREDIÇÃO DO EKF
        previous_state = self.state.copy()  # Salvar para comparação
        delta_x = global_x - previous_state[0]
        delta_y = global_y - previous_state[1]
        delta_theta = self.get_yaw_from_quaternion(odom_msg.pose.pose.orientation) - previous_state[2]

        control_input = np.array([delta_x, delta_y, delta_theta])
        self.state, self.covariance = self.kalman_predict(self.state, self.covariance, control_input, self.motion_noise)

        # LOG: Predição
        self.get_logger().debug(f"Controle preditivo: delta_x={delta_x}, delta_y={delta_y}, delta_theta={np.rad2deg(delta_theta)}°")

        # CORREÇÃO DO EKF
        z = np.array([global_x, global_y])  # Medição simulada, pode-se ajustar dependendo da real medida do LiDAR
        self.state, self.covariance = self.kalman_update(self.state, self.covariance, z, self.sensor_noise)

        # LOG: Correção
        self.get_logger().debug(f"Estado predito: {self.state}")

        # ATUALIZAÇÃO DA GRID
        self.posicao_robô_x, self.posicao_robô_y = self.transformar_coordenadas_para_grid(global_x, global_y)
        if self.leituras_ignoradas >= self.leituras_para_ignorar:
            self.get_logger().debug("Atualizando mapa com LIDAR")
            self.atualizar_mapa_lidar(scan_msg, self.posicao_robô_x, self.posicao_robô_y)

    def kalman_predict(self, state, covariance, control_input, motion_noise):
        """
        Fase de predição do Filtro de Kalman.
        Prediz o estado baseado no controle aplicado.
        """
        # Atualizar o estado com o controle
        state = state + control_input

        # Modelo de movimento - linear aproximação
        F = np.eye(3)  # Matriz de transição de estado

        # Atualização da covariância
        covariance = F @ covariance @ F.T + motion_noise
        
        return state, covariance
    
    def localizar_posicao_global(self, odom_msg):
        # Transformar odometria, SAX proxy for calcular a posição global mais precisa
        return odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y

    def get_yaw_from_quaternion(self, orientation_q):
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = self.euler_from_quaternion(orientation_list)
        return yaw

    def kalman_update(self, state, covariance, z, R):
        """
        Fase de correção do Filtro de Kalman.
        'z' são as medições do LiDAR; 'R' é a matriz de ruído da observação.
        """
        # Atualize H para refletir somente as estimativas de x e y
        H = np.eye(3)[:2, :]  # Selecionando apenas os dois primeiros componentes do estado para a atualização
        
        # Calcular o erro de inovação somente no espaço de observação
        y = z - H @ state

        # Calcular a matriz de inovação
        S = H @ covariance @ H.T + R

        # Calcular o ganho de Kalman
        K = covariance @ H.T @ np.linalg.inv(S)

        # Atualizar o estado
        state = state + K @ y

        # Atualizar a covariância
        I = np.eye(covariance.shape[0])
        covariance = (I - K @ H) @ covariance

        return state, covariance

    def transformar_coordenadas_para_grid(self, x_mundo_real, y_mundo_real):
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
        Projeta as leituras do LiDAR no mapa (Occupancy Grid),
        atualizando as células e expandindo a grid se necessário.
        """
        self.verificar_expansao_grid()

        angulo_inicial = scan_msg.angle_min
        incremento_angular = scan_msg.angle_increment
        
        # Inicialize a variável `ocupadas`
        ocupadas = 0

        for i, distancia in enumerate(scan_msg.ranges):
            # Ignorar leituras inválidas
            if distancia <= scan_msg.range_min or distancia >= scan_msg.range_max:
                continue

            angulo_atual = angulo_inicial + i * incremento_angular

            # Coordenadas no mundo real do ponto de final de leitura do LiDAR
            x_obstaculo = self.state[0] + distancia * np.cos(self.state[2] + angulo_atual)
            y_obstaculo = self.state[1] + distancia * np.sin(self.state[2] + angulo_atual)

            # Converte coordenadas do mundo real para o grid
            ix_grid = int((x_obstaculo / self.grid_res) + self.origin_x)
            iy_grid = int((y_obstaculo / self.grid_res) + self.origin_y)

            self.get_logger().debug(f"Coordenadas no grid: ix_grid = {ix_grid}, iy_grid = {iy_grid}")

            if ix_grid < 0 or ix_grid >= self.grid_width or iy_grid < 0 or iy_grid >= self.grid_height:
                self.get_logger().debug(f"Coordenadas fora do grid: ix_grid = {ix_grid}, iy_grid = {iy_grid}")
                continue  # Pula se os valores projetados estiverem fora do grid atual

            # Agora o `bresenham` recebe as coordenadas do grid
            pontos = bresenham(posicao_robô_x, posicao_robô_y, ix_grid, iy_grid)

            for (px, py) in pontos:
                if 0 <= px < self.grid_width and 0 <= py < self.grid_height:
                    self.occupancy_grid[py, px] = 0  # Marca a linha como livre

            if 0 <= ix_grid < self.grid_width and 0 <= iy_grid < self.grid_height:
                self.occupancy_grid[iy_grid, ix_grid] = 1  # Marca o ponto final como ocupado
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