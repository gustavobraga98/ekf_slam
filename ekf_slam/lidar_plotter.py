import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LidarPlotter(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        angle_increment = msg.angle_increment
        ranges = msg.ranges
        angles = np.arange(len(ranges)) * angle_increment
        # Converte os dados polares para coordenadas cartesianas
        points = np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles)])
        points = points[~np.isnan(points).any(axis=1)]
        points = points[~np.isinf(points).any(axis=1)]
        points = np.array([-points[:,1], points[:,0]]).T
        lines = self.ransac(points,degree_range=5)
        self.plot_lines_and_points(points, lines)


    def ransac(self,points, degree_range=5, consensus_theshold=30, max_iterations=1000, distance_threshold=0.02, delete_threshold=0.05):
        lines = []
        epoch = 0
        print(f"points: {points}")
        while len(points) > 0 and len(points) > consensus_theshold and epoch < max_iterations:
            # Selecionar um ponto aleatório do conjunto points
            random_index = randint(0,len(points)-1)
            random_point = points[random_index]

            # Calcular o ângulo do ponto aleatório
            random_point_angle = np.arctan2(random_point[1], random_point[0])
            
            # Calcular 5 graus positivos e 5 graus negativos
            lower_angle = random_point_angle - np.radians(degree_range)
            upper_angle = random_point_angle + np.radians(degree_range)
            
            # Obter todos os pontos dentro desse intervalo de ângulo
            angles = np.arctan2(points[:,1], points[:,0])
            mask = (angles >= lower_angle) & (angles <= upper_angle)
            points_in_range = points[mask]
            
            if len(points_in_range) >= 6:
                sample = []
                while len(sample) < 6:
                    random_index = randint(0, len(points_in_range)-1)
                    sample.append(points_in_range[random_index])
                    points_in_range = np.delete(points_in_range, random_index, axis=0)

                m, b = self.calculate_best_fit_line(np.array(sample))
                consensus = self.find_points_close_to_line(points, m, b, distance_threshold)
                
                if len(consensus) >= consensus_theshold:
                    print(f"salvando linha")
                    m, b = self.calculate_best_fit_line(consensus)
                    best_line = [m, b]
                    lines.append(best_line)
                    consensus = self.find_points_close_to_line(points, m, b, delete_threshold)
                    # Calcular os índices dos pontos de consenso no array points
                    consensus_indices = np.where(np.isin(points, consensus).all(axis=1))[0]
                    # Criar um novo array points que não inclui os pontos de consenso
                    points = np.delete(points, consensus_indices, axis=0)
                    print(f"Deletando {len(consensus)} de {len(points)} pontos")
                    
                else:
                    pass
            epoch +=1
        print(f"epochs: {epoch}")
        return lines
    
    def plot_lines_and_points(self,points, lines):
        # Plotar os pontos
        plt.scatter(points[:,0], points[:,1], label='Pontos')
        # Gerar um conjunto de valores x para plotar as linhas
        x_values = np.linspace(-3.5,3.5, 400)

        # Plotar cada linha
        for i, line in enumerate(lines):
            m, b = line
            y_values = (m * x_values + b)
            plt.plot(x_values,y_values, label=f'Linha {i+1}')
        
        # Adicionar um triângulo no ponto (0,0) com tamanho (0.5,0.5)
        triangle = patches.RegularPolygon((0, 0), numVertices=3, radius=0.2, orientation=0, color='red')
        plt.gca().add_patch(triangle)

        # Definir os limites dos eixos x e y
        plt.xlim(-3.5, 3.5)
        plt.ylim(-3.5, 3.5)

        plt.draw()
        plt.pause(0.00001)
        plt.clf()

    def calculate_best_fit_line(self, points):
        # Separar as coordenadas x e y
        x = points[:,0]
        y = points[:,1]

        # Use numpy's polyfit function with degree 1 to find the best fit line
        coefficients = np.polyfit(x, y, 1)

        # The coefficients are in the form [slope, y-intercept]
        slope, intercept = coefficients

        return slope, intercept


    def calculate_distance_to_line(self, points, m, b):
        # Calcular a distância de cada ponto à linha
        distances = np.abs(-m * points[:,0] + points[:,1] - b) / np.sqrt(m**2 + 1)
        return distances

    def find_points_close_to_line(self,points, m, b, distance_threshold):
        # Calcular a distância de cada ponto à linha
        distances = self.calculate_distance_to_line(points, m, b)
        
        # Encontrar os pontos que estão a uma distância menor ou igual ao limiar
        mask = distances <= distance_threshold
        points_close_to_line = points[mask]
        
        return points_close_to_line


def main(args=None):
    rclpy.init(args=args)

    lidar_plotter = LidarPlotter()

    rclpy.spin(lidar_plotter)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lidar_plotter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()