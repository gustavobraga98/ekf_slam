import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
import message_filters

class OdomPlotter(Node):
    def __init__(self):
        super().__init__('odom_plotter')
        self.odom_data = []
        self.noisy_odom_data = []
        self.ekf_odom_data = []

        # Criando os assinantes
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        self.noisy_odom_sub = message_filters.Subscriber(self, Odometry, '/noisy_odom')
        self.ekf_odom_sub = message_filters.Subscriber(self, Odometry, '/ekf_odom')

        # Sincronizador com uma janela de tolerância de 0.5 segundos
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.noisy_odom_sub, self.ekf_odom_sub], 
            queue_size=10, 
            slop=0.5)  # slop define a tolerância em segundos

        # Conectar o callback de sincronização
        self.ts.registerCallback(self.odom_callback)

    def odom_callback(self, odom_msg, noisy_odom_msg, ekf_odom_msg):
        # Salvar os dados de cada tópico ao mesmo tempo
        self.odom_data.append((odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y))
        self.noisy_odom_data.append((noisy_odom_msg.pose.pose.position.x, noisy_odom_msg.pose.pose.position.y))
        self.ekf_odom_data.append((ekf_odom_msg.pose.pose.position.x, ekf_odom_msg.pose.pose.position.y))
        self.get_logger().info("Odometrias recebidas")
    def calculate_metrics(self):
        if self.odom_data and self.noisy_odom_data and self.ekf_odom_data:
            # Usar o último elemento de cada vetor
            gt = self.odom_data[-1]  # Posição verdadeira
            ekf = self.ekf_odom_data[-1]  # Estimativa corrigida
            noisy = self.noisy_odom_data[-1]  # Posição com ruído

            # Calcular erros instantâneos
            error_noisy = np.sqrt((gt[0] - noisy[0])**2 + (gt[1] - noisy[1])**2)
            error_ekf = np.sqrt((gt[0] - ekf[0])**2 + (gt[1] - ekf[1])**2)

            # Calcular MAE acumulado
            noisy_errors = [np.sqrt((gt[0] - noisy[0])**2 + (gt[1] - noisy[1])**2)
                            for gt, noisy in zip(self.odom_data, self.noisy_odom_data)]
            ekf_errors = [np.sqrt((gt[0] - ekf[0])**2 + (gt[1] - ekf[1])**2)
                            for gt, ekf in zip(self.odom_data, self.ekf_odom_data)]

            mae_noisy = np.mean(noisy_errors)
            mae_ekf = np.mean(ekf_errors)

            # Calcular RMSE acumulado
            rmse_noisy = np.sqrt(np.mean(np.array(noisy_errors)**2))
            rmse_ekf = np.sqrt(np.mean(np.array(ekf_errors)**2))

            # Calcular desvio padrão dos erros
            std_noisy = np.std(noisy_errors)
            std_ekf = np.std(ekf_errors)

            # Calcular desempenho relativo do EKF
            improvement = ((mae_noisy - mae_ekf) / mae_noisy) * 100

            # Calcular distância total percorrida
            def calculate_total_distance(data):
                return sum(
                    np.sqrt((data[i][0] - data[i-1][0])**2 + (data[i][1] - data[i-1][1])**2)
                    for i in range(1, len(data))
                )

            distance_gt = calculate_total_distance(self.odom_data)
            distance_noisy = calculate_total_distance(self.noisy_odom_data)
            distance_ekf = calculate_total_distance(self.ekf_odom_data)

            # Exibir métricas
            print(f"Len of all data: GT: {len(self.odom_data)}, Noisy: {len(self.noisy_odom_data)}, EKF: {len(self.ekf_odom_data)}")
            print(f"Last position of all data: GT: {gt}, Noisy: {noisy}, EKF: {ekf}")
            print(f"Erro último ponto (Simulada): {error_noisy:.4f} m")
            print(f"Erro último ponto (EKF): {error_ekf:.4f} m")
            print(f"MAE (Simulada): {mae_noisy:.4f} m, MAE (EKF): {mae_ekf:.4f} m")
            print(f"RMSE (Simulada): {rmse_noisy:.4f} m, RMSE (EKF): {rmse_ekf:.4f} m")
            print(f"Desvio padrão (Simulada): {std_noisy:.4f} m, Desvio padrão (EKF): {std_ekf:.4f} m")
            print(f"Desempenho relativo do EKF: {improvement:.2f}%")
            print(f"Distância total percorrida (GT): {distance_gt:.2f} m")
            print(f"Distância total percorrida (Simulada): {distance_noisy:.2f} m")
            print(f"Distância total percorrida (EKF): {distance_ekf:.2f} m")

    def plot_trajectories(self):
        if self.odom_data and self.noisy_odom_data and self.ekf_odom_data:
            odom_x, odom_y = zip(*self.odom_data)
            noisy_x, noisy_y = zip(*self.noisy_odom_data)
            ekf_x, ekf_y = zip(*self.ekf_odom_data)

            plt.figure()
            plt.plot(odom_x, odom_y, label='Posição Verdadeira')
            plt.plot(noisy_x, noisy_y, label='Odômetria Simulada')
            plt.plot(ekf_x, ekf_y, label='Posição Estimada (EKF)')
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.xlabel('Posição X (m)')
            plt.ylabel('Posição Y (m)')
            plt.title('Comparação de Trajetórias')
            plt.grid()
            plt.tight_layout()
            plt.show()
        else:
            print("Não há dados suficientes para plotar as trajetórias.")

def main(args=None):
    rclpy.init(args=args)
    node = OdomPlotter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.calculate_metrics()
        node.plot_trajectories()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
