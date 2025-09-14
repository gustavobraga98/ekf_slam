import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
import message_filters
import os
import csv

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
        if not self.odom_data or not self.noisy_odom_data or not self.ekf_odom_data:
            return None

        gt = self.odom_data[-1]
        ekf = self.ekf_odom_data[-1]
        noisy = self.noisy_odom_data[-1]

        error_noisy = np.sqrt((gt[0] - noisy[0])**2 + (gt[1] - noisy[1])**2)
        error_ekf = np.sqrt((gt[0] - ekf[0])**2 + (gt[1] - ekf[1])**2)

        noisy_errors = [np.sqrt((g[0] - n[0])**2 + (g[1] - n[1])**2) for g, n in zip(self.odom_data, self.noisy_odom_data)]
        ekf_errors = [np.sqrt((g[0] - e[0])**2 + (g[1] - e[1])**2) for g, e in zip(self.odom_data, self.ekf_odom_data)]

        mae_noisy = np.mean(noisy_errors)
        mae_ekf = np.mean(ekf_errors)
        rmse_noisy = np.sqrt(np.mean(np.array(noisy_errors)**2))
        rmse_ekf = np.sqrt(np.mean(np.array(ekf_errors)**2))
        std_noisy = np.std(noisy_errors)
        std_ekf = np.std(ekf_errors)

        improvement = ((mae_noisy - mae_ekf) / mae_noisy) * 100 if mae_noisy > 0 else 0

        def calculate_total_distance(data):
            return sum(np.sqrt((data[i][0] - data[i-1][0])**2 + (data[i][1] - data[i-1][1])**2) for i in range(1, len(data)))

        distance_gt = calculate_total_distance(self.odom_data)
        distance_noisy = calculate_total_distance(self.noisy_odom_data)
        distance_ekf = calculate_total_distance(self.ekf_odom_data)

        metrics = {
            'error_noisy': error_noisy,
            'error_ekf': error_ekf,
            'mae_noisy': mae_noisy,
            'mae_ekf': mae_ekf,
            'rmse_noisy': rmse_noisy,
            'rmse_ekf': rmse_ekf,
            'std_noisy': std_noisy,
            'std_ekf': std_ekf,
            'improvement': improvement,
            'distance_gt': distance_gt,
            'distance_noisy': distance_noisy,
            'distance_ekf': distance_ekf
        }
        return metrics

    def plot_trajectories(self, save_path=None):
        if self.odom_data and self.noisy_odom_data and self.ekf_odom_data:
            odom_x, odom_y = zip(*self.odom_data)
            noisy_x, noisy_y = zip(*self.noisy_odom_data)
            ekf_x, ekf_y = zip(*self.ekf_odom_data)

            plt.figure()
            plt.plot(odom_x, odom_y, label='Trajetória de Referência')
            plt.plot(noisy_x, noisy_y, label='Odometria com Ruído')
            plt.plot(ekf_x, ekf_y, label='Trajetória Estimada (EKF)')
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.xlabel('Posição X (m)')
            plt.ylabel('Posição Y (m)')
            plt.title('Comparação de Trajetórias')
            plt.grid()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
        else:
            print("Não há dados suficientes para plotar as trajetórias.")

def get_next_run_number(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith('run') and os.path.isdir(os.path.join(base_dir, d))]
    
    run_numbers = []
    for run in existing_runs:
        try:
            run_numbers.append(int(run[3:]))
        except ValueError:
            continue

    if not run_numbers:
        return 1
    
    return max(run_numbers) + 1

def main(args=None):
    rclpy.init(args=args)
    node = OdomPlotter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nInterrupt received, saving data...")
        
        # 1. Determinar o número da próxima execução
        images_dir = 'images'
        run_number = get_next_run_number(images_dir)
        run_name = f"run{run_number}"
        save_dir = os.path.join(images_dir, run_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        image_path = os.path.join(save_dir, 'trajectory.png')

        # 2. Calcular métricas
        metrics = node.calculate_metrics()
        
        if metrics:
            # 3. Plotar e salvar trajetórias
            node.plot_trajectories(save_path=image_path)

            # 4. Salvar métricas no CSV
            csv_path = 'odometry_data.csv'
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, 'a', newline='') as csvfile:
                header = ['run', 'error_noisy', 'error_ekf', 'mae_noisy', 'mae_ekf', 'rmse_noisy', 'rmse_ekf', 'std_noisy', 'std_ekf', 'improvement', 'distance_gt', 'distance_noisy', 'distance_ekf']
                writer = csv.DictWriter(csvfile, fieldnames=header)

                if not file_exists:
                    writer.writeheader()
                
                row_data = {'run': run_name, **metrics}
                writer.writerow(row_data)
                print(f"Metrics saved to {csv_path}")
        else:
            print("No metrics to save.")

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
