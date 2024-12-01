import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


while True:

    # Carregar os dados do CSV
    data = pd.read_csv('/home/gustavo/turtlebot3_ws/src/ekf_slam/odometry_data.csv')

    # Separar os dados por fonte
    ground_truth = data[data['source'] == '/odom']
    ekf = data[data['source'] == '/ekf_odom']
    noisy_odom = data[data['source'] == '/noisy_odom']

    # Função para encontrar o índice do ponto mais próximo baseado no tempo
    def find_closest_time_index(ref_time, times):
        return np.argmin(np.abs(times - ref_time))

    # Garantir que o dataset com menos pontos seja a referência
    if len(ground_truth) > len(ekf):
        reference = ekf
        comparison = ground_truth
    else:
        reference = ground_truth
        comparison = ekf

    # Realizar a correspondência de tempos (EKF e Ground Truth)
    matched_indices = [find_closest_time_index(t, comparison['time'].values) for t in reference['time'].values]
    comparison_matched = comparison.iloc[matched_indices]

    # Calcular o erro médio entre EKF e Ground Truth
    ekf_errors = np.sqrt((reference['x'].values - comparison_matched['x'].values)**2 +
                        (reference['y'].values - comparison_matched['y'].values)**2)
    mean_ekf_error = ekf_errors.mean()

    # Realizar a correspondência de tempos (Noisy Odom e Ground Truth)
    matched_indices_noisy = [find_closest_time_index(t, noisy_odom['time'].values) for t in ground_truth['time'].values]
    noisy_odom_matched = noisy_odom.iloc[matched_indices_noisy]

    # Calcular o erro médio entre Noisy Odom e Ground Truth
    noisy_errors = np.sqrt((ground_truth['x'].values - noisy_odom_matched['x'].values)**2 +
                        (ground_truth['y'].values - noisy_odom_matched['y'].values)**2)
    mean_noisy_error = noisy_errors.mean()

    # Calcular o alcance do movimento para normalizar o erro
    ground_truth_range = np.sqrt((ground_truth['x'].max() - ground_truth['x'].min())**2 +
                                (ground_truth['y'].max() - ground_truth['y'].min())**2)

    relative_ekf_error = mean_ekf_error / ground_truth_range
    relative_noisy_error = mean_noisy_error / ground_truth_range

    # Exibir os resultados
    print(f"Erro médio do EKF em relação ao Ground Truth: {mean_ekf_error:.4f}")
    print(f"Erro médio do Noisy Odom em relação ao Ground Truth: {mean_noisy_error:.4f}")
    print(f"Erro relativo do EKF (em relação ao alcance do movimento): {relative_ekf_error:.4%}")
    print(f"Erro relativo do Noisy Odom (em relação ao alcance do movimento): {relative_noisy_error:.4%}")

    # Plotar os dados
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth['x'], ground_truth['y'], label="Ground Truth", linestyle="--", color="black")
    plt.plot(ekf['x'], ekf['y'], label="EKF", linestyle="-", color="blue")
    plt.plot(noisy_odom['x'], noisy_odom['y'], label="Noisy Odometry", linestyle="-", color="red")

    # Personalizar o gráfico
    plt.title("Comparação de Odometria")
    plt.xlabel("Posição X")
    plt.ylabel("Posição Y")
    plt.legend()
    plt.grid(True)

    # Exibir o gráfico
    plt.show()
    time.sleep(1)
