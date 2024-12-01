import matplotlib.pyplot as plt
import pandas as pd

# Carregar os dados do CSV
data = pd.read_csv('/home/gustavo/turtlebot3_ws/src/ekf_slam/ekf_slam/odometry_data.csv')

# Criar um gráfico para cada fonte
plt.figure(figsize=(10, 6))
for source in data['source'].unique():
    source_data = data[data['source'] == source]
    plt.plot(source_data['x'], source_data['y'], label=source)

# Personalizar o gráfico
plt.title("Comparação de Odometria")
plt.xlabel("Posição X")
plt.ylabel("Posição Y")
plt.legend()
plt.grid(True)

# Exibir o gráfico
plt.show()
