import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

def plot_robot_movement(x_prev, y_prev, theta_prev, x, y, theta):
    # Configuração do plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(min(x_prev, x) - 1.5, max(x_prev, x) + 1.5)
    ax.set_ylim(min(y_prev, y) - 1.5, max(y_prev, y) + 1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title("Modelo de Movimento do Robô: Rotação-Translação-Rotação", fontsize=14, pad=20)
    
    # Cálculos
    dx, dy = x - x_prev, y - y_prev
    s = np.sqrt(dx**2 + dy**2)
    delta_theta1 = np.arctan2(dy, dx) - theta_prev
    delta_theta2 = theta - theta_prev - delta_theta1
    
    # --- Desenho do Robô ---
    def draw_robot(x, y, theta, color, label):
        # Corpo do robô (retângulo)
        robot_size = 0.3
        corners = [
            [x + robot_size*np.cos(theta), y + robot_size*np.sin(theta)],
            [x + robot_size*np.cos(theta + np.pi/2), y + robot_size*np.sin(theta + np.pi/2)],
            [x + robot_size*np.cos(theta + np.pi), y + robot_size*np.sin(theta + np.pi)],
            [x + robot_size*np.cos(theta - np.pi/2), y + robot_size*np.sin(theta - np.pi/2)]
        ]
        ax.add_patch(plt.Polygon(corners, color=color, alpha=0.4, label=label))
        
        # Cabeça do robô (seta)
        ax.add_patch(FancyArrowPatch(
            (x, y),
            (x + 0.5*np.cos(theta), y + 0.5*np.sin(theta)),
            color=color, arrowstyle='->', mutation_scale=20, linewidth=2))

    # --- Plotagem ---
    # 1. Posição inicial (t-1)
    draw_robot(x_prev, y_prev, theta_prev, 'blue', 'Posição inicial')
    ax.text(x_prev, y_prev-0.5, f'θ_prev = {np.rad2deg(theta_prev):.1f}°', 
            ha='center', color='blue')

    # 2. Rotação inicial (Δθ₁)
    arc_radius = 0.6
    arc1 = np.linspace(theta_prev, theta_prev + delta_theta1, 30)
    ax.plot(x_prev + arc_radius*np.cos(arc1), y_prev + arc_radius*np.sin(arc1), 
            'r-', linewidth=2, label=f'Δθ₁ = {np.rad2deg(delta_theta1):.1f}°')
    ax.text(x_prev + arc_radius*np.cos(theta_prev + delta_theta1/2), 
            y_prev + arc_radius*np.sin(theta_prev + delta_theta1/2),
            'Rotação\ninicial', color='red', ha='center')

    # 3. Translação (s)
    ax.plot([x_prev, x], [y_prev, y], 'g-', linewidth=3, 
            label=f'Translação (s = {s:.2f})')
    ax.text((x_prev+x)/2, (y_prev+y)/2 + 0.2, 'Movimento linear', 
            color='green', ha='center')

    # 4. Posição final (t)
    draw_robot(x, y, theta, 'red', 'Posição final')
    ax.text(x, y-0.5, f'θ = {np.rad2deg(theta):.1f}°', ha='center', color='red')

    # 5. Rotação final (Δθ₂)
    arc2 = np.linspace(np.arctan2(dy, dx), np.arctan2(dy, dx) + delta_theta2, 30)
    ax.plot(x + arc_radius*np.cos(arc2), y + arc_radius*np.sin(arc2), 
            'm-', linewidth=2, label=f'Δθ₂ = {np.rad2deg(delta_theta2):.1f}°')
    ax.text(x + arc_radius*np.cos(np.arctan2(dy, dx) + delta_theta2/2), 
            y + arc_radius*np.sin(np.arctan2(dy, dx) + delta_theta2/2),
            'Rotação\nfinal', color='purple', ha='center')

    # Elementos adicionais
    ax.set_xlabel('Eixo X', fontsize=12)
    ax.set_ylabel('Eixo Y', fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.show()

# Exemplo com θ inicial = 0 (olhando para o eixo X)
x_prev, y_prev = 0, 0
theta_prev = 0  # 0° (olhando para direita)
x, y = 3, 2
theta = np.deg2rad(0)  # -30° final

plot_robot_movement(x_prev, y_prev, theta_prev, x, y, theta)