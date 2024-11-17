import numpy as np

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