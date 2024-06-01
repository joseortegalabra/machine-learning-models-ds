### script con las librerías y la clase del modelo ###

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class lr_custom_alerts(LinearRegression):
    """
    Clase Regresión Lineal customizada que retorna la predicción y un grado de alerta
    """
    
    def train_custom_thresholds_alerts(self, y):
        '''
        Método auxiliar. "Entrenamiento" custom para obtener thresholds de alertas

        Args:
            y (array): valores y de entrenamiento para obtener thresholds

        Returns:
            None: No retorna nada. Guarda los valores obtenidos como parámetros de la clase
        '''
        # params
        percentil_bajo_medio = 0.5
        percentil_medio_alto = 0.9

        # ""train"" threshold
        self.min_value = np.round(y.min(), 2)
        self.max_value = np.round(y.max(), 2)
        self.threshold_1 = np.round(y.quantile(percentil_bajo_medio), 2)
        self.threshold_2 = np.round(y.quantile(percentil_medio_alto), 2)

    def get_interval_alert(self, y):
        """
        Método auxiliar. Obtener intervalo de alerta a partir del valor.

        Args:
            y (array list): array de numpy con los valores a calcularles el rango. Formato: (shape, )

        Returns:
            array: array de strings con la clasificación de la alerta. Formato (shape, )
        """
        alerts = pd.cut(
            x = y,
            bins = [self.min_value, self.threshold_1, self.threshold_2, self.max_value],
            labels = ['bajo', 'medio', 'alto']
        )
        return np.array(alerts)

    
    def fit(self, X, y):
        """
        Método train custom. Llamar al método train de la clase base y al método auxiliar para entrenar el intervalo de alerta
        """
        super().fit(X, y)
        self.train_custom_thresholds_alerts(y)
    
    
    def predict(self, X):
        """
        Método predict custom. Llama al método predict de la clase base y al método auxiliar para obtener el intervalo de alerta
        """
        y_pred = super().predict(X)
        y_alert = self.get_interval_alert(y_pred)
        
        return y_pred, y_alert
