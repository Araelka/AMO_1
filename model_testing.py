import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from joblib import load
import numpy as np

# Функция для чтения данных из папки
def load_data(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            data = pd.read_csv(filepath)
            all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

# Главная функция
def main():
    # Загружаем предобработанные данные
    data = load_data('test_processed')

    # Извлекаем признаки и целевую переменную
    X = data[['temperature']].values
    y_true = np.sin(data['temperature'].values)  # Пример использования синусоиды для моделирования целевой переменной

    # Загружаем обученную модель
    model = load('trained_model.joblib')

    # Прогнозируем на тестовых данных
    y_pred = model.predict(X)

    # Оцениваем качество модели
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Squared Error on test data: {mse:.4f}')

if __name__ == "__main__":
    main()