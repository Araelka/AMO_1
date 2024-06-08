import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load
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

# Функция для обучения модели
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Главная функция
def main():
    # Загружаем предобработанные данные
    data = load_data('train_processed')

    # Извлекаем признаки и целевую переменную
    X = data[['temperature']].values
    y = np.sin(data['temperature'].values)  # Пример использования синусоиды для моделирования целевой переменной

    # Обучаем модель
    model = train_model(X, y)

    # Сохраняем модель
    dump(model, 'trained_model.joblib')

    # Оценка модели на тренировочных данных
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error on training data: {mse:.4f}')

if __name__ == "__main__":
    main()