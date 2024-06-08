import numpy as np
import pandas as pd
import os

# Функция для создания папок, если они не существуют
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Функция для генерации данных
def generate_data(days=365, noise=False, anomalies=False):
    np.random.seed(42)  # Фиксируем генерацию случайных чисел для воспроизводимости
    base_temperature = np.sin(np.linspace(0, 2 * np.pi, days)) * 10 + 20  # Основной тренд температуры

    if noise:
        base_temperature += np.random.normal(0, 1, days)  # Добавление шума

    if anomalies:
        anomaly_indices = np.random.choice(days, size=5, replace=False)
        base_temperature[anomaly_indices] += np.random.choice([-10, 10], size=5)  # Добавление аномалий

    dates = pd.date_range(start="2024-01-01", periods=days)
    return pd.DataFrame({'date': dates, 'temperature': base_temperature})

# Функция для сохранения данных в CSV
def save_data(data, filename):
    data.to_csv(filename, index=False)

# Главная функция для создания и сохранения данных
def main():
    # Создаем папки train и test
    create_directory('train')
    create_directory('test')

    # Генерация и сохранение данных
    for i in range(3):  # Три различных набора данных для train
        data = generate_data(noise=(i==1), anomalies=(i==2))
        save_data(data, f'train/train_data_{i}.csv')

    for i in range(2):  # Два различных набора данных для test
        data = generate_data(noise=(i==1), anomalies=(i==1))
        save_data(data, f'test/test_data_{i}.csv')

if __name__ == "__main__":
    main()