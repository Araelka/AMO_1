import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Функция для чтения и предобработки данных
def preprocess_data(input_directory, output_directory):
    # Создаем папку для предобработанных данных, если она не существует
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Создаем экземпляр StandardScaler
    scaler = StandardScaler()

    # Перебираем все файлы в директории
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            # Читаем данные
            filepath = os.path.join(input_directory, filename)
            data = pd.read_csv(filepath)

            # Проверяем наличие столбца temperature
            if 'temperature' in data.columns:
                # Применяем StandardScaler к столбцу temperature
                data['temperature'] = scaler.fit_transform(data[['temperature']])

                # Сохраняем предобработанные данные в новую директорию
                output_filepath = os.path.join(output_directory, filename)
                data.to_csv(output_filepath, index=False)

# Главная функция
def main():
    # Предобработка данных в папке train
    preprocess_data('train', 'train_processed')

    # Предобработка данных в папке test
    preprocess_data('test', 'test_processed')

if __name__ == "__main__":
    main()