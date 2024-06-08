#!/bin/bash

# Устанавливаем права на исполнение скриптов
chmod +x data_creation.py
chmod +x model_preprocessing.py
chmod +x model_preparation.py
chmod +x model_testing.py

echo "1. Генерация данных..."
python data_creation.py

echo "2. Предобработка данных..."
python model_preprocessing.py

echo "3. Обучение модели..."
python model_preparation.py

echo "4. Тестирование модели..."
python model_testing.py

echo "Пайплайн выполнен успешно!"