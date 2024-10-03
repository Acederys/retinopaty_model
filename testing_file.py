import os
import io
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from sklearn.utils import resample

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Убедитесь, что MLflow запущен
mlflow.set_experiment("New Retinopathy Classification Experiment")

# Загрузка модели
model_path = '/home/acederys/retinopaty_model/last.pt'
model = YOLO(model_path)


# Путь к папке с тестовыми изображениями
test_path = '/home/acederys/retinopaty_model/data/test'

# Список для хранения результатов
results_list = []

# Словарь для хранения соответствия меток классов
class_names = {0: 'healthy', 1: 'unhealthy'}

# Начало нового трека в MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "YOLO")
    # Перебор всех файлов в папке и подкаталогах
    for root, dirs, files in os.walk(test_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)

                # Выполнение инференса
                results = model(img_path)  # Возвращает список объектов Results

                for result in results:
                    # Извлечение данных
                    if result.probs is not None:
                        top_class = result.probs.top1  # Индекс класса с наибольшей вероятностью
                        confidence = result.probs.top1conf.item()  # Уверенность для этого класса

                        # Определение истинного класса на основе структуры папок
                        if 'unhealthy' in os.path.basename(root):
                            true_class = 1
                        elif 'healthy' in os.path.basename(root):
                            true_class = 0

                        results_list.append({
                            'Image': file,
                            'TrueClass': true_class,
                            'PredClass': top_class,
                            'Confidence': confidence
                        })
                        print(f"Файл: {file}, Истинный класс: {true_class}, Предсказанный класс: {top_class}, Уверенность: {confidence}")

    # Создание DataFrame из результатов
    results_df = pd.DataFrame(results_list)
    results_df

    # Подсчет метрик
    precision = precision_score(results_df['TrueClass'], results_df['PredClass'])
    recall = recall_score(results_df['TrueClass'], results_df['PredClass'])
    accuracy = accuracy_score(results_df['TrueClass'], results_df['PredClass'])

    # Логирование метрик в MLflow
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("accuracy", accuracy)

    # Вывод метрик
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

    # Построение матрицы путаницы
    conf_matrix = confusion_matrix(results_df['TrueClass'], results_df['PredClass'])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[class_names[0], class_names[1]])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    # Сохранение изображения в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Перемещение курсора в начало буфера

    # Преобразование буфера в объект PIL Image
    image = Image.open(buf)

    # Логирование изображения напрямую в MLflow
    mlflow.log_image(image, "confusion_matrix.png")


    # Сохранение результатов в CSV
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'results.csv')
    results_df.to_csv(csv_path, index=False)

    # Логирование CSV файла как артефакт
    mlflow.log_artifact(csv_path)

    # Вывод результата
    print(f'Результаты сохранены в {csv_path}')

    # Загрузка данных из CSV файла
    df = pd.read_csv(csv_path)

    # Выбор столбца с вероятностями для одного эксперимента (например, Pred_Exp_1)
    prob_column = 'PredClass'
    probs = df[prob_column].values

    # Определение функции для вычисления бутстраппинг-доверительных интервалов
    def bootstrap_ci(data, num_iterations=1000, ci=0.95):
        """ Compute the confidence interval for the mean using bootstrap resampling. """
        n = len(data)
        means = np.zeros(num_iterations)

        for i in range(num_iterations):
            sample = resample(data, n_samples=n, replace=True)
            means[i] = np.mean(sample)

        lower_bound = np.percentile(means, (1 - ci) / 2 * 100)
        upper_bound = np.percentile(means, (1 + ci) / 2 * 100)

        return lower_bound, upper_bound

    # Рассчитываем доверительные интервалы для вероятностей
    lower, upper = bootstrap_ci(probs)

    # Логирование доверительных интервалов в MLflow
    mlflow.log_metric("bootstrap_ci_lower", lower)
    mlflow.log_metric("bootstrap_ci_upper", upper)

    # Вывод результатов
    print(f'95% доверительный интервал для вероятностей {prob_column}: [{lower:.4f}, {upper:.4f}]')
