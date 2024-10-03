FROM python:3.8-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем все файлы проекта
COPY . /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для MLflow
EXPOSE 5000

# Запускаем MLflow Tracking Server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "s3://retinopatyml/b1gbqcgk1bfajrkrfdfr/mlflow/"]
