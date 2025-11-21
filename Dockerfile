FROM python:3.12-slim

WORKDIR /app

# Install only runtime deps; no mlflow container helper, no nginx
RUN pip install --no-cache-dir \
    mlflow==3.4.0 \
    numpy==2.1.3 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    xgboost==3.0.5 \
    fastapi==0.115.6 \
    uvicorn[standard]==0.34.0

# Copy the MLflow model into the image
COPY wine-model-2 /opt/ml/model

# Copy the FastAPI server
COPY server.py /app/server.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

