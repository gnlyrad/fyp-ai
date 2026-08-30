FROM python:3.12-slim
WORKDIR /opt/program
COPY requirements.txt requirements-sagemaker.txt .
RUN pip install --no-cache-dir -r requirements-sagemaker.txt
COPY app ./app
COPY data ./data
COPY sagemaker ./sagemaker
ENV PYTHONPATH=/opt/program
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
