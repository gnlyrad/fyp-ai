FROM python:3.12-slim
WORKDIR /app

COPY langchain-backend ./langchain-backend
COPY langgraph-backend ./langgraph-backend

RUN cd langgraph-backend && pip install --no-cache-dir -r requirements.txt

COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

EXPOSE 2024 8001
ENTRYPOINT ["./docker-entrypoint.sh"]
