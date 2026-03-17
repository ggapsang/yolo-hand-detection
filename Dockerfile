FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/

EXPOSE 8911

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8911"]
