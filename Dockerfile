FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY adapters/ ./adapters/
COPY environments/ ./environments/
COPY results/dqn_v2/model.pth ./results/dqn_v2/model.pth
COPY results/qlearning_improved_v2/q_table.npy ./results/qlearning_improved_v2/q_table.npy

EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]