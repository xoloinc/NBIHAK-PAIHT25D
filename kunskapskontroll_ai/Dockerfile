FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgles2 \
    libegl1 \
    libvulkan1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_stream.py .
COPY asl_modell.pkl .
COPY asl_scaler.pkl .
COPY asl_label_encoder.pkl .
COPY hand_landmarker.task .
COPY asl_alphabet_test/ ./asl_alphabet_test/

EXPOSE 8501

CMD ["streamlit", "run", "app_stream.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
