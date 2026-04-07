FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables for HF Space
# These will be overridden by HF Space secrets if set
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
ENV OPENENV_URL="http://localhost:8000"

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]