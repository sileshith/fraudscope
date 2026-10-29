FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /app
COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock
COPY fraudscope ./fraudscope
RUN pip install --no-cache-dir --no-deps . && useradd --uid 10001 --create-home appuser
COPY reports/demo/seed_42/model.joblib reports/demo/seed_42/model.json ./models/
RUN mkdir /app/runtime && chown appuser:appuser /app/runtime
USER 10001
ENV FRAUDSCOPE_MODEL=/app/models/model.joblib FRAUDSCOPE_DB=/app/runtime/fraudscope.sqlite
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2)"
CMD ["python","-m","uvicorn","fraudscope.service:create_app","--factory","--host","0.0.0.0","--port","8000","--workers","1"]

