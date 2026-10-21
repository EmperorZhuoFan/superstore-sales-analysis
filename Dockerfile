FROM python:3.13-slim

ARG APP_VERSION=1.0
ENV APP_ENV=production

RUN useradd --create-home appuser

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

RUN chown -R appuser:appuser /app

USER appuser

CMD ["python", "main.py"]