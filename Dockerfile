FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY nexo ./nexo
RUN pip install --no-cache-dir .
COPY . .
ENV NEXO_MODEL=/models/nexo
EXPOSE 8000
CMD ["nexo-serve", "--host", "0.0.0.0", "--port", "8000"]
