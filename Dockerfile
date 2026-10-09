FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Package files first: this layer is rebuilt only when dependencies or source change
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY tests ./tests
COPY scripts ./scripts

CMD ["pytest", "-q"]
