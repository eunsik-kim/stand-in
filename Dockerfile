FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
ENV HOST=0.0.0.0 PORT=8010
EXPOSE 8010
CMD ["uv", "run", "--no-sync", "server.py"]
