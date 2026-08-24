FROM rapidsai/rapidsai:24.12-cuda12.5-runtime-ubuntu22.04-py3.12

WORKDIR /app

USER root
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

COPY pyproject.toml ./
COPY uv.lock* ./

RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8888

# Запускаем Jupyter Lab
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
