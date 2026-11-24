FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY . .

RUN uv sync --frozen --no-dev --extra ml

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "detector_neumonia.py"]
