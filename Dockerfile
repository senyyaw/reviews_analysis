# The full API: collects, analyses and serves reviews (the read-only serverless deploy is app.py).
#   git lfs pull                      # the ONNX models are stored with Git LFS
#   docker build -t review-analysis .
#   docker run -p 8000:8000 review-analysis          # http://localhost:8000/docs
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra nlp --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra nlp

COPY data ./data
COPY models ./models
RUN for f in models/*/*.onnx; do \
      [ "$(stat -c %s "$f")" -gt 1000000 ] || { echo "$f is a Git LFS pointer: run 'git lfs pull'"; exit 1; }; \
    done

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["review-analysis", "serve", "--host", "0.0.0.0", "--port", "8000"]
