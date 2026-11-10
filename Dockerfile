# Stage 1: build the demo data. Needs torch (train group) to train the demo bundle.
FROM python:3.13-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv
ENV UV_NO_CACHE=1

RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user . .
RUN uv sync --frozen --no-dev

ENV PATH=/home/user/app/.venv/bin:$PATH

# Sapienza labels only (hu_pd_logs.tsv is never in the image), then train the demo bundle.
RUN idsai labels && idsai demo-pack && python deploy/check_no_hu.py data

# Stage 2: runtime without torch.
FROM python:3.13-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv
ENV UV_NO_CACHE=1

RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-group train --no-install-project

COPY --chown=user . .
RUN uv sync --frozen --no-dev --no-group train

RUN mkdir -p data
COPY --from=build --chown=user /home/user/app/data /home/user/app/data
RUN test -w data && test -w data/app.duckdb

ENV PATH=/home/user/app/.venv/bin:$PATH
# 0.1 CPU box: one thread per pool avoids throttling; results unchanged.
ENV POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
