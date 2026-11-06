FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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

EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
