FROM python:3.13.15-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS builder

ARG PIP_VERSION=26.2.1
ARG PIPENV_VERSION=2026.8.0

RUN python -m pip install --no-cache-dir --disable-pip-version-check "pip==${PIP_VERSION}" "pipenv==${PIPENV_VERSION}"
RUN python -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIPENV_IGNORE_VIRTUALENVS=0
ENV PIPENV_NOSPIN=1

COPY Pipfile Pipfile.lock /tmp/docmind/
WORKDIR /tmp/docmind

RUN --mount=type=cache,target=/root/.cache/pip \
    if /opt/venv/bin/python -m pip install --help | grep -q -- "--no-deprecated"; then \
        pipenv sync --categories default --extra-pip-args="--no-deprecated --require-hashes"; \
    else \
        pipenv sync --categories default --extra-pip-args="--require-hashes"; \
    fi \
    && /opt/venv/bin/python -m pip check

FROM base AS runtime

ARG GIT_VERSION=1:2.39.5-0+deb12u3
ARG APP_UID=10001
ARG APP_GID=10001

RUN apt-get update \
    && apt-get install -y --no-install-recommends "git=${GIT_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --home-dir /home/appuser --shell /usr/sbin/nologin appuser \
    && install -d -o "${APP_UID}" -g "${APP_GID}" /home/appuser/data /home/appuser/.index_cache /home/appuser/.cache /home/appuser/logs \
    && install -d -o 0 -g 0 /home/appuser/.docmind \
    && chown root:root /home/appuser

COPY --from=builder /opt/venv /opt/venv

ENV HOME=/home/appuser
ENV XDG_CACHE_HOME=/home/appuser/.cache
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV DOCMIND_LOG_FILE=/home/appuser/logs/docmind.log
ENV DOCMIND_INGESTION_LOCK_PATH=/home/appuser/data/.docmind-ingestion.lock
ENV DOCMIND_R2R_STATE_DIR=/home/appuser/data/r2r

WORKDIR /home/appuser

COPY main.py ./
COPY components/ ./components/
COPY utils/ ./utils/
COPY .streamlit/ ./.streamlit/

USER 10001:10001

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["python", "-m", "streamlit"]
CMD ["run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
