FROM python:3.13-slim AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1


FROM base AS python-deps

RUN pip install pipenv
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY Pipfile .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install


FROM base AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

RUN useradd --create-home appuser \
    && mkdir -p /home/appuser/data /home/appuser/.index_cache \
    && chown -R appuser:appuser /home/appuser

WORKDIR /home/appuser
USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["python", "-m", "streamlit"]
CMD ["run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
