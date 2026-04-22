FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY embedding/ ./embedding/

RUN pip install --no-cache-dir uv && \
    uv pip install --system -e ".[st]"

VOLUME ["/root/.cache/retrieval_survey/models"]
VOLUME ["/data"]

ENV KMP_DUPLICATE_LIB_OK=TRUE

ENTRYPOINT ["retrieval-embedding"]
