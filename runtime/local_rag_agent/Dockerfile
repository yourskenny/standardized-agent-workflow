FROM python:3.11-slim

WORKDIR /app

COPY . /app
RUN python -m pip install --no-cache-dir .

EXPOSE 8765

CMD ["sh", "-c", "python -m local_rag_agent serve --project ${LOCAL_AGENT_PROJECT:-/project} --config ${LOCAL_AGENT_CONFIG:-/project/runtime.toml} --port ${LOCAL_AGENT_PORT:-8765}"]
