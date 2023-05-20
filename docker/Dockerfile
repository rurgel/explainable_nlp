FROM python:3.9.16-slim

WORKDIR /app

COPY docker/docker_requirements.txt .

RUN pip install --no-cache-dir --upgrade -r docker_requirements.txt

COPY ../src ./src

CMD ["python", "-m", "src.app.summary_renderer"]