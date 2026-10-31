FROM python:3.11-slim

WORKDIR /app

RUN python --version && pip --version

CMD ["python"]