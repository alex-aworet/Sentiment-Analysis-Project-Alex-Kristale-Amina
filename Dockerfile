FROM python:3.11
WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "src/inference.py"]
CMD []
