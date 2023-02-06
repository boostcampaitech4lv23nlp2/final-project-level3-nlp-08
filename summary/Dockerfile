FROM python:3.8.5-slim-buster

COPY . /app 
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN pip install pip==21.2.4 && \
    pip install -r requirements.txt

CMD ["python", "summary_model.py"]
