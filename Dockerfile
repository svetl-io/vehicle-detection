FROM python:3.8-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir opencv-python-headless imutils confluent-kafka numpy

EXPOSE 80

CMD ["python", "main.py"]
