FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

#  FORCE TRAIN
RUN python train.py

EXPOSE 7860

#  MUST BE THIS

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
