FROM python:3
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python model.py
CMD python main.py