FROM python:3.9

WORKDIR /code

COPY ../requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ../src/app.py /code/app.py

CMD ["python", "app.py"]