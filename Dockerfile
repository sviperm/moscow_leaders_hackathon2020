FROM python:3.8

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt .

RUN pip install -U pip && \
    pip install -U wheel && \
    pip install -r requirements.txt

COPY . .

RUN apt-get update && \
    apt-get install -y libinsighttoolkit4-dev

EXPOSE 8080

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8080", "--allow-root", "--no-browser"]
