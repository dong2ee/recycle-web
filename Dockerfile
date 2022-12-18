FROM python:3.8-buster

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED 1

ARG UID=1000
ARG GID=1000

RUN groupadd -g "${GID}" python \
  && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python
WORKDIR /home/python

RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

COPY --chown=python:python requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

USER python:python
ENV PATH="/home/${USER}/.local/bin:${PATH}"
RUN git lfs pull
COPY --chown=python:python . .

ARG FLASK_ENV

ENV FLASK_ENV=${FLASK_ENV}


EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
