FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
COPY . /home
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6
RUN pip install -r /home/requirements.txt
WORKDIR /home
CMD [ "streamlit", "run", "app.py" ]