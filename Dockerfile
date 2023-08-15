FROM nvcr.io/nvidia/tritonserver:23.07-py3



COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /opt/tritonserver/
