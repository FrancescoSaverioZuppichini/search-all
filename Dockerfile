FROM nvcr.io/nvidia/pytorch:23.06-py3

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
COPY . /workspace

EXPOSE 7860

CMD [ "python", "app.py" ]