FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app

COPY ["requirements.txt", "starter.py", "./" ]

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "starter.py", "2022", "4" ]