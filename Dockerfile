FROM python:3.10
WORKDIR /app
RUN pip install -U litestar uvicorn pydantic transformers websockets "petals>=2"
COPY startup.sh .
COPY server.py .
CMD bash startup.sh
