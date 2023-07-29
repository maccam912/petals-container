FROM python:3.10
WORKDIR /app
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -U litestar uvicorn pydantic transformers websockets "petals>=2"
COPY startup.sh .
COPY server.py .
EXPOSE 8080
CMD bash startup.sh
