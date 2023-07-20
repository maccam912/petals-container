FROM python:3.10
WORKDIR /
RUN git clone https://github.com/borzunov/chat.petals.ml.git
WORKDIR /chat.petals.ml
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
CMD gunicorn app:app --bind 0.0.0.0:5000 --worker-class gthread --threads 100 --timeout 1000
