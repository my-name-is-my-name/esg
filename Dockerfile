FROM mineru:latest

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app

EXPOSE 8132
CMD ["sh", "-c", "umask 000; exec python3 -m esg.api serve"]
