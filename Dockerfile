FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "post.py", "--gnews", "--interval", "600", "--days", "2"]