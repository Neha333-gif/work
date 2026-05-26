FROM python:3.12-slim

WORKDIR /app

COPY email_agent/ /app/

RUN pip install --no-cache-dir flask flask-sqlalchemy flask-login groq

EXPOSE 5050

CMD ["python", "app.py"]