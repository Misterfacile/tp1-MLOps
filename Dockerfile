FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Command to run the application
CMD ["fastapi", "dev", "app.py", "--host", "0.0.0.0"]