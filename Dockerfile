# start from an official Python base image
FROM python:3.13-slim

# set working directory inside the container
WORKDIR /app

# copy requirements first (for Docker layer caching)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY src/ ./src/
COPY models/ ./models/

# expose the port FastAPI runs on
EXPOSE 8000

# command to run the API
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]