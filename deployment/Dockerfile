# Start from a base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code into the container
COPY ["model.joblib", "stroke_api.py", "./"] .

# Expose the app port
EXPOSE 80

# Run command
CMD ["uvicorn", "stroke_api:app", "--host", "0.0.0.0", "--port", "80"]