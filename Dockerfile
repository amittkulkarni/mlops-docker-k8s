# Dockerfile
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run the training script to generate the model.pkl file
# This is part of the image build process
RUN python train.py

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the application using uvicorn
# The API will be available on http://0.0.0.0:8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]