# Use official Python slim image (smaller & faster)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first → better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "DecisionNode.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false"]