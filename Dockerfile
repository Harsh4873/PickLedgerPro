FROM python:3.11-slim

# Install system deps for Playwright Chromium
RUN apt-get update && apt-get install -y \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
    libcairo2 libatspi2.0-0 libgtk-3-0 libx11-xcb1 \
    fonts-liberation wget curl --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and its Chromium browser binary
RUN pip install playwright && python -m playwright install chromium

# Copy app code
COPY . .

# Render injects PORT env var — our server already reads it
EXPOSE 8765

CMD ["python", "pickgrader_server.py"]
