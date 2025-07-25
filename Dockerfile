# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Pre-install all packages and cache wheels
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Stage 2: Minimal runtime
FROM python:3.10-slim

WORKDIR /app

# Copy only what’s needed
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Copy all code and models
COPY . .

# Create output directory
RUN mkdir -p output

# Inference entrypoint
CMD ["python", "outline_extractor.py"]
