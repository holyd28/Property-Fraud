#!/bin/bash

echo "Setting up Anomaly Detection Environment..."

# building Docker image
docker build -t property_fraud .

echo "Running Anomaly Detection..."

docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/results:/app/results" \
    property_fraud
