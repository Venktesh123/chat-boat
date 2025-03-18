#!/bin/bash

APP_DIR="/var/www/chatbot"  # Update this with your actual project directory
VENV_DIR="$APP_DIR/venv"    # Virtual environment directory
SERVICE_NAME="chatbot"      # Systemd service name

# Navigate to the application directory
cd $APP_DIR || exit

# Pull the latest changes
echo "Pulling latest changes from Git..."
git pull origin main

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Restart the application using systemd
echo "Restarting the application service..."
sudo systemctl restart $SERVICE_NAME

echo "Deployment completed successfully!"
