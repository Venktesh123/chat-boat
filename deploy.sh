#!/bin/bash
# Deployment script for the Chatbot API
# This script sets up the environment and deploys the chatbot service

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Starting deployment process ==="

# Define the application directory
APP_DIR=${APP_DIR:-$(pwd)}

echo "Application directory: $APP_DIR"
cd $APP_DIR

# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Installing Python..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip python3-dev
fi

# Create or update virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install or update dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt

# Make sure specific dependencies are installed
echo "Ensuring all dependencies are installed..."
pip install flask-cors
pip install sentence-transformers

# Check for transcript files
echo "Checking for required files..."
if [ ! -f "transcript.txt" ]; then
    echo "Warning: transcript.txt not found. Creating a placeholder..."
    echo "This is a placeholder for the transcript file." > transcript.txt
fi

if [ ! -f "cleaned_transcript.txt" ]; then
    echo "Warning: cleaned_transcript.txt not found. Using transcript.txt as a fallback..."
    cp transcript.txt cleaned_transcript.txt
fi

if [ ! -f "course_outcomes.txt" ]; then
    echo "Warning: course_outcomes.txt not found. Creating a placeholder..."
    echo "CO1: Understand basic principles of the subject" > course_outcomes.txt
    echo "CO2: Apply key concepts in problem-solving scenarios" >> course_outcomes.txt
    echo "CO3: Analyze complex situations and develop solutions" >> course_outcomes.txt
fi

# Check if the .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating a placeholder..."
    echo "# Google API Keys for Gemini" > .env
    echo "GOOGLE_API_KEY_1=your_api_key_here" >> .env
    echo "GOOGLE_API_KEY_2=your_api_key_here" >> .env
    echo "GOOGLE_API_KEY_3=your_api_key_here" >> .env
    echo "GOOGLE_API_KEY_4=your_api_key_here" >> .env
fi

# Set up the systemd service file
echo "Setting up systemd service..."
cat > /tmp/chatbot.service << EOF
[Unit]
Description=Chatbot AI API Service
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/api.py
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Move and enable the service
sudo mv /tmp/chatbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable chatbot
sudo systemctl restart chatbot

# Set up Nginx if not already configured
if ! command -v nginx &>/dev/null; then
    echo "Installing Nginx..."
    sudo apt-get update
    sudo apt-get install -y nginx
fi

# Configure Nginx for HTTPS (if certificates exist)
if [ -f "/etc/ssl/certs/nginx-selfsigned.crt" ] && [ -f "/etc/ssl/private/nginx-selfsigned.key" ]; then
    echo "Setting up Nginx with HTTPS..."
    sudo bash -c 'cat > /etc/nginx/sites-available/chatbot << EOF
server {
    listen 443 ssl;
    server_name $(curl -s http://checkip.amazonaws.com);
    ssl_certificate /etc/ssl/certs/nginx-selfsigned.crt;
    ssl_certificate_key /etc/ssl/private/nginx-selfsigned.key;

    location / {
        proxy_pass http://127.0.0.1:4000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}

server {
    listen 80;
    server_name $(curl -s http://checkip.amazonaws.com);
    return 301 https://\$host\$request_uri;
}
EOF'
else
    echo "Setting up Nginx with HTTP only..."
    sudo bash -c 'cat > /etc/nginx/sites-available/chatbot << EOF
server {
    listen 80;
    server_name $(curl -s http://checkip.amazonaws.com);

    location / {
        proxy_pass http://127.0.0.1:4000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF'
fi

# Enable the site
sudo ln -sf /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
echo "Testing Nginx configuration..."
sudo nginx -t

# Restart Nginx
echo "Restarting Nginx..."
sudo systemctl restart nginx

# Check if the services are running
echo "Checking service status..."
sudo systemctl status chatbot --no-pager
sudo systemctl status nginx --no-pager

echo ""
echo "=== Deployment complete! ==="
IP_ADDRESS=$(curl -s http://checkip.amazonaws.com)
echo "Your chatbot API should now be accessible at:"
echo "HTTP: http://$IP_ADDRESS"
if [ -f "/etc/ssl/certs/nginx-selfsigned.crt" ] && [ -f "/etc/ssl/private/nginx-selfsigned.key" ]; then
    echo "HTTPS: https://$IP_ADDRESS"
fi
echo ""
echo "You can test the chatbot with:"
echo "curl -X POST -H \"Content-Type: application/json\" -d '{\"query\":\"What is concrete?\"}' http://localhost:4000/ask"
echo ""
echo "You can test the question generator with:"
echo "curl -X POST -H \"Content-Type: application/json\" -d '{\"course_outcome\":\"CO1: Understand basic principles\",\"bloom_level\":\"Understand\"}' http://localhost:4000/generate-questions"
echo ""