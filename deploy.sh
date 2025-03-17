name: Deploy to EC2 🚀
on:
  push:
    branches:
      - "main" # Trigger on push to main branch
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout current branch ✅
        uses: actions/checkout@v2

      - name: Set up SSH key and whitelist EC2 IP address 🐻‍❄️
        run: |
          mkdir -p ~/.ssh
          echo "${{ secrets.EC2_SSH_KEY }}" > ~/.ssh/id_rsa
          chmod 600 ~/.ssh/id_rsa
          ssh-keyscan ${{ secrets.EC2_HOST }} >> ~/.ssh/known_hosts

      - name: Create requirements.txt 📦
        run: |
          cat > requirements.txt << 'EOF'
streamlit==1.26.0
langchain==0.0.329
langchain_google_genai==0.0.5
langchain_community==0.0.10
faiss-cpu==1.7.4
python-dotenv==1.0.0
fastapi==0.103.1
uvicorn==0.23.2
gunicorn==21.2.0
flask==2.3.3
pydantic==2.3.0
EOF

      - name: Create .env file dynamically 🧨
        env:
          ENV: ${{ secrets.ENV }}
          EC2_USERNAME: ${{ secrets.EC2_USERNAME }}
          GOOGLE_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          echo "ENV=${ENV}" >> env
          echo "EC2_USERNAME=${EC2_USERNAME}" >> env
          echo "GOOGLE_API_KEY=${GOOGLE_API_KEY}" >> env

      - name: Create deploy.sh script 🚀
        run: |
          cat > deploy.sh << 'EOF'
#!/bin/bash
echo "=== Starting deployment process ==="
echo "Deleting old app"
sudo rm -rf /var/www/langchain-app
echo "Creating app folder"
sudo mkdir -p /var/www/langchain-app
echo "Moving files to app folder"
sudo cp -r * /var/www/langchain-app/
# Navigate to the app directory
cd /var/www/langchain-app/
# Ensure the .env file exists
if [ -f env ]; then
    sudo mv env .env
    echo ".env file created from env"
else
    echo "WARNING: env file not found, creating empty .env"
    touch .env
fi
# Update system packages
sudo apt-get update
echo "Installing python and pip"
sudo apt-get install -y python3 python3-pip python3-dev
# Install application dependencies from requirements.txt
echo "Installing application dependencies from requirements.txt"
if [ -f requirements.txt ]; then
    sudo pip3 install -r requirements.txt
else
    echo "WARNING: requirements.txt not found, installing essential packages"
    sudo pip3 install streamlit langchain langchain_google_genai langchain_community faiss-cpu python-dotenv fastapi uvicorn gunicorn pydantic
fi
# Create sample transcript if not exists
if [ ! -f cleaned_transcript.txt ]; then
    echo "Creating sample transcript file"
    echo "Welcome to today's lecture on Artificial Intelligence and Machine Learning." > cleaned_transcript.txt
    echo "This is a sample transcript file created during deployment." >> cleaned_transcript.txt
fi
# Update and install Nginx if not already installed
if ! command -v nginx > /dev/null; then
    echo "Installing Nginx"
    sudo apt-get update
    sudo apt-get install -y nginx
fi
# Configure Nginx to use HTTP instead of Unix socket
echo "Configuring Nginx for HTTP proxy"
sudo bash -c 'cat > /etc/nginx/sites-available/myapp <<EOF
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF'
# Enable the site
sudo ln -sf /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled
sudo rm -f /etc/nginx/sites-enabled/default
# Restart Nginx
echo "Restarting Nginx"
sudo systemctl restart nginx
# Stop any existing Gunicorn processes
echo "Stopping any existing Gunicorn processes"
sudo pkill gunicorn || true
# Start Gunicorn with HTTP binding (not Unix socket)
echo "Starting Gunicorn with HTTP binding"
cd /var/www/langchain-app/
nohup sudo gunicorn --workers 3 --bind 0.0.0.0:8000 app:api --timeout 120 > gunicorn.log 2>&1 &
# Give Gunicorn time to start
echo "Waiting for Gunicorn to start..."
sleep 5
# Verify the app is running
echo "Verifying application is running"
curl -s http://127.0.0.1:8000/ || echo "WARNING: Application is not responding on port 8000"
echo "=== Deployment complete ==="
echo "Check application logs at: /var/www/langchain-app/gunicorn.log"
EOF
          chmod +x deploy.sh

      - name: Copy files to remote server 🚙
        env:
          EC2_HOST: ${{ secrets.EC2_HOST }}
          EC2_USERNAME: ${{ secrets.EC2_USERNAME }}
        run: |
          scp -r app.py transcription.py requirements.txt env deploy.sh $EC2_USERNAME@$EC2_HOST:/home/ubuntu/

      - name: Run Bash Script To Deploy App 🚀
        env:
          EC2_HOST: ${{ secrets.EC2_HOST }}
          EC2_USERNAME: ${{ secrets.EC2_USERNAME }}
        run: |
          ssh -o StrictHostKeyChecking=no $EC2_USERNAME@$EC2_HOST "chmod +x ./deploy.sh && ./deploy.sh"

      - name: Clean up SSH key 🚀
        if: always()
        run: rm -f ~/.ssh/id_rsa