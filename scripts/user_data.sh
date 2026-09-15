#!/bin/bash
exec > /var/log/user-data.log 2>&1
echo "=== Starting EC2 UserData Streamlit Setup ==="

sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git

# Clone repo
git clone https://github.com/priyanshu-projects/Movie_Recommender.git /home/ubuntu/app
cd /home/ubuntu/app

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install boto3

# Download model artifacts and data from S3
mkdir -p /home/ubuntu/app/models/bert4rec
mkdir -p /home/ubuntu/app/data/raw/ml-32m

# Pull bert4rec model
aws s3 cp s3://movie-recommender-mlops-745600/models/bert4rec/bert4rec_candidate.pkl /home/ubuntu/app/models/bert4rec/bert4rec_candidate.pkl --region ap-south-1 || true

# Pull champion model metadata
aws s3 cp s3://movie-recommender-mlops-745600/models/champion/bert4rec_candidate.pkl /home/ubuntu/app/models/bert4rec/bert4rec_candidate.pkl --region ap-south-1 || true
aws s3 cp s3://movie-recommender-mlops-745600/models/champion/bert4rec_metrics.json /home/ubuntu/app/models/champion_meta.yaml --region ap-south-1 || true

# Pull movies vocab (54K movie titles for search)
aws s3 cp s3://movie-recommender-mlops-745600/data/movies_vocab.csv /home/ubuntu/app/data/raw/ml-32m/movies_vocab.csv --region ap-south-1 || true

# Configure AWS credentials for S3 model pull
mkdir -p /home/ubuntu/.aws
cat > /home/ubuntu/.aws/credentials << 'CREDSEOF'
[default]
aws_access_key_id = __AWS_ACCESS_KEY_ID__
aws_secret_access_key = __AWS_SECRET_ACCESS_KEY__
CREDSEOF

cat > /home/ubuntu/.aws/config << 'CFGEOF'
[default]
region = ap-south-1
output = json
CFGEOF

chown -R ubuntu:ubuntu /home/ubuntu/.aws /home/ubuntu/app

# Create systemd service for Streamlit
cat > /etc/systemd/system/streamlit.service << 'SVCEOF'
[Unit]
Description=Movie Recommender Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
Environment="PATH=/home/ubuntu/app/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/ubuntu/app/.venv/bin/streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable streamlit
systemctl start streamlit

echo "=== Setup complete! Streamlit active on port 8501 ==="
