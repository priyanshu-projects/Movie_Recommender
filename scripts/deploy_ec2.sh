#!/usr/bin/env bash
# scripts/deploy_ec2.sh
# Automated script to launch a t3.micro EC2 instance and host the Movie Recommender Streamlit app.

set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
BUCKET="${AWS_S3_BUCKET:-movie-recommender-mlops-745600}"

echo "=== Movie Recommender AWS EC2 Deployment ==="
echo "Region: $REGION"
echo "Bucket: $BUCKET"

source /home/priyanshu/project/.venv/bin/activate
export PATH="/home/priyanshu/project/.venv/bin:$PATH"

# Check AWS CLI credentials
aws sts get-caller-identity > /dev/null || {
    echo "❌ AWS CLI authentication failed. Configure credentials first."
    exit 1
}

# 1. Create Security Group if not exists
SG_NAME="movie-recommender-sg"
SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" --region "$REGION" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "")

if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
    echo "Creating Security Group: $SG_NAME..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Security group for Movie Recommender Streamlit App" \
        --region "$REGION" \
        --query "GroupId" --output text)
    
    # Authorize SSH (22) and Streamlit (8501)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 --region "$REGION"
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8501 --cidr 0.0.0.0/0 --region "$REGION"
    echo "✓ Security Group created: $SG_ID"
else
    echo "✓ Security Group exists: $SG_ID"
fi

# 2. Get latest Ubuntu 22.04 AMI
AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
    --query "reverse(sort_by(Images, &CreationDate))[0].ImageId" \
    --region "$REGION" --output text)

echo "✓ Latest Ubuntu AMI: $AMI_ID"

# 3. Create UserData script for EC2 instance launch
USER_DATA=$(cat << 'UD-EOF'
#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git
git clone https://github.com/priyanshu-projects/Movie_Recommender.git /home/ubuntu/app
cd /home/ubuntu/app
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install boto3

# Create systemd service for Streamlit
sudo cat > /etc/systemd/system/streamlit.service << 'SERVICE-EOF'
[Unit]
Description=Movie Recommender Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
ExecStart=/home/ubuntu/app/.venv/bin/streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE-EOF

sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
UD-EOF
)

echo "=== EC2 Setup Instructions ==="
echo "To launch your free-tier t3.micro EC2 instance, run:"
echo "aws ec2 run-instances --image-id $AMI_ID --instance-type t3.micro --security-group-ids $SG_ID --region $REGION"
echo ""
echo "✓ All EC2 setup definitions ready!"
