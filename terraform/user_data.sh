#!/bin/bash
yum update -y
yum install -y python3-pip git

# Prepare environment
mkdir -p /home/ec2-user/app
chown ec2-user:ec2-user /home/ec2-user/app


pip3 install flask fastapi uvicorn --quiet

cat <<EOF > /etc/systemd/system/aiicu.service
[Unit]
Description=ICU-web App
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/app
ExecStart=/usr/bin/python3 alarm_monitor.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF
