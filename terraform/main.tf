terraform {
  required_version = ">= 1.8.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
  }
}

provider "aws" {
  region = var.aws_region
}


# Save private key locally
resource "local_file" "private_key_pem" {
  content         = tls_private_key.icu_key.private_key_pem
  filename        = "${path.module}/icu_model.pem"
  file_permission = "0400"
}

# Generate a new SSH key locally
resource "tls_private_key" "icu_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

#Create AWS key pair using the generated public key
resource "aws_key_pair" "icu_key" {
  key_name   = "icu-model-key"
  public_key = tls_private_key.icu_key.public_key_openssh
}

#Security Group — Allow HTTP & SSH
resource "aws_security_group" "app_sg" {
  name        = "icu-model-sg"
  description = "Allow HTTP and SSH access"

  ingress {
    description = "HTTP access"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 Instance
resource "aws_instance" "app_server" {
  ami                    = "ami-0f5fcdfbd140e4ab7"
  instance_type          = var.instance_type
  key_name               = aws_key_pair.icu_key.key_name
  vpc_security_group_ids = [aws_security_group.app_sg.id]

  user_data = file("${path.module}/user_data.sh")

  tags = {
    Name = "AI-ICU-Monitoring"
  }
}

# Elastic IP
resource "aws_eip" "app_eip" {
  instance = aws_instance.app_server.id
  domain   = "vpc"

  tags = {
    Name = "ICU-model-EIP"
  }
}


