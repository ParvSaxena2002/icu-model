variable "aws_region" {
  description = "AWS region"
  default     = "us-east-2"
}

variable "ssh_public_key_path" {
  description = "Path to your SSH public key"
  default     = "C:/Users/amanp/.ssh/id_rsa.pub"

}

variable "ssh_private_key_path" {
  description = "Path to your SSH private key"
  default     = "C:/Users/amanp/.ssh/id_rsa.pub"

}

variable "instance_type" {
  description = "EC2 instance type"
  default     = "c7i-flex.large"
}
