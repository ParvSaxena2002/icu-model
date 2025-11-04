output "web_url" {
  value = "http://${aws_eip.app_eip.public_ip}"
}

output "ssh_command" {
  value = "ssh -i ${var.ssh_private_key_path} ec2-user@${aws_eip.app_eip.public_ip}"
}

output "static_ip" {
  description = "Elastic (static) IP address"
  value       = aws_eip.app_eip.public_ip
}

output "private_key_file" {
  description = "Location of the private key file"
  value       = local_file.private_key_pem.filename
}

