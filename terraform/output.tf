output "web_url" {
  value = "http://${aws_eip.app_eip.public_ip}"
}

output "ssh_command" {
  value = "ssh -i ${var.ssh_private_key_path} ec2-user@${aws_eip.app_eip.public_ip}"
}
