
#IAM Role for EC2
resource "aws_iam_role" "ec2_s3_role" {
  name = "icu-ec2-s3-access-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

#IAM Policy to allow access to your S3 bucket
resource "aws_iam_policy" "s3_read_policy" {
  name        = "icu-s3-read-policy"
  description = "Allow EC2 to read objects from the icu-model-datasets bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::icu-model-datasets",
          "arn:aws:s3:::icu-model-datasets/*"
        ]
      }
    ]
  })
}

#Attach the policy to the role
resource "aws_iam_role_policy_attachment" "attach_s3_policy" {
  role       = aws_iam_role.ec2_s3_role.name
  policy_arn = aws_iam_policy.s3_read_policy.arn
}

#Instance profile for EC2
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "icu-ec2-instance-profile"
  role = aws_iam_role.ec2_s3_role.name
}

