data "aws_availability_zones" "available" {
  state = "available"
}

# AMI for Jenkins Server (Amazon Linux 2)
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"] # Or use kernel-*-hvm for specific kernel needs
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
