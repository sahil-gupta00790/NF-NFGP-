resource "aws_security_group" "jenkins_sg" {
  name        = "${var.project_name}-jenkins-sg"
  description = "Allow SSH and HTTP to Jenkins server"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # CHANGE TO YOUR IP FOR PRODUCTION
  }
  ingress {
    from_port   = 8080 # Jenkins default port
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # CHANGE TO YOUR IP for production, or keep open if webhook needs it
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "${var.project_name}-jenkins-sg"
  }
}

resource "aws_instance" "jenkins_master" {
  ami           = data.aws_ami.amazon_linux_2.id
  instance_type = var.jenkins_instance_type
  key_name      = var.jenkins_key_name # Ensure this key pair exists in your AWS account
  subnet_id     = aws_subnet.public[0].id # Deploy Jenkins in a public subnet
  vpc_security_group_ids = [aws_security_group.jenkins_sg.id]
  iam_instance_profile = aws_iam_instance_profile.jenkins_profile.name
  associate_public_ip_address = true # Needs a public IP to be accessible

  user_data = <<-EOF
              #!/bin/bash
              sudo yum update -y
              sudo amazon-linux-extras install java-openjdk11 -y
              sudo wget -O /etc/yum.repos.d/jenkins.repo https://pkg.jenkins.io/redhat-stable/jenkins.repo
              sudo rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io-2023.key
              sudo yum install jenkins -y
              sudo systemctl enable jenkins
              sudo systemctl start jenkins
              sudo yum install -y git docker # Install Git and Docker on Jenkins server
              sudo systemctl enable docker
              sudo systemctl start docker
              sudo usermod -aG docker jenkins
              sudo usermod -aG docker ec2-user
              # Install Docker Compose
              sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
              sudo chmod +x /usr/local/bin/docker-compose
              # Install AWS CLI (often pre-installed on Amazon Linux 2, but good to ensure)
              # curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
              # unzip awscliv2.zip
              # sudo ./aws/install
              sudo systemctl restart jenkins # Restart Jenkins for group changes to take effect
              EOF
  tags = {
    Name = "${var.project_name}-jenkins-master"
  }
}
