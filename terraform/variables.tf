variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1" # Choose your preferred region
}

variable "project_name" {
  description = "A unique name for the project"
  type        = string
  default     = "neuroforge"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "List of CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"] # Ensure you have at least 2 for ALB/NAT GW HA
}

variable "private_subnet_cidrs" {
  description = "List of CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"] # Match number of public subnets
}

variable "frontend_image_tag" {
  description = "Docker image tag for the frontend (e.g., latest, or a specific version)"
  type        = string
  default     = "latest"
}

variable "backend_image_tag" {
  description = "Docker image tag for the backend/worker (e.g., latest)"
  type        = string
  default     = "latest"
}

variable "redis_image_name" {
  description = "Docker image for Redis (e.g., redis:alpine)"
  type        = string
  default     = "redis:alpine"
}

variable "frontend_port" {
  description = "Port the frontend container listens on"
  type        = number
  default     = 3000
}

variable "backend_port" {
  description = "Port the backend container listens on"
  type        = number
  default     = 8000
}

variable "redis_port" {
  description = "Port the Redis container listens on"
  type        = number
  default     = 6379
}

variable "fargate_cpu" {
  description = "Fargate task CPU units (e.g., 256, 512, 1024)"
  type        = number
  default     = 512 # Adjust based on needs, affects cost
}

variable "fargate_memory" {
  description = "Fargate task memory in MiB (e.g., 512, 1024, 2048)"
  type        = number
  default     = 1024 # Adjust based on needs, affects cost
}

variable "desired_task_count_frontend" {
  description = "Desired number of tasks for the frontend service"
  type        = number
  default     = 1 # Start with 1 for free tier considerations
}

variable "desired_task_count_backend" {
  description = "Desired number of tasks for the backend service"
  type        = number
  default     = 1
}

variable "desired_task_count_worker" {
  description = "Desired number of tasks for the celery worker service"
  type        = number
  default     = 1
}

variable "jenkins_instance_type" {
  description = "EC2 instance type for Jenkins server"
  type        = string
  default     = "t2.micro" # Free tier eligible
}

variable "jenkins_key_name" {
  description = "Name of the EC2 Key Pair to use for Jenkins server (must exist in your AWS account)"
  type        = string
  # default   = "your-ec2-key-pair-name" # IMPORTANT: Set this or create one
}
