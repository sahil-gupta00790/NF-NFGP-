# Security Groups
resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-alb-sg"
  description = "Allow HTTP/HTTPS to ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  # Add HTTPS (443) if you plan to use it
  # ingress {
  #   from_port   = 443
  #   to_port     = 443
  #   protocol    = "tcp"
  #   cidr_blocks = ["0.0.0.0/0"]
  # }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "${var.project_name}-alb-sg"
  }
}

resource "aws_security_group" "ecs_tasks_sg" {
  name        = "${var.project_name}-ecs-tasks-sg"
  description = "Allow traffic from ALB to ECS tasks"
  vpc_id      = aws_vpc.main.id

  # Ingress from ALB
  ingress {
    from_port       = 0 # Allows any port from the ALB
    to_port         = 0
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id] # Only from our ALB
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"] # Allow tasks to pull images, talk to AWS services
  }
  tags = {
    Name = "${var.project_name}-ecs-tasks-sg"
  }
}


# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public[*].id # Deploy ALB in public subnets

  enable_deletion_protection = false # Set to true for production
  tags = {
    Name = "${var.project_name}-alb"
  }
}

# ALB Target Group for Frontend
resource "aws_lb_target_group" "frontend" {
  name        = "${var.project_name}-frontend-tg"
  port        = var.frontend_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # For Fargate

  health_check {
    enabled             = true
    path                = "/" # Adjust if your frontend has a specific health check path
    protocol            = "HTTP"
    matcher             = "200-399"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
  tags = {
    Name = "${var.project_name}-frontend-tg"
  }
}

# ALB Listener for HTTP
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
}
# Add HTTPS listener here if you have a certificate in ACM

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
  tags = {
    Name = "${var.project_name}-cluster"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ecs_logs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 7 # Adjust as needed
  tags = {
    Name = "${var.project_name}-ecs-logs"
  }
}

# --- Task Definitions ---
# Note: For simplicity, putting all in one task. You might separate them for independent scaling.
# If separated, they'd need service discovery or use ALB for backend/worker too.

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = var.aws_region

  frontend_image_uri = "${local.account_id}.dkr.ecr.${local.region}.amazonaws.com/${var.project_name}/frontend:${var.frontend_image_tag}"
  backend_image_uri  = "${local.account_id}.dkr.ecr.${local.region}.amazonaws.com/${var.project_name}/backend:${var.backend_image_tag}" # Also for worker
  redis_image_uri    = var.redis_image_name # e.g., redis:alpine (public)
}

data "aws_caller_identity" "current" {}

resource "aws_ecs_task_definition" "app_task" {
  family                   = "${var.project_name}-app"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.fargate_cpu
  memory                   = var.fargate_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  # task_role_arn          = aws_iam_role.ecs_task_role.arn # If tasks need more specific permissions

  container_definitions = jsonencode([
    {
      name      = "${var.project_name}-frontend"
      image     = local.frontend_image_uri
      essential = true
      cpu       = floor(var.fargate_cpu / 3) # Example allocation
      memory    = floor(var.fargate_memory / 3)
      portMappings = [
        {
          containerPort = var.frontend_port
          hostPort      = var.frontend_port # Not strictly needed for Fargate IP mode, but good practice
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "NEXT_PUBLIC_API_URL"
          # For intra-task communication if backend is in same task:
          value = "http://localhost:${var.backend_port}/api/v1"
          # If backend is a separate service, this would be its discovery name or ALB DNS
        }
        # Add other frontend ENV VARS
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "frontend"
        }
      }
      # Healthcheck for frontend might not be strictly necessary if ALB handles it
      # dependsOn = [
      #   { containerName = "${var.project_name}-backend", condition = "HEALTHY" }
      # ]
    },
    {
      name      = "${var.project_name}-backend"
      image     = local.backend_image_uri
      essential = true
      cpu       = floor(var.fargate_cpu / 3)
      memory    = floor(var.fargate_memory / 3)
      portMappings = [
        { containerPort = var.backend_port, hostPort = var.backend_port, protocol = "tcp" }
      ]
      environment = [ # Pass ENV VARS from .env via Jenkins or secrets manager
        { name = "REDIS_HOST", value = "localhost" }, # If Redis is in the same task
        { name = "REDIS_PORT", value = tostring(var.redis_port) }
        # Add other backend ENV VARS (GEMINI_API_KEY, etc.)
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "backend"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.backend_port}/api/v1/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 300 # Ample time for backend + RAG init
      }
      dependsOn = [
        { containerName = "${var.project_name}-redis", condition = "HEALTHY" }
      ]
    },
    {
      name      = "${var.project_name}-celeryworker"
      image     = local.backend_image_uri # Assuming same image as backend
      essential = true
      cpu       = floor(var.fargate_cpu / 3)
      memory    = floor(var.fargate_memory / 3)
      # No port mappings needed for worker usually
      command = ["celery", "-A", "app.core.celery_app.celery_app", "worker", "--loglevel=INFO", "-P", "solo"] # Your Celery command
      environment = [
        { name = "REDIS_HOST", value = "localhost" },
        { name = "REDIS_PORT", value = tostring(var.redis_port) }
        # Add other worker ENV VARS
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "celeryworker"
        }
      }
      # Worker might not need its own health check in the task def, depends on its resilience
      dependsOn = [
        { containerName = "${var.project_name}-redis", condition = "HEALTHY" },
        { containerName = "${var.project_name}-backend", condition = "HEALTHY" } # Ensure backend (API) is up if worker calls it
      ]
    },
    {
      name      = "${var.project_name}-redis"
      image     = local.redis_image_uri
      essential = true
      # Allocate minimal CPU/Memory for Redis if it's just for Celery/caching
      cpu       = 128 # Example small allocation
      memory    = 256
      portMappings = [
        { containerPort = var.redis_port, hostPort = var.redis_port, protocol = "tcp" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "redis"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "redis-cli -h localhost -p ${var.redis_port} ping | grep PONG || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
  tags = {
    Name = "${var.project_name}-app-task-def"
  }
}

# --- ECS Service for the combined App Task ---
# This service will manage tasks running frontend, backend, worker, redis together.
# ALB will only target the frontend container's port in these tasks.
resource "aws_ecs_service" "app_service" {
  name            = "${var.project_name}-app-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app_task.arn
  desired_count   = var.desired_task_count_frontend # The service manages N copies of the entire task (all 4 containers)
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id # Run tasks in private subnets
    security_groups = [aws_security_group.ecs_tasks_sg.id]
    assign_public_ip = false # Fargate tasks in private subnets get ENIs but no public IPs directly
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.frontend.arn
    container_name   = "${var.project_name}-frontend" # ALB targets the frontend container
    container_port   = var.frontend_port
  }

  deployment_circuit_breaker { # Optional: helps with bad deployments
    enable   = true
    rollback = true
  }

  depends_on = [aws_lb_listener.http] # Ensure listener is created before service
  tags = {
    Name = "${var.project_name}-app-service"
  }
}
