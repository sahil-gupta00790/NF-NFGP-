terraform {
  backend "s3" {
    bucket         = "your-unique-neuroforge-tfstate-bucket" # REPLACE with your S3 bucket name
    key            = "global/s3/terraform.tfstate"
    region         = "us-east-1" # Match your var.aws_region
    # dynamodb_table = "terraform-locks" # Optional: for state locking with DynamoDB
  }
}
