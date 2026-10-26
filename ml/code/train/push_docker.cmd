docker build -t acm_cltv_train:latest .

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 008971674832.dkr.ecr.us-east-1.amazonaws.com

aws ecr create-repository --repository-name acm_cltv_train

:: docker tag acm_cltv_train:latest <AWS Account ID>.dkr.ecr.us-east-1.amazonaws.com/acm_cltv_train:latest
docker tag acm_cltv_train:latest 008971674832.dkr.ecr.us-east-1.amazonaws.com/acm_cltv_train:latest

:: docker push <AWS Account ID>.dkr.ecr.us-east-1.amazonaws.com/acm_cltv_train:latest
docker push 008971674832.dkr.ecr.us-east-1.amazonaws.com/acm_cltv_train:latest
