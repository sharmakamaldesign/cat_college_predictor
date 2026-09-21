aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 264965790211.dkr.ecr.ap-south-1.amazonaws.com
docker buildx build --platform linux/amd64 -f Dockerfile-qa -t cat-college-predictor-lambda  --load .
docker tag cat-college-predictor-lambda:latest 264965790211.dkr.ecr.ap-south-1.amazonaws.com/cat-college-predictor-lambda:qa
docker push 264965790211.dkr.ecr.ap-south-1.amazonaws.com/cat-college-predictor-lambda:qa