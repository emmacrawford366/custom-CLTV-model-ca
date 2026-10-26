docker build -t acm_cltv_train:latest .

# docker run -v "<path to the ml folder>":/opt/ml acm_cltv_train:latest
docker run -v /home/ec2-user/SageMaker/AMC_Custom_Models/Models/CLTV Prediction/ml:/opt/ml acm_cltv_train:latest
