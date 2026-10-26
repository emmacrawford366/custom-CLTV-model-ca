curl -X GET http://127.0.0.1:8080/ping

curl -X POST http://127.0.0.1:8080/invocations -H "Content-Type: text/csv" --data-binary @/home/ec2-user/SageMaker/AMC_Custom_Models/Models/CLTV_Prediction/ml/input/data/infer/synthetic_data.csv -o /home/ec2-user/SageMaker/AMC_Custom_Models/Models/CLTV_Prediction/ml/output/data/audiences/output.csv
