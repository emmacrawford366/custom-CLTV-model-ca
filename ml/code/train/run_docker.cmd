docker build -t acm_cltv_train:latest .

:: docker run -v "<path to the ml folder>":/opt/ml acm_cltv_train:latest
docker run -v "C:\Users\bkumards\Desktop\ATS\ACM\Models\CLTV Prediction\ml":/opt/ml acm_cltv_train:latest
