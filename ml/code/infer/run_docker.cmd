docker build -t acm_cltv_infer:latest .

:: docker run -v "<path to the ml folder>":/opt/ml cltv_infer:latest
docker run -p 8080:8080 -e top_pct=15 -v "C:\Users\bkumards\Desktop\ATS\ACM\Models\CLTV Prediction\ml":/opt/ml acm_cltv_infer:latest

