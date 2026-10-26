curl -X GET http://127.0.0.1:8080/ping

curl -X POST http://127.0.0.1:8080/invocations -H "Content-Type: text/csv" --data-binary @"C:\Users\bkumards\Desktop\ATS\ACM\Models\CLTV Prediction\ml\input\data\infer\synthetic_data.csv" -o "C:\Users\bkumards\Desktop\ATS\ACM\Models\CLTV Prediction\ml\output\data\audiences\output.csv"
