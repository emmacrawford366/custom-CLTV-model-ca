docker build -t acm_cltv_infer:latest .

# docker run -v "<path to the ml folder>":/opt/ml cltv_infer:latest
docker run -v "/Users/clarkeee/Desktop/ATS/Audience_Modeler/CustomerLongtermValue/Customer_Longterm_Value_Prediction/ml":/opt/ml acm_cltv_infer:latest

