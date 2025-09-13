# Student Assessment & Disability Analysis

This repository contains two Python classes designed for analyzing student performance, attendance, and assessment data
This data is being pulled from PQ database hence thje Config postgresClient.py as well as incoming request are processed by rabbitMQ.

- **`DisabilityAnalysis`**  
  Uses assessment and attendance data together with a pre-trained **logistic regression model** to classify whether a student may require additional support.  

- **`AssessmentAnalysis`**  
  Provides multiple utilities for analyzing raw assessment scores, subjects, moving averages, and generates predictions with a **linear regression model**.  


Both classes are built to work with **pandas** DataFrames and **pickled sklearn-style models**.

Some planned improvements
1. Pipeline to train model on new dataset rather from Kaggle dataset, combine and process new model.
2. Dynamic training model -> Run corn job -> Train model -> (Results > prev model) -> deploy.
3. Discover new relationships between database tables for new insights.


---

## 📂 Project Structure

.
├── Models/
│   ├── logistic_model.pkl   
│   └── linear_model.pkl     
├── Client/
│   └── main.py
├── S3/
│   └── main.py
├── Config/
│   ├── PostgresClient.py  
│   └── RabbitMQ.py   
├── Disability_analysis/
│   ├── test  
│   └── Main.py   
├── Assessment_analysis/
│   ├── test  
│   └── Main.py 
├── main.py  
├── Dockerfile
├── Makefile
├── Requirements.txt 
└── README.md



---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/student-analysis.git
cd student-analysis
pip install -r requirements.txt

python3 main.py

