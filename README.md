# FSI AIxData Challenge 2024

- 주최 / 주관 : 금융보안원
- 후원 : 금융위원회, KB국민은행, 하나은행, 미래에셋증권, 생명보험협회
- 운영 : 데이콘

- Link : https://dacon.io/competitions/official/236297/overview/description

1) 클래스 불균형이 심한 데이터셋의 특성을 고려하여 분류 AI모델 개발

2) 제공하는 데이터셋을 오픈소스 생성형 AI 모델 등 AI 기술에 응용 

3) 이를 분류 AI모델에 활용함으로써 분류 AI모델의 성능을 개선

## Experimental environment
- CPU : Intel i7-7700
- GPU : NVIDIA GeForce RTX 2080 2way
- RAM : 64Gb
- OS : Windows 10
- You need a GPU with at least 8GB of VRAM to run this code.

## Requirements
- python==3.8.19
- numpy==1.23.4
- pandas==2.0.3
- scipy== 1.10.1
- tqdm==4.66.5
- scikit-learn==1.3.2  
- xgboost==1.6.2
- ForestDiffusion==1.0.6
- sdv==1.16.0
- torch==2.4.1
- transformers==4.44.2
- langchain==0.2.16
- openpyxl==3.1.5
- langchain_community== 0.2.16
- bitsandbytes==0.43.3
- accelerate==0.22.0

## Install required packages
- pip install numpy==1.23.4
- pip install pandas==2.0.3
- pip install scipy== 1.10.1
- pip install tqdm==4.66.5
- pip install scikit-learn==1.3.2  
- pip install xgboost==1.6.2
- pip install ForestDiffusion==1.0.6
- pip install sdv==1.16.0
- pip install torch==2.4.1
- pip install transformers==4.44.2
- pip install langchain==0.2.16
- pip install openpyxl==3.1.5
- pip install langchain_community== 0.2.16
- pip install bitsandbytes==0.43.3
- pip install accelerate==0.22.0

## Package Description
- CTGANGenerator.py : Generate synthetic data. It creates ctgan.csv.
- ForestDiffusionGenerator.py : Generate synthetic data. It creates forestdiffusion.csv.
- FraudDetectionModel.py : Generate metadata and perform stacking ensemble learning.
- LLM_Masking.py : Mask personal information features in ctgan.csv.
- main.py : Run FraudDetectionModel.py and LLM_Masking.py

## Directory Structure
<pre><code>
/workspace
├── data
│   ├── sample_submission.zip
│   ├── sample_submission.csv
│   ├── train.csv
│   ├── test.csv
│   ├── 데이터_명세_및_생성조건.xlsx
├── meta_data
│   ├── meta_ml_X_test_721.npy
│   ├── meta_ml_X_test_723.npy
│   ├── meta_ml_X_test_826.npy
│   ├── meta_ml_X_test_1005.npy
│   ├── meta_ml_X_test_1008.npy
│   ├── meta_ml_X_test_1011.npy
│   ├── meta_ml_X_test_forestdiffusion_826.npy
│   ├── meta_ml_X_train_721.npy
│   ├── meta_ml_X_train_723.npy
│   ├── meta_ml_X_train_826.npy
│   ├── meta_ml_X_train_1005.npy
│   ├── meta_ml_X_train_1008.npy
│   ├── meta_ml_X_train_1011.npy
│   ├── meta_ml_X_train_forestdiffusion_826.npy
├── submission
│   ├── FSI_TH.zip
├── syn_data
│   ├── ctgan.csv
│   ├── ctgan_syn_submission.csv
│   ├── forestdiffusion.csv
├── CTGANGenerator.py
├── ForestDiffusionGenerator.py
├── FraudDetectionModel.py
├── LLM_Masking.py
├── main.py
      .
      .
      .
</code></pre>