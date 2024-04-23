# Heart-Disease-AI-kaggle
## 목차
- 프로젝트 설명
- 데이터 정보
- 데이터 로드
- EDA
    - 데이터 확인
    - 중복값 및 결측치 처리
    - 데이터 이상치 처리
- 모델 학습
    - 데이터 분류
    - 머신러닝 모델 학습
    - 딥러닝 모델 학습
    - 모델 결정
- 최종 모델 학습
    - 모델 학습
    - 조정
- 결론

---

## 프로젝트 설명


## 데이터 정보
이 심장 질환 데이터 세트는 이미 독립적으로 사용 가능하지만 이전에는 결합되지 않은 5개의 인기 심장 질환 데이터 세트를 결합하여 선별되었습니다. 이 데이터 세트에는 5개의 심장 데이터 세트가 11개의 공통 기능과 결합되어 있어 지금까지 연구 목적으로 사용할 수 있는 심장 질환 데이터 세트 중 가장 큰 것입니다. 큐레이션에 사용되는 5개의 데이터 세트는 다음과 같습니다.  
  
- 클리블랜드
- 헝가리 인
- 스위스
- 롱비치 버지니아
- Statlog(심장) 데이터 세트.
  
이 데이터 세트는 11가지 기능을 갖춘 1,190개의 인스턴스로 구성됩니다. 이러한 데이터 세트는 CAD 관련 기계 학습 및 데이터 마이닝 알고리즘에 대한 고급 연구를 돕고 궁극적으로 임상 진단 및 조기 치료를 발전시키기 위해 한곳에 수집 및 결합되었습니다.

### 컬럼
- Sex 
    - 1 = male, 0= female; 
- Chest Pain Type
    - Value 1: typical angina  
    - Value 2: atypical angina  
    - Value 3: non-anginal pain  
    - Value 4: asymptomatic  
- Fasting Blood sugar
    - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)  
- Resting electrocardiogram results|    
    - Value 0: normal  
    - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)  
    - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria  
- Exercise induced angina
    - 1 = yes; 0 = no  
- the slope of the peak exercise ST segment    
    - Value 1: upsloping  
    - Value 2: flat  
    - Value 3: downsloping  
- class 
    - 1 = heart disease, 0 = Normal  

## 데이터 로드
프로젝트에 사용할 라이브러리를 설치하고 불러온다. pandas의 read_csv로 "archive/heart_statlog_cleveland_hungary_final.csv" 파일을 불러온다. 
```python
# 사용한 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap

# 데이터 로드
df = pd.read_csv("archive/heart_statlog_cleveland_hungary_final.csv")
```
  

## EDA
### 데이터 확인
"info()"사용하여 데이터프레임에 있는 데이터의 결측치와 Dtype을 확인함 총 1190개의 row와 12개의 column 이 존재하는 것을 알 수 있고 데이터는 데체로 int64와 float64로 이루어져 있다는 것을 알 수 있다.

RangeIndex: 1190 entries, 0 to 1189  
Data columns (total 12 columns):  
     Column               Non-Null Count  Dtype    
---  ------               --------------  -----    
 0   age                  1190 non-null   int64    
 1   sex                  1190 non-null   int64    
 2   chest pain type      1190 non-null   int64    
 3   resting bp s         1190 non-null   int64    
 4   cholesterol          1190 non-null   int64    
 5   fasting blood sugar  1190 non-null   int64    
 6   resting ecg          1190 non-null   int64    
 7   max heart rate       1190 non-null   int64    
 8   exercise angina      1190 non-null   int64    
 9   oldpeak              1190 non-null   float64  
 10  ST slope             1190 non-null   int64    
 11  target               1190 non-null   int64    
dtypes: float64(1), int64(11)  
memory usage: 111.7 KB  
  
### 중복값 및 결측치 확인
#### 결측치 확인 
info 에서도 알 수 있지만 정확히 하기 위해서 "insull().sum()"으로 컬럼별 결측치를 확인한다. 해당 데이터에는 결측치는 존재하지 않는다.
  
age                    0  
sex                    0  
chest pain type        0  
resting bp s           0  
cholesterol            0  
fasting blood sugar    0  
resting ecg            0  
max heart rate         0  
exercise angina        0  
oldpeak                0  
ST slope               0  
target                 0  
dtype: int64  
  
#### 중복값
"duplicated().sum()" 이용하여 데이터프레임에 있는 중복값을 확인한다. 그 결과 "272"개의 중복값이 존재하는 것을 확인했다. 해당 값의 중복값이 진짜인지 확인하기 위해서
"loc"을 이용해 중복인 데이터의 인덱스를 추출해서 직접적으로 확인하였다. 확인 결과 데이터들을 확실한 중복값으로 나이, 성별, 혈압 등 모든 데이터가 일치 했다. 해당 데이터를 지우개 되면 1190개에서 916개로 데이터가 확실하게 줄어드는 것을 확인 할 수 있다. 그래서 해당 데이터가 모델의 학습을 위해서 데이터를 인위적으로 확장한 데이터 일 가능을 생각해 따로 중복값이 존재하는 데이터프레임을 새로 생산해서 보존하고 원본 데이터는 중복값을 지우고 프로젝트를 진행한다.
  
### 데이터 이상치 확인
int64 데이터 타입을 가진 컬럼은 대체로 

## 모델 학습
### 데이터 분류
### 머신러닝 모델 학습
### 딥러닝 모델 학습

## 결론