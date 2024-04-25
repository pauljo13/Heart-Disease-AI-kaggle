# 🫀 Heart-Disease-AI-kaggle
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
- age : 나이  
- Sex : 성별
    - 1 = male, 0= female; 
- Chest Pain Type : 흉통 유형
    - Value 1: typical angina  
    - Value 2: atypical angina  
    - Value 3: non-anginal pain  
    - Value 4: asymptomatic
- resting bp s : 안정시 혈압(s)
- cholesterol : 콜레스테롤  
- Fasting Blood sugar : 공복 혈당
    - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)  
- Resting electrocardiogram results : 안정시 심전도    
    - Value 0: normal  
    - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)  
    - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
- max heart rate: 최대 심박수  
- Exercise induced angina : 운동 유발성 협심증
    - 1 = yes; 0 = no
- oldpeak: ST 분절 하강  
- the slope of the peak exercise ST segment : ST 분절 기울기    
    - Value 1: upsloping  
    - Value 2: flat  
    - Value 3: downsloping  
- target: 목표
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
데이터 유형에 따라 이상치를 확인한다.
- 범주형 데이터 Categorical Data
    - 범주형 데이터 컬럼 : sex, chest pain type, fasting blood sugar, resting ecg, exercise angina, ST slope, target
    - 범주형 데이터의 이상치는 레어 카테고리, 잘못된 라벨링, 구조적 이상치가 있다.
    - 이러한 이상치를 확인하기 위해 value_counts() 이용하여 컬럼의 값들을 살펴보면서 동시에 plot로 막대그래프를 그려서 데이터의 비율을 동시에 살펴 보았다.
    - 이상치가 있던 컬럼 : ST slope
        - 혼자만 "0" 인 데이터로 잘못된 라벨링이거나 구조적 이상치로 보여 해당 row을 삭제하는 것으로 이상치를 해결했다.
- 수치형 데이터 Numerical Data
    - 수치형 데이터 컬럼 : age, resting bp s, cholesterol, max heart rate, oldpeak
    - 수치형 데이터에서 이상치(outliers)는 데이터 세트에서 다른 데이터와 비교하여 비정상적으로 높거나 낮은 값을 가진 관측치를 말합니다. 이러한 이상치는 다양한 원인, 예를 들어 측정 오류, 데이터 입력 실수, 과정 변동성 또는 예외적인 사건 등으로 인해 발생할 수 있습니다. 
    - 수치형 데이터의 이상치를 보기 위해 컬럼의 최댓값, 최솟값, 평균과 편차, Z-점수, IQR, 정균 분포도 등을 통해서 데이터를 살펴보았다.
    - 이상치가 있던 컬럼 : resting bp s, cholesterol
    - resting bp s : 쉬고 있을 때의 혈압으로 인간의 혈압이 60미만인 경우는 거의 죽은 사람으로 봐야한다. 해당 컬럼에는 60미만, 심지어 0인 값이 존재한다. 이는 잘못된 이상치로 판단하여 제거함
    - cholesterol : 값이 0인 데이터가 생각보다 많다. 인간이 cholesterol이 0인 경우가 정상인일 수 없으며 해당 데이터가 오기입된 데이터라고 볼 수 있다. 이게 해당 데이터를 지우거나 데체해야 하는데 지우기에는 데이터 양이 많아서 평균으로 대체하거나 특정 컬럼의 비율과 표준편차를 고려하여 랜덤으로 값을 대체하는 방법등으로 데이터를 만들어 모델 학습에 사용하여 가장 좋은 방법이 무엇인지 알아보기로 하였다.
  

## 모델 학습
### 데이터 분류
데이터 이상치 처리 과정에서 3개의 데이터프레임 유형이 나왔다. 이 프로젝트에서 이상치 처리에서 좀 더 이상적인 것이 무엇지에 대해서 알아보기 위해 3개의 데이터프레임 모두 사용하여 데이터를 학습시켜서 학습률을 비교할 생각이다.  

### 머신러닝 모델 학습
### 딥러닝 모델 학습

## 결론