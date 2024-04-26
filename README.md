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
데이터의 분류는 train_test_split을 사용해서 옵션은 test_size = 0.25, random_state = 42을 설정하여 4개의 데이터프레임으로 train, val, test로 3개의 데이터로 각각 분류하였다.  
  
### 머신러닝 모델 학습
- 사용한 모델 종류
1. DecisionTreeClassifier (결정 트리 분류기):
    - 결정 트리는 데이터의 특성을 기반으로 결정 경로를 트리 구조로 나타내어 결과를 예측하는 모델입니다. 각 노드는 하나의 특성에 대한 결정(예/아니오)을 나타내고, 이를 통해 데이터를 분류합니다.
2. RandomForestClassifier (랜덤 포레스트 분류기):
    - 랜덤 포레스트는 여러 결정 트리를 조합해서 사용하는 앙상블 학습 방법입니다. 각각의 트리는 데이터의 부분 집합을 무작위로 선택하여 학습하며, 최종 결정은 각 트리의 예측을 평균내어 결정합니다.
3. XGBClassifier (XGBoost 분류기):
    - XGBoost는 향상된 그라디언트 부스팅 알고리즘으로, 여러 약한 학습기(주로 결정 트리)를 순차적으로 학습하여 강력한 예측 모델을 만드는 방식입니다. 각 단계에서 이전 모델의 오류를 줄이는 방향으로 학습합니다.
4. LogisticRegression (로지스틱 회귀):
    - 로지스틱 회귀는 분류를 위해 사용되는 선형 모델입니다. 결과가 두 가지 중 하나로 분류되는 이진 분류 문제에 주로 사용됩니다. 결과는 로지스틱 함수를 사용하여 확률로 표현됩니다.
총 4가지의 모델을 사용하여 머신러닝 모델을 학습하기로 한다. 해당 모델 중에서 가장 좋은 학습률이 높은 모델을 채택해서 해당 모델을 조정하는 방식으로 하기위해 가장 간단한 옵션으로 모델을 학습시켰다.
- 결과
    - 머신러닝 모델 학습률 평균
        - logistic  =  0.86
        - decision_tree  =  0.851
        - random_forest  =  0.867
        - xgb  =  0.881
    - 데이터별 학습률 평균
        - 이상치 있는 그대로  =  0.862
        - 이상치 제거  =  0.861
        - 비율에 맞게 랜덤으로 대체  =  0.866
        - 평균으로 대체  =  0.87
    - 데이터는 cholesterol을 평균으로 대체했을 때 가장 좋은 학습률을 보였고 머신러닝 모델은 XGBClassifier가 제일 좋은 학습률을 보였다.

#### 머신러닝 모델 학습결과 
위 데이터에서 xgb 모델은 mean 데이터셋에 대한 정밀도, 재현율, F1 점수, 그리고 정확도가 매우 높게 나타나 (precision_0 = 0.926, precision_1 = 0.865, recall_0 = 0.818, recall_1 = 0.947, f1_score_0 = 0.869, f1_score_1 = 0.905, accuracy = 0.890), 전체적으로 가장 우수한 성능을 보이는 것으로 평가됩니다. 이를 통해 xgb가 다양한 데이터셋에서 높은 성능을 제공할 수 있는 강력한 모델임을 알 수 있습니다.
  
### 딥러닝 모델 학습
- PyTorch
    - Facebook의 AI Research lab에 의해 개발된 PyTorch는 동적 계산 그래프를 지원하는 것이 특징입니다. 이는 모델을 실행하는 도중에도 구조를 변경할 수 있게 해주어 실험적인 프로젝트와 연구에 매우 적합합니다. PyTorch는 직관적이고 사용하기 쉬운 인터페이스를 제공하며, 빠른 속도와 효율적인 메모리 사용으로 많은 연구자와 개발자에게 인기가 있습니다.
- 학습 과정
    1. PyTorch 텐서로 변환  
        - PyTorch에서 처리할 수 있도록 데이터프레임을 텐서로 변환합니다. 이 때, 입력 변수와 타겟 변수를 명확히 구분해야 합니다.  
    2. 데이터 로더 설정
        - PyTorch의 DataLoader를 사용하여 배치 처리, 셔플링, 다양한 데이터 로딩 옵션을 설정할 수 있습니다.
    3. 모델 학습
        - 모델 정의 :  PyTorch에서는 nn.Module을 상속받는 클래스를 정의하여 모델의 아키텍처를 구성합니다. 이 클래스 내에서 forward 메서드를 구현하여 데이터가 네트워크를 통과하는 방식을 정의합니다.
        - 손실 함수와 최적화 알고리즘 선택: 모델 학습을 위해 손실 함수(loss function)를 정의하고, 최적화 알고리즘(optimizer)을 선택합니다. 손실 함수는 모델의 예측과 실제 레이블 사이의 차이를 측정하고, 이 정보를 바탕으로 모델을 개선하는 데 사용됩니다. PyTorch에서는 다양한 손실 함수와 최적화 알고리즘을 제공합니다.
        - 학습 과정: 모델을 실제로 학습시키는 단계입니다. 일반적으로 데이터셋을 여러 번 반복하면서(에포크), 각 배치에 대해 모델의 예측을 실행하고, 손실을 계산한 후, 이를 바탕으로 모델의 가중치를 업데이트합니다. 이 과정에서는 backward() 메서드를 통해 손실에 대한 그라디언트를 계산하고, optimizer.step() 메서드로 가중치를 업데이트합니다.
        - 평가: 모델의 성능을 평가하기 위해 검증 데이터셋 또는 테스트 데이터셋을 사용합니다. 모델 평가는 학습 과정과 별도로 진행되며, 이 단계에서는 모델의 일반화 능력을 확인할 수 있습니다. 모델이 새로운 데이터에 대해 얼마나 잘 작동하는지를 측정하고, 필요한 경우 학습 과정을 조정합니다.
        - 모델 저장 및 재사용: 학습이 완료된 모델은 파일로 저장할 수 있으며, 나중에 다시 로드하여 사용할 수 있습니다. PyTorch에서는 torch.save 함수를 사용하여 모델의 상태를 저장하고, torch.load로 모델을 다시 로드할 수 있습니다.
- 결과
    - 평균 학습률이 50% 이하로 사실상 과정과 시간에 비해 학습률이 너무 낮으며 학습률의 편차가 심한 것을 알 수도 있다.


### 모델결정
 결과를 토대로, 머신러닝 모델 중 XGBClassifier가 다양한 데이터셋에서 가장 우수한 성능을 보였습니다. 특히, cholesterol 데이터를 평균값으로 대체했을 때 가장 높은 학습률을 나타내며 정밀도, 재현율, F1 점수, 그리고 정확도 모두에서 높은 평가를 받았습니다. 반면, 딥러닝 모델로 사용된 PyTorch의 학습률은 평균 50% 이하로 낮고, 학습률의 편차도 심해 효율적이지 못했습니다. 이에 따라, 향후 분석과 예측 작업에는 XGBClassifier를 주 모델로 채택하고 최적화하는 방향이 가장 적합할 것으로 결정됩니다. 이 모델은 높은 성능을 제공할 뿐만 아니라, 이상치 처리 방법에 따른 성능 변화도 잘 관리할 수 있음을 보여주었습니다.
  


## 최종 모델 학습

### 모델 구성

이진 분류를 위한 최종 모델은 XGBoost 라이브러리의 `XGBClassifier`를 사용하여 구성하였습니다. `dart` 부스터와 함께 다음의 하이퍼파라미터 설정을 사용하였습니다:

- `max_depth`: 3 — 트리의 최대 깊이.
- `learning_rate`: 0.1 — 학습률.
- `n_estimators`: 100 — 부스팅 라운드 수.
- `subsample`: 0.8 — 각 트리에 대한 트레이닝 데이터 샘플 비율.
- `colsample_bytree`: 0.6 — 각 트리에 대한 피처 샘플링 비율.
- `colsample_bylevel`: 0.7 — 각 레벨에 대한 피처 샘플링 비율.
- `colsample_bynode`: 0.8 — 각 노드에 대한 피처 샘플링 비율.
- `max_bin`: 256 — 히스토그램의 최대 bin 개수.
- `objective`: 'binary:logistic' — 이진 분류를 위한 로지스틱 회귀 목적.
- `sample_type`: "weighted" — 더트 부스터의 샘플링 유형.

### 학습 과정

모델 학습은 `X_train` 트레이닝 세트와 `y_train` 레이블을 사용하여 수행되었습니다. 학습 과정에서는 `watchlist`에 지정된 트레이닝 세트와 검증 세트(`X_val`, `y_val`)의 성능을 모니터링하였으며, `early_stopping_rounds=50`을 설정하여 조기 종료 기능을 활성화하였습니다.

### 모델 평가

학습된 모델은 검증 데이터셋 `X_val`에 대한 예측을 수행하였으며, 성능 평가를 위해 혼동 행렬(confusion matrix)과 AUC 점수를 출력하였습니다. 다음은 모델 평가 결과입니다:


### 조정


## 결론
