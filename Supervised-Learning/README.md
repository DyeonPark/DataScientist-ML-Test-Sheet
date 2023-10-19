- [Supervised-Learning](#supervised-learning)
  * [Regression](#regression)
    + [대표적인-모델](#대표적인-모델)
    + [Regression Metric](#regression-metric)
  * [Classification](#classification)
    + [대표적인-모델](#대표적인-모델)
    + [Classification Metric](#classification-metric)

# Supervised-Learning
본 페이지에서는 Task에 따라서 머신러닝 중 지도학습 방법을 나누어 설명한다
지도학습이란 입력과 출력 샘플 데이터가 있고, 주어진 입력으로부터 출력을 예측하고자 할 때 사용한다
이의 궁극적인 목표는 이전에 본적이 없는 새로운 데이터에 대해 정확한 예측 및 출력을 하기 위함이다

그리고 지도학습에는 대표적인 Task로 Regression(회귀)와 Classification(분류)가 있다

## Regression
### 대표적인-모델
```
# case1) 일반적인 경우
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

print(linear_reg.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(linear_reg.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
```
# case2) 다항식을 추가한 경우
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
poly.fit(X_train)
X_poly = poly.transform(X_train) # 다항식 추가

linear_reg.fit(X_train, y_train)

print(linear_reg.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(linear_reg.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
* **LinearRegression**
  - 가장 일반적이고 대표적인 선형 모델 중 하나인 LinearRegression
  - 구간 나누기를 통해서 성능을 향상시킬 수 있으며, 원본 특성에 다항식을 추가하는 등의 방법을 사용할 수 있음
  - 단, 이 방법을 사용할 때 학습 시간이 매우 오래 걸릴 수 있음에 주의해야 함

<br/>

```
from sklearn.linear_model import Ridge
ridge = Ridge()

print(ridge.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(ridge.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
* **Ridge**
  - L2 규제를 사용하여 가중치의 절댓값을 가능한 작게 만드는 회귀 모델
  - `alpha` 계수 값이 커질수록 계수의 절댓값 크기가 작아짐 (기본값 1.0)
  - 일반적으로 Lasso보다 Ridge를 더 선호함
  - `solver=sag` 옵션을 주면 대용량 데이터 처리 버전 사용 가능

<br/>

```
from sklearn.linear_model import Lasso
lasso = Lasso()

print(lasso.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(lasso.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
* **Lasso**
  - Ridge의 대안으로, L1 규제를 사용하며 계수가 0이 될 수도 있음 (완전히 제외되는 특성이 있다는 뜻)
  - `alpha` 값을 줄일 수 있으나, 기본 값인 1.0보다 줄이려면 `max_iter을` 늘려야 함
  - 특성이 많고, 그 중에 일부가 중요하다면 Lasso를 사용하기도 함


<br/>

### Regression Metric
회귀 모델에서 사용할 수 있는 주요 메트릭은 크게 3가지가 있다
```
from sklearn.metrics import mean_squared_error, r2_score

print(mean_squared_error(y_test, y_pred)) # MSE
print(round(mean_squared_error(y_test, y_pred) ** 0.5, 3)) # 반올림 적용한 RMSE
print(r2_score(y_test, y_pred)) # 위 에러와 달리 1에 가까울 수록 좋은 값
```
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* r2-score

<br/>

## Classification
### 대표적인 모델
```
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

print(log_reg.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(log_reg.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
* **LogisticRegression**
  - LogisticRegression은 기본적으로 L2 규제를 사용한다. 따라서 L1 규제를 사용하고 싶다면 `penalty="l1"` 옵션을 주어야 한다.
  - L2 규제일 때는 모든 특성을 사용하고, L1 규제일 때는 속성을 선별해서 사용한다
  - 소프트맥스 함수를 사용하기 때문에 다중 클래스 분류를 지원한다
  - `solver='sag'` 옵션을 주면 대용량 데이터 처리 버전으로 사용할 수 있다
  - 회귀 모델인 LinearRegression과 혼동하지 않도록 주의해야 한다

<br/>

```
from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, y_train)

print(svc.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(svc.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
* **SVC**
  - Support Vector Machine을 Classification 목적으로 사용한 모델이다
  - L2 규제를 사용한다
  - C 값이 낮아질수록, 안정적인 일반화를 위해 계수를 0에 맞추려고 하는 경향이 있다
  - 기본적으로 이진 분류를 지원한다

<br/>

```
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
print(rfc.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(rfc.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
* **RandomForestClassifier**
  - 랜덤 포레스트의 원리는 서로 다른 방향으로 과대적합된 트리(DecisionTreeClassifier)를 많이 만들면, 그 결과를 평균냄으로써 과대적합된 양을 줄일 수 있다는 것임
  - `n_estimators`: 몇 개의 트리를 만들 것인가? -> 클수록 좋음
  - `max_features`: 클 수록 트리들이 비슷해지고, 작을수록 가장 두드러진 특성에 맞춰서 깊어짐
    + 분류 기본값 = `sqrt(n_features)`
    + 회귀 기본값 = `n_features`
  - `n_jobs`: cpu를 몇 개 사용할 것인가? -> -1로 하면 모두 사용
  - `random_state`: 랜덤시드값으로 이를 일관되게 지정해주어야 매번 다른 모델이 생성되지 않고 재현성을 보장할 수 있음
  - `oob_score`: 랜덤 샘플링에 포함되지 않은 샘플들을 기반으로 훈련 모델을 평가하게 되는 파라미터 (T/F)

<br/>

```
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)
print(gbc.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(gbc.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
* **GradientBoostingClassifier**
  - 그래디언트부스팅 분류기의 원리는 깊이가 얕은, 즉 약한 학습기를 여러 개 많이 연결하는 것이다. 이 또한 많이 추가할수록 성능이 좋아진다
  - 랜덤 포레스트보다는 매개변수 설정에 민감하지만, 잘 조정하면 정확도가 더 높아진다
  - `max_depth`: 기본값 3이고, 약한 학습기라는 정의와 목적에 맞게 최대한 5 이상을 넘어가지 않도록 함에 유의
  - `n_estimators`: 무작정 늘리는 것보다는 가용 범위에서 늘리고, 적절한 learning_rate를 찾는 것을 권장
  - `learning_rate`: 이전 트리의 오차를 보정하는 정도로, n_estimators와 깊은 상관관계가 있다

<br/>

### Classification Metric
분류 작업에서 사용할 수 있는 메트릭은 여러가지가 있는데, 이 또한 이진 분류와 다중 분류에 따라서 사용할 수 있는 메트릭이 달라진다
```
from sklearn.metrics import f1_score, classification_report

print(f1_score(y_test, y_pred)) # 이진 분류의 경우
print(f1_score(y_test, y_pred, average="옵션")) # 다중분류

classification_report(y_test, y_pred) # 여러 항목에 대한 성능 값을 한 번에 볼 수 있다
```
* **이진 분류**
  - 정확도(Accuracy)
  - 정밀도(Precision)
  - 재현율(Recall)
  - ROC
  - AUC
* **다중 분류**
  - `accuracy_score(y_test, y_pred)`
  - `classification_score(y_test, y_pred)`
  - `f1_score(y_test, y_pred, average='옵션')`
    + `average=micro`: 각 샘플을 똑같이 간주할 때
    + `average=weighted`: 클래스별 샘플 수로 가중치를 줄 때
    + `average=macro`: 각 클래스를 동일한 비중으로 간주할 때


  
