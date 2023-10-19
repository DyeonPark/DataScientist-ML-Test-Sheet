# 1. Task에 따른 분류
본 페이지에서는 Task에 따라서 머신러닝 중 지도학습 방법을 나누어 설명한다
지도학습이란 입력과 출력 샘플 데이터가 있고, 주어진 입력으로부터 출력을 예측하고자 할 때 사용한다
이의 궁극적인 목표는 이전에 본적이 없는 새로운 데이터에 대해 정확한 예측 및 출력을 하기 위함이다

그리고 지도학습에는 대표적인 Task로 Regression(회귀)와 Classification(분류)가 있다

## Regression
### 대표적인 모델
* **LinearRegression**

대표적인 선형 모델 중 하나인 LinearRegression
```
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

print(linear_reg.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(linear_reg.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```

구간 나누기를 통해서 성능을 향상시킬 수 있으며, 원본 특성에 다항식을 추가하는 등의 방법을 사용할 수 있음
단, 이 방법을 사용할 때 학습 시간이 매우 오래 걸릴 수 있음에 주의해야 함
```
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
poly.fit(X_train)
X_poly = poly.transform(X_train) # 다항식 추가

linear_reg.fit(X_train, y_train)

print(linear_reg.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(linear_reg.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
<br/>

* **SVR (Support Vector Regression)**

Regression을 위한 Support Vector Machine 
```
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)

print(svr.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(svr.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```
<br/>

* **RandomForestRegressor**

앙상블의 일종인 Random Forest를 사용한 Regressor
```
from sklearn.ensemble import RandomForestRegressor

# 주요 파라미터
# max_iteration
# max_depth
# random_state
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

print(rfr.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(rfr.score(X_test, y_test)) # 평가 데이터셋 성능 결과

```
<br/>

* **GradientBoostingRegressor**

앙상블의 일종인 GradientBoosting을 사용한 Regressor

```
from sklearn.ensemble import GradientBoostingRegressor

# 주요 파라미터
# max_iteration
# max_depth
# random_state
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

print(gbr.score(X_train, y_train)) # 학습 데이터셋 성능 결과
print(gbr.score(X_test, y_test)) # 평가 데이터셋 성능 결과
```

### Regression Metric
회귀 모델에서 사용할 수 있는 주요 메트릭은 크게 3가지가 있다
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* r2-score
```
from sklearn.metrics import mean_squared_error, r2_score

print(mean_squared_error(y_test, y_pred)) # MSE
print(round(mean_squared_error(y_test, y_pred) ** 0.5, 3)) # 반올림 적용한 RMSE
print(r2_score(y_test, y_pred)) # 위 에러와 달리 1에 가까울 수록 좋은 값
```


## Classification

