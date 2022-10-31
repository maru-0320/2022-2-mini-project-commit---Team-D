import pandas as pd

# Load the CSV file
place = pd.read_csv('공공자전거 대여소 정보.csv', encoding ='cp949')

rental = pd.read_csv('bike_rental_all.csv', encoding = 'cp949', low_memory=False)

popular = pd.read_csv('주민등록인구.csv', encoding = 'cp949')

mode_of_t = pd.read_csv('이용 교통수단 비율.csv', encoding = 'cp949')



# 필요없는 행 삭제
rental = rental.drop(rental.index[1323516:])
place = place.drop(place.index[0:4])



# place의 대여소 번호와 자치구 열만 추출
place_new = place[["대여소 번호","자치구"]]



# rental의 대여소 번호와 이용건수 추출
rental_new = rental[["대여소 번호","이용건수"]]



# for문으로 대여소를 각 구별로 순회하고, 해당 자치구의 대여소 번호를 추출해서 같은 값을 갖는 rental_new의 행 데이터를 찾은 후,
# 이용건수를 더해 해당 구의 총 이용건수를 구함. 데이터프레임을 만들기 위해 리스트 생성.
  
p =["강남구","강동구","강북구","강서구","관악구","광진구","구로구","금천구","노원구",
"도봉구","동대문구","동작구","마포구","서대문구","서초구","성동구","성북구","송파구",
"양천구","영등포구","용산구","은평구","종로구","중구","중랑구"]

B = 0
l1 = []
l2 = []
for i in p:
    place_자치구= place_new[place_new["자치구"] == i]
    A_sum = 0
    for row in place_자치구.itertuples():
        place_number = row[1]
        A = rental_new[rental_new["대여소 번호"] == place_number]
        A_sum = A_sum + A["이용건수"].sum()
    B = B+A_sum
    l1.append(i)
    l2.append(A_sum)
    # print(i,A_sum)
# print(B)


# 위에서 생성한 리스트에 구한 값을 이용해서 데이터프레임 생성.
df = pd.DataFrame({"자치구" : l1, "따릉이 이용자 수" : l2})
df_new1 = df.to_string(index=False)



# 실제 csv의 이용건수의 합
R = rental_new["이용건수"].sum()
# print(R)

# 자치구별 이용건수를 구한 것과 실제 csv의 이용건수를 모두 더한값이 약 1프로정도 차이를 보이는데
# 이는 아마도 사라진 대여소도 있을 것이고 새로 생긴 대여소도 있기 때문인 듯함 (확실한 이유는 모르겠다.)



# 필요한 열,행만 추출, 인덱스 값 제거
popular_new1 = popular.drop(0)
popular_new2 = popular_new1[["자치구","합계 (명)"]]
popular_new3 = popular_new2.reset_index()


mode_of_t_new1 = mode_of_t[["자치구","자전거","버스","지하철(철도)"]]



# 데이터 프레임 합친 후, 중복 열 제거
df_new = pd.concat([popular_new3, mode_of_t_new1, df], axis = 1)
df_new2 = df_new.loc[:,~df_new.T.duplicated()]
df_new3 = df_new2.drop(["index"], axis = 1)
df_new3 = df_new3.astype({"합계 (명)": "int"}) # 합계 (명)의 dtypes가 object로 돼있으므로 int로 변환



# 인구 수 대비 따릉이 이용자수 비율화한 값을 데이터프레임에 추가
df_new3["따릉이 이용 비율"] = df_new3["따릉이 이용자 수"] / df_new3["합계 (명)"] 
# print(df_new3)


# 필요한 열만 추출
df_new4 = df_new3[["자전거", "버스", "지하철(철도)", "따릉이 이용 비율"]]



# 회귀분석을 위해서 열 이름을 영어로 바꿈
df_new4.columns = ["bicycle", "bus", "subway", "bike_rate"]
# print(df_new4)



## 분석 방법 1 - statsmodels 다중회귀분석 ##



# 정규화
def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

df_mean_norm = mean_norm(df_new4)



# 필요한 패키지 실행
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# print(df_mean_norm.head())



# 모델 만들기
model1 = ols('bike_rate ~ bicycle + bus + subway', df_mean_norm)
res = model1.fit()

# print(res.summary())

#  OLS Regression Results
# ==============================================================================
# Dep. Variable:              bike_rate   R-squared:                       0.222
# Model:                            OLS   Adj. R-squared:                  0.111
# Method:                 Least Squares   F-statistic:                     1.998
# Date:                Sat, 29 Oct 2022   Prob (F-statistic):              0.145
# Time:                        22:13:52   Log-Likelihood:                -37.837
# No. Observations:                  25   AIC:                             83.67
# Df Residuals:                      21   BIC:                             88.55
# Df Model:                           3
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     -0.8215      1.858     -0.442      0.663      -4.686       3.042
# bicycle        0.3247      0.173      1.877      0.075      -0.035       0.685
# bus            0.0491      0.054      0.902      0.377      -0.064       0.162
# subway         0.0826      0.057      1.442      0.164      -0.037       0.202
# ==============================================================================
# Omnibus:                        0.452   Durbin-Watson:                   2.123
# Prob(Omnibus):                  0.798   Jarque-Bera (JB):                0.574
# Skew:                           0.245   Prob(JB):                        0.751
# Kurtosis:                       2.443   Cond. No.                         229.
# ==============================================================================

# 모델의 R-squared값과 Adj. R-squared 값이 0.222와 0.111로 낮은 값이 나왔다. 
# 따라서 위 모델은 해당 회귀 모델을 10프로 수준으로밖에 설명하지 못하기 때문에 현실적으로 사용하기 힘들다.



## 분석 방법 2 - statsmodels 단순회귀분석 ##
Y = df_mean_norm["bike_rate"]
X = df_mean_norm["bus"]

X =sm.add_constant(X)
model2 = sm.OLS(Y, X)
results = model2.fit()
# print(results.params)
# print(results.rsquared)

# const    2.655914e-16
# bus      9.244024e-02
# dtype: float64
# 0.008545197110305547
# 0.8 %의 정확도..



## 분석 방법 3 - 상관 분석 ##
X3 = df_new4.bicycle.values
Y3 = df_new4.bike_rate.values

cov1 = (np.sum(X3 * Y3) - len(X) * np.mean(X3) * np.mean(Y3)) / len(X3) # 공분산 계산 방법1
print("cov1 =",cov1)

cov2 = np.cov(X3,Y3)[0,1] # 공분산 계산 방법2
print("cov2 =",cov2)

print(df_new4.corr()) # 피어슨 상관계수 이용.
sns.heatmap(df_new4.corr(), annot = True)
plt.show()

P_value = stats.pearsonr(X3, Y3)
print(P_value)
print("\n")

# bicycle과 bike_rate는 하나의 값이 상승하면 다른 값도 상승하는 경향을 보이는 
# 양의 상관관계를 가진다는 것을 공분산을 통해 알 수 있다.

# 피어슨 상관계수를 이용한 결과, bicycle은 0.36의 상관계수를 가져서 뚜렷한 상관관계를 가진다고 볼 수 있고,
# bus와 subway는 약하거나 또는 상관관계가 거의 없다고 볼 수 있다. 하지만 bicycle을 검정해 본 결과,
# pvalue > 0.05 였으므로 이는 상관계수 값 자체는 의미가 없다는 결과를 도출해낸다.


for item in ["bicycle", "bus", "subway"]:
    print(item)
    item_ = df_new4[item].values
    print("Covariance: ", np.cov(item_, Y3)[0,1])
    print("Correlation: ", stats.pearsonr(item_,Y3)[0])
    print("P-value: ", stats.pearsonr(item_,Y3)[1])
    print('\n')