#Developed by :POOJA A
#REGISTER NUMBER:22007907
import pandas as pd
from sklearn import linear_model

df=pd.read_csv("carsemission.csv")

x=df[["Weight","Volume"]]
y=df["CO2"]

regr=linear_model.LinearRegression()
regr.fit(x,y)

print("Coefficients:",regr.coef_)
print("Intercept:",regr.intercept_)

predictedCO2=regr.predict([[3300,1300]])
print("Predicted CO2 for the corresponding weight and volume",predictedCO2)