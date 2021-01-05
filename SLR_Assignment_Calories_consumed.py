# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:27:00 2020

@author: RAVI
"""

# For reading data set
# importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

CC = pd.read_csv("C:/RAVI/Data science/Assignments/Module 6 Simple linear regression/DataSets/calories_consumed.csv")

CC.columns="Weightgained","Caloriesconsumed"
CC
import matplotlib.pylab as plt #for different types of plots

plt.scatter(x=CC['Caloriesconsumed'], y=CC['Weightgained'],color='green')# Scatter plot

np.corrcoef(CC.Caloriesconsumed, CC.Weightgained) #correlation

help(np.corrcoef)

import statsmodels.formula.api as smf
plt.hist(CC["Weightgained"])
plt.hist(CC["Caloriesconsumed"])
model = smf.ols('Weightgained ~ Caloriesconsumed', data=CC).fit()
model.summary()

#values prediction
#Confidence interval Calculation
pred1 = model.predict(pd.DataFrame(CC['Caloriesconsumed']))
pred1
print (model.conf_int(0.95)) # 95% confidence interval

res = CC.Weightgained - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)

######### Model building on Transformed Data#############

# Log Transformation
# x = log(Caloriesconsumed); y = Weightgained
plt.scatter(x=np.log(CC['Caloriesconsumed']),y=CC['Weightgained'],color='brown')
np.corrcoef(np.log(CC.Caloriesconsumed), CC.Weightgained) #correlation

model2 = smf.ols('Weightgained ~ np.log(Caloriesconsumed)',data=CC).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(CC['Caloriesconsumed']))
pred2
print(model2.conf_int(0.95)) # 95% confidence level

res2 = CC.Weightgained - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)

# Exponential transformation
plt.scatter(x=CC['Caloriesconsumed'], y=np.log(CC['Weightgained']),color='orange')

np.corrcoef(CC.Caloriesconsumed, np.log(CC.Weightgained)) #correlation

model3 = smf.ols('np.log(Weightgained) ~ Caloriesconsumed',data=CC).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(CC['Caloriesconsumed']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.95)) # 95% confidence level

res3 = CC.Weightgained - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)

############Polynomial model with 2 degree (quadratic model)  ;x = Caloriesconsumed*Caloriesconsumed; y = Weightgained############
#### input=x & X^2 (2-degree); output=y  ####
model4 = smf.ols('Weightgained ~ Caloriesconsumed+I(Caloriesconsumed*Caloriesconsumed)', data=CC).fit()
model4.summary()

pred_p2 = model4.predict(pd.DataFrame(CC['Caloriesconsumed']))
pred_p2

print(model3.conf_int(0.95)) # 95% confidence level

res4 = CC.Weightgained - pred_p2
sqres4 = res4*res4
mse4 = np.mean(sqres4)
rmse4 = np.sqrt(mse4)

###########Polynomial model with 3 degree (quadratic model)  ;x = Caloriesconsumed*Caloriesconsumed*Caloriesconsumed; y = Weightgained############
#### input=x & X^2 (2-degree); output=y  ####
model5 = smf.ols('Weightgained ~ Caloriesconsumed+I(Caloriesconsumed*Caloriesconsumed)+I(Caloriesconsumed*Caloriesconsumed*Caloriesconsumed)', data=CC).fit()
model5.summary()

pred_p3 = model5.predict(pd.DataFrame(CC['Caloriesconsumed']))
pred_p3

print(model5.conf_int(0.95)) # 95% confidence level

res5 = CC.Weightgained - pred_p3
sqres5 = res5*res5
mse5 = np.mean(sqres5)
rmse5 = np.sqrt(mse5)

