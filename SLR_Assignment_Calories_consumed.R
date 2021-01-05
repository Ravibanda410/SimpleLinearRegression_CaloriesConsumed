#  Simple Linear Regression Assignment #
#  1) Calories_consumed-> predict weight gained using calories consumed
#  Do the necessary transformations for input variables for getting better R^2 value for the model prepared.

library(readr)
library(ggplot2)
library(moments)
CC <- read_csv("C:/RAVI/Data science/Assignments/Module 6 Simple linear regression/DataSets/calories_consumed.csv")
View(CC)
attach(CC)
colnames(CC) <- c("Weightgained","Caloriesconsumed") 
View(CC)
summary(CC)
range(CC$Caloriesconsumed)
range(CC$Weightgained)
skewness(CC$Weightgained)
skewness(CC$Caloriesconsumed)

#Exploratory Data Analysis
boxplot(CC$Weightgained)
boxplot(CC$Caloriesconsumed)

#scatter plot for Caloriesconsumed vs Weightgained (Plot x,y)
plot(CC$Caloriesconsumed,CC$Weightgained)

#calculate correlation coefficient
cor(CC$Caloriesconsumed,CC$Weightgained)

#Simple Regression model
reg <- lm(CC$Weightgained~CC$Caloriesconsumed,data = CC)
summary(reg)

#values prediction
#Confidence interval Calculation
confint(reg,level = 0.95)
pred <- predict(reg,interval = "predict")
#predict function gives fit value and its lower and upeer values as a range
pred <- as.data.frame(pred)
pred

#####Plot Graph for both Actual values and also the predicted linear Graph(Actual:Red,Predicted:Blue)#########
ggplot() + 
  geom_point(aes(x =CC$Caloriesconsumed , y =CC$Weightgained ),
             colour='red') + 
  geom_line(aes(x = CC$Caloriesconsumed, y = predict(reg, newdata=CC)),
            colour='blue') + 
  ggtitle('Caloriesconsumed vs Weightgained') +
  xlab('Caloriesconsumed') +
  ylab('Weightgained')

cor(pred$fit,CC$Weightgained)

#Calculate Residuals "Errors"
reg$residuals
reg$residuals^2
mean(reg$residuals^2)
rmse <- sqrt(mean(reg$residuals^2))
rmse

############ Applying transformations##############
############ lOGORITHMIC MODEL    x = log(Caloriesconsumed); y = Weightgained ############
plot(log(CC$Caloriesconsumed),CC$Weightgained)
cor(log(CC$Caloriesconsumed),CC$Weightgained)

log_reg <- lm(CC$Weightgained ~ log(CC$Caloriesconsumed),data = CC)
summary(log_reg)

#values prediction
#Confidence interval Calculation
confint(log_reg,level = 0.95)
pred_log <- predict(log_reg,interval ="predict")
#predict function gives fit value and its lower and upeer values as a range
pred_log <- as.data.frame(pred_log)
pred_log


cor(pred_log$fit,CC$Weightgained)

rmse_log <- sqrt(mean(log_reg$residuals^2)) 
rmse_log
######or####
res_log=CC$Weightgained-pred_log$fit
rmse_log<-sqrt(mean(res_log^2))
rmse_log
##########Plot Graph for both Actual values and also the predicted linear Graph(Actual:Red,Predicted:Blue)#########
ggplot() + 
  geom_point(aes(x =CC$Caloriesconsumed , y =CC$Weightgained ),
             colour='red') + 
  geom_line(aes(x = CC$Caloriesconsumed, y = predict(log_reg, newdata=CC)),
            colour='blue') + 
  ggtitle('Caloriesconsumed vs Weightgained') +
  xlab('Caloriesconsumed') +
  ylab('Weightgained')

############ EXPONENTIAL MODEL   x = Caloriesconsumed; y = log(Weightgained) ############
plot(CC$Caloriesconsumed,log(CC$Weightgained))
cor(CC$Caloriesconsumed,log(CC$Weightgained))

log_reg2 <- lm(log(CC$Weightgained) ~ CC$Caloriesconsumed,data = CC)
summary(log_reg2)

#values prediction
#Confidence interval Calculation
confint(log_reg2,level = 0.95)
pred_log2 <- predict(log_reg2,interval ="predict")
#predict function gives fit value and its lower and upeer values as a range
pred_log2 <- as.data.frame(pred_log2)

log_reg2$residuals #output is log(AT) so we are getting less values apply antilog
pred<- exp(pred_log2)  #anti-log=exponential
pred

cor(pred$fit,CC$Weightgained)

res_log2=CC$Weightgained-pred$fit
rmse2 <- sqrt(mean(res_log2^2))
rmse2


##########Plot Graph for both Actual values and also the predicted linear Graph(Actual:Red,Predicted:Blue)#########
ggplot() + 
  geom_point(aes(x =CC$Caloriesconsumed , y =CC$Weightgained ),
             colour='red') + 
  geom_line(aes(x = CC$Caloriesconsumed, y = predict(log_reg2,data=CC)),
            colour='blue') + 
  ggtitle('Caloriesconsumed vs Weightgained') +
  xlab('Caloriesconsumed') +
  ylab('Weightgained')



############Polynomial model with 2 degree (quadratic model)  ;x =Caloriesconsumed^2 ; y = Weightgained ############
#### input=x & X^2 (2-degree); output=y  ####
reg_quad2<- lm(CC$Weightgained ~ CC$Caloriesconsumed+I(CC$Caloriesconsumed*CC$Caloriesconsumed),data =CC)
summary(reg_quad2)

#prediction
#Confidence interval Calculation
confint(reg_quad2,level = 0.95)
pred_quad2<-predict(reg_quad2,interval = "predict")
pred_quad2  <- as.data.frame(pred_quad2)
pred_quad2

resq=CC$Weightgained-pred_quad2$fit
rmse_quad<-sqrt(mean(resq^2))
rmse_quad

cor(pred_quad2$fit,CC$Weightgained)
##########Plot Graph for both Actual values and also the predicted linear Graph(Actual:Red,Predicted:Blue)#########
ggplot() + 
  geom_point(aes(x =CC$Caloriesconsumed , y =CC$Weightgained ),
             colour='red') + 
  geom_line(aes(x = CC$Caloriesconsumed, y = predict(reg_quad2,data=CC)),
            colour='blue') + 
  ggtitle('Caloriesconsumed vs Weightgained') +
  xlab('Caloriesconsumed') +
  ylab('Weightgained')

############Polynomial model with 3 degree (quadratic model)  ;x = Caloriesconsumed^3; y = Weightgained ############
#### input=x & X^2 & x^3 (3-degree); output=y  ####
reg_quad3<- lm(CC$Weightgained ~ CC$Caloriesconsumed+I(CC$Caloriesconsumed*CC$Caloriesconsumed)+I(CC$Caloriesconsumed*CC$Caloriesconsumed*CC$Caloriesconsumed),data =CC)
summary(reg_quad3)

#prediction
#Confidence interval Calculation
confint(reg_quad3,level = 0.95)
pred_quad3<-predict(reg_quad3,interval = "predict")
pred_quad3  <- as.data.frame(pred_quad3)
pred_quad3

cor(pred_quad3$fit,CC$Weightgained)

resq3=CC$Weightgained-pred_quad3$fit
rmse_quad3<-sqrt(mean(resq3^2))
rmse_quad3
##########Plot Graph for both Actual values and also the predicted linear Graph(Actual:Red,Predicted:Blue)#########
ggplot() + 
  geom_point(aes(x =CC$Caloriesconsumed , y =CC$Weightgained ),
             colour='red') + 
  geom_line(aes(x = CC$Caloriesconsumed, y = predict(reg_quad3,data=CC)),
            colour='blue') + 
  ggtitle('Caloriesconsumed vs Weightgained') +
  xlab('Caloriesconsumed') +
  ylab('Weightgained')








