#Loading the dataset
df <- read.csv("student-mat.csv", sep = ";")
print(dim(df))
head(df)

#preprocessing 
#Target Variable: G3
#Predictors: All excpet G1 and G2
categorical_vars <- c("school", "sex", "address","famsize","Pstatus","Mjob","Fjob","reason","guardian",
                      "schoolsup","famsup","paid","activities","nursery", "higher","internet","romantic")  

df[, categorical_vars] <- lapply(df[, categorical_vars], as.factor)
df <- df[, !(names(df) %in% c("g1", "g2"))]
df

#Define null and full models
null_model <- lm(G3 ~ 1 , data = df)
full_model <- lm(G3 ~ ., data = df)

#Perform backward AIC and BIC
aic <- step(full_model, direction = "backward")
bic <- step(full_model, direction = "backward", k=log(n))
aic
bic

aic_model <- lm(formula = G3 ~ sex + age + famsize + Medu + Mjob + studytime + 
                  failures + schoolsup + famsup + romantic + freetime + goout + 
                  absences, data = df)
bic_model <-lm(formula = G3 ~ Medu + failures, data = df)
#Compare both AIC and BIC models
anova(bic_model,aic_model)
adj_r_squared_aicmodel <- summary(aic_model)$adj.r.squared
adj_r_squared_bicmodel <- summary(bic_model)$adj.r.squared
cat(adj_r_squared_aicmodel,adj_r_squared_bicmodel)
first_final_model <-aic_model
summary(first_final_model)

#Use AIC to find the best model with account for interactions
full_model <- lm(G3 ~ sex * age * famsize  * Mjob * romantic * failures * schoolsup 
                 * famsup * freetime * goout * 
                   studytime * absences, data = df)
aic_full_model <- step(null_model, scope = formula(full_model), direction = "both")
summary(aic_full_model)
final_model <- aic_full_model

#Check if any of the assumptions are violated 
plot(fitted(final_model), resid(final_model), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residual",cex=2,
     main = "Residual plot")
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(final_model), col = "grey",pch=20,cex=2)
qqline(resid(final_model), col = "dodgerblue", lwd = 2)
library(lmtest)
bptest(final_model)
shapiro.test(resid(final_model))

#Add cube-root transformation to the model 
final_model <- lm((G3-0.00001)^(1/3) ~ failures + Mjob + sex + goout + romantic + 
                    absences + schoolsup + studytime +studytime + famsup + age + failures:absences + 
                    failures:Mjob + failures:schoolsup + failures:goout + schoolsup:studytime^2 + 
                    studytime:famsup + failures:age + famsup:age + sex:romantic, 
                  data = df)

#Check if assumptions hold
plot(fitted(final_model), resid(final_model), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residual",cex=2,
     main = "Residual plot")
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(final_model), col = "grey",pch=20,cex=2)
qqline(resid(final_model), col = "dodgerblue", lwd = 2)
bptest(final_model)
shapiro.test(resid(final_model))

#summary of the final model 
summary(final_model)

