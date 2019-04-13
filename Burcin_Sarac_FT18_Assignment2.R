library(tidyverse)
library(readxl)
library(car)
library(MASS)
library(nnet)
library(class)
library(tree)
library(rpart)
library(rpart.plot)
library(ranger)
library(corrplot)
library(caret)
library(klaR)
library(Metrics)
library(mclust)
library(vtreat)


setwd("E:/dersler/Statistics 2/Assignment 1/")
churn <- read_excel("churn.xls")
str(churn)
summary(churn)
sapply(churn, function(x) sum(is.na(x)))

churn$`VMail Plan` <- as.factor(churn$`VMail Plan`)
churn$`Int'l Plan` <- as.factor(churn$`Int'l Plan`)
churn$Churn <- as.factor(churn$Churn)
churn$Gender <- factor(churn$Gender, levels=c("Male","Female"), labels=c(0,1))
str(churn)

colnames(churn) <- c( "Account_Length","VMail_Message","Day_Mins",
"Eve_Mins", "Night_Mins", "Intl_Mins", "CustServ_Calls", "Churn" ,        
"Intl_Plan","VMail_Plan", "Day_Calls", "Day_Charge","Eve_Calls" ,
"Eve_Charge","Night_Calls","Night_Charge" ,"Intl_Calls",
"Intl_Charge", "State","Area_Code" ,  "Gender" )

churn$State <- as.factor(churn$State)
churn$Area_Code <- as.factor(churn$Area_Code)


numchurn <- churn[,sapply(churn, is.numeric)]

##Pairwise Comparisons

for (i in 1:ncol(numchurn)){
  print(ggplot(churn, aes(churn[[(names(numchurn)[i])]], Churn))+
    geom_jitter() + xlab(colnames(numchurn[i])) + 
    ggtitle(paste("Figure :",colnames(numchurn[i]),"vs. Churn")))
}


for (i in 1:ncol(numchurn)){
  print(ggplot(churn, aes(churn[[(names(numchurn)[i])]]))+
        geom_bar()+facet_wrap(~Churn, ncol = 3) +
        xlab(colnames(numchurn[i])) + 
        ggtitle(paste("Figure :",colnames(numchurn[i]),"vs. Churn")))
  }


par(mfrow=c(1,2))
plot(table(churn$Churn,churn$`Intl_Plan`),xlab="Churn",ylab="Int'l Plan")
plot(table(churn$Churn,churn$`VMail_Plan`),xlab="Churn",ylab="VMail Plan")
plot(table(churn$Gender,churn$Churn),ylab="Churn",xlab="Genders: 0=M & 1=F")

#probability of customer churn
round(table(churn$Churn)/length(churn$Churn),2)

par(mfrow=c(1,1))
####check for correlation
round(cor(churn[sapply(churn, is.numeric)], y=as.numeric(churn$Churn)),2)

corrplot(cor(churn[sapply(churn, is.numeric)]))


########Random Forest################
set.seed(2019)
modelrf1 <- ranger(formula= Churn~., data= churn, num.trees = 200, 
                   classification = TRUE)  
predrf1 <- predict(modelrf1, churn)
accuracy(churn$Churn, modelrf1$predictions)
#accuracy 0.9522952
confusionMatrix(table(churn$Churn, modelrf1$predictions))
#      0    1
# 0 2826   24
# 1  135  348

########Classification Trees#####################
set.seed(2019)
modelct <- rpart(Churn~., data=churn, method = "class")
predct <- predict(modelct, churn, type="class")
rpart.plot(modelct, type=3, box.palette=c("red","green"), 
           fallen.leaves=TRUE)
accuracy(churn$Churn, predct)
# accuracy 0.9507951
table(churn$Churn, predct)
#      0    1
# 0 2815   35
# 1  129  354
plotcp(modelct)

##pruning tree######

modelct_pruned <- prune(modelct, cp=0.035)
plotcp(modelct_pruned)
rpart.plot(modelct_pruned,type=3,box.palette=c("green","red"))
predct_pruned <- predict(modelct_pruned, churn, type="class")
accuracy(churn$Churn, predct_pruned)
# accuracy 0.9312931
confusionMatrix(table(churn$Churn, predct_pruned))
#      0    1
# 0 2800   50
# 1  179  304


##According to our tree the most important variables are
# Day Mins, CustServ Calls, Vmail Message, INtl PLan, 
# Intl Calls, Intl Mins, so I pick only these variables and run
# a tree again 


churn2 <- churn[c("Churn","Intl_Plan", "VMail_Message", "Day_Mins", 
                  "CustServ_Calls", "Intl_Calls","Intl_Mins")]
#### split train test data
set.seed(2019)
sample_rows <- sample(nrow(churn2), nrow(churn2)*0.70)
train <- churn2[sample_rows, ]
test <- churn2[-sample_rows,]

######### decision(classification) tree ############

modeldt <- rpart(Churn~., data=train, method = "class")
preddt <- predict(modeldt, test, type="class")
rpart.plot(modeldt, type=3, box.palette=c("green","red"), 
           fallen.leaves=TRUE)
table(test$Churn, preddt)
#     0   1
# 0 838  19
# 1  57  86
preddt2 <- predict(modeldt, churn, type="class")
summary(preddt2)
#    0    1 
# 2979  354 
accuracy(churn$Churn, preddt2)
# accuracy 0.9312931
confusionMatrix(table(churn$Churn, preddt2))
#      0    1
# 0 2800   50
# 1  179  304

#################################################
##### Predictions via Cross Validation #####
##########################################

train_control <- trainControl(method="repeatedcv", number=10, 
                              repeats = 3)



################  naive bayes #####################3##
set.seed(2019)
modelnb <- train(Churn~., data=churn, 
                 trControl=train_control, method="nb")
modelnb$bestTune
# fL usekernel adjust
# 0      TRUE      1
print(modelnb)

prednb <- predict(modelnb, churn)
confusionMatrix(table(churn$Churn, prednb))
###   prednb
#      0    1
# 0 2850    0
# 1  483    0
accuracy(churn$Churn, prednb)
#0.8829883

##dealing with Naive Bayes with balanced sampling
set.seed(2019)
for (i in 1:10){
  balancedsampling <- rbind(subset(churn, churn$Churn==0)
                          [sample(1:483), ], 
                          subset(churn, churn$Churn==1))
  modelnb2 <- train(Churn~., data=balancedsampling, 
                   trControl=trainControl(method="cv", number=10), 
                   method="nb")
}
prednb2 <- predict(modelnb2, churn)
print(modelnb2)
confusionMatrix(table(churn$Churn, prednb2))
#      0    1
# 0 2646  204
# 1  186  297
accuracy(churn$Churn, prednb2)
# accuracy 0.8829883


##dealing with Naive Bayes with separate sampling

for (i in 1:10){
  separatesampling <- rbind(subset(churn, churn$Churn==0)
                            [sample(1:483*(7/3)), ], 
                            subset(churn, churn$Churn==1))
  modelnb3 <- train(Churn~., data=separatesampling, 
                    trControl=trainControl(method="cv", number=10), 
                    method="nb")
}
prednb3 <- predict(modelnb3, churn)
print(modelnb3)
confusionMatrix(table(churn$Churn, prednb3))
#      0    1
# 0 2444  406
# 1  134  349
accuracy(churn$Churn, prednb3)
# accuracy 0.8379838

##################  SVM #####################
set.seed(2019)
# train the model
modelsvm <- train(Churn~., data=churn, 
                  trControl=train_control, method="svmLinearWeights")
# summarize results
modelsvm$bestTune
#   cost weight
#   0.25      1
print(modelsvm)
## accuracy  0.8550871
predsvm <- predict(modelsvm, churn)
confusionMatrix(table(churn$Churn, predsvm))
#      0    1
# 0 2850    0
# 1  483    0
accuracy(churn$Churn,predsvm)
# accuracy 0.8550855

##dealing with SVM with balanced sampling
set.seed(2019)
for (i in 1:10){
  balancedsamplingSVM <- rbind(subset(churn, churn$Churn==0)
                               [sample(1:483), ], 
                               subset(churn, churn$Churn==1))
  modelsvm2 <- train(Churn~., data=balancedsamplingSVM, 
                     trControl=trainControl(method="cv", number=10), 
                     method="svmLinearWeights")
}
print(modelsvm2)
predsvm2 <- predict(modelsvm2, churn)
confusionMatrix(table(churn$Churn, predsvm2))
#      0    1
# 0 2162  688
# 1  107  376
accuracy(churn$Churn, predsvm2)
## accuracy  0.7614761


##dealing with SVM with separate sampling
set.seed(2019)
for (i in 1:10){
  separatesamplingSVM <- rbind(subset(churn, churn$Churn==0)
                            [sample(1:483*(7/3)), ], 
                            subset(churn, churn$Churn==1))
  modelsvm3 <- train(Churn~., data=separatesamplingSVM, 
                    trControl=trainControl(method="cv", number=10), 
                    method="svmLinearWeights")
}
print(modelsvm3)
predsvm3 <- predict(modelsvm3, churn)
confusionMatrix(table(churn$Churn, predsvm3))
#      0    1
# 0 2164  686
# 1  110  373
accuracy(churn$Churn, predsvm3)
## accuracy  0.7611761


########### random forest with CV ############
set.seed(2019)
train_control_forrf <- trainControl(method="cv", number=10)
modelrf <- train(Churn~., data=churn, 
                 trControl=train_control_forrf, method="ranger")
modelrf$bestTune
# mtry  splitrule min.node.size (mtry=Randomly Selected Predictors)
# 6   36   gini             1
print(modelrf)
predrf <- predict(modelrf, churn)
confusionMatrix(table(churn$Churn, predrf))
#      0    1
# 0 2850    0
# 1    0  483
accuracy(churn$Churn, predrf)
#1
predictors(modelrf)



churn2 <- churn <- subset(churn, select = -c(State))
set.seed(2019)
modelrf2 <- train(Churn~., data=churn2, 
                 trControl=train_control_forrf, method="ranger")
modelrf2$bestTune
# mtry  splitrule min.node.size (mtry=Randomly Selected Predictors)
# 6   36   gini             1
print(modelrf2)
predrf2 <- predict(modelrf2, churn2)
confusionMatrix(table(churn$Churn, predrf2))
#      0    1
# 0 2850    0
# 1    0  483
accuracy(churn$Churn, predrf2)
#1
predictors(modelrf2)

##############  knn ##################
# train the model
set.seed(2019)
grid <- expand.grid(k=c(1,2,3,4,5,10,20,50,100))
modelknn <- train(Churn~., data=churn, 
                  trControl=train_control, method="knn",
                  tuneGrid=grid)
# summarize results
modelknn$bestTune
# k=10
print(modelknn)
predknn <- predict(modelknn, churn)
confusionMatrix(table(churn$Churn, predknn))
#      0    1
# 0 2826   24
# 1  339  144
accuracy(churn$Churn, predknn)
## accuracy 0.8910891


#####################################################
################CLUSTERING##########################


#####Picking variables, get dummies and scaling

churnclus <- subset(churn, select = c(State, Intl_Plan, Area_Code,
                                      Gender,Account_Length,Day_Mins,
                                      Eve_Mins,Intl_Mins,VMail_Plan,
                                      Eve_Calls,Day_Calls,Intl_Calls))
for (i in 1:ncol(churnclus)){
  if(sapply(churnclus[i], is.numeric)==TRUE){
    churnclus[i] <- scale(churnclus[i])
  }
}

dummychurnclus <- dummyVars(~.,churnclus)
dummychurnclus
# Dummy Variable Object
# Formula: ~.
# 12 variables, 5 factors
# Variables and levels will be separated by '.'
# A less than full rank encoding is used

dumchurn <- predict(dummychurnclus, churnclus)

############`Kmeans###################`
set.seed(2019)
tss <- 0
for(i in 1:15){
  modelkmeans <- kmeans(dumchurn, centers = 15, nstart = 20,
                        iter.max = 20)
  tss[i] <- modelkmeans$tot.withinss
} 
plot(1:15, tss, type = "b", xlab = "Nr of Clusters")

modelkmeans2 <- kmeans(dumchurn, 3, trace = TRUE)
modelkmeans2$totss
# 32262.16
modelkmeans2$betweenss
# 3823.612
modelkmeans2$centers
table(modelkmeans2$cluster)
#    1    2    3 
# 1134 1152 1047 

churnclustered <- cbind(churn, cluster= modelkmeans2$cluster)
churnclustered$cluster <- as.factor(churnclustered$cluster)
summary(churnclustered[churnclustered$cluster==1,])
summary(churnclustered[churnclustered$cluster==2,])
summary(churnclustered[churnclustered$cluster==3,])

for(i in 1:ncol(churnclus)){
  if(sapply(churnclus[i], is.numeric)==TRUE){
    print(ggplot(churnclustered, aes(cluster, 
                                   churn[[(names(churnclustered)[i])]]))+
        geom_boxplot() + ylab(names(churnclustered)[i]))
  }
}

plot(table(churnclustered$cluster, churnclustered$Intl_Plan),
     xlab = "cluster",ylab = "Intl_Plan")
plot(table(churnclustered$cluster, churnclustered$VMail_Plan),
     xlab = "cluster",ylab = "VMail_Plan")
plot(table(churnclustered$cluster, churnclustered$Gender),
     xlab = "cluster",ylab = "Gender: M=0, F=1")

########## Hierarchical Clustering###################
set.seed(2019)
dist_matrix <- dist(dumchurn)
modelhc <- hclust(dist_matrix)
plot(modelhc, ylab="Dist.btw Clusters")
abline(h=10, col="red")
clustershc <- cutree(modelhc, h=10)
table(clustershc)
#    1    2    3    4 
# 3262    9   53    9 
 
clustershc <- cutree(modelhc, k=3)
table(clustershc)
#    1    2    3 
# 3262   62    9 

set.seed(2019)
modelhc_wd <- hclust(dist_matrix, method="ward")
plot(modelhc_wd)
abline(h=200, col="red")
clustershc_wd <- cutree(modelhc_wd, k=3)
table(clustershc_wd)
#    1    2    3 
# 2025  979  329 

clustershc_wd <- cutree(modelhc_wd, k=4)
table(clustershc_wd)
#   1    2    3    4 
# 881 1144  979  329 

clustershc_wd <- cutree(modelhc_wd, k=8)
table(clustershc_wd)
#   1   2   3   4   5   6   7   8 
# 586 438 566 413 295 329 401 305


modelkmeans3 <- kmeans(dumchurn, 8, trace = TRUE)
table(modelkmeans3$cluster)
#   1   2   3   4   5   6   7   8 
# 423 437 383 321 441 449 403 476
