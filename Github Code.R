getwd() #Knowing the directory location
TrainData<-read.csv("qb.train.csv") # Setting the Training Environment
str(TrainData)
TestData<-read.csv("qb.test.csv") #Setting the Testing Environment
str(TestData)
install.packages("e1071")
install.packages("cwhmisc")
library(e1071) #Downloading the necessary packages
library(cwhmisc)

#The below code creates a function called paren_match that will try to check if the string that is inside the bracket in the Page column is present in the Text column. If yes then the value returned is TRUE else FALSE
paren_match<-function(page,text) {
  start<-cpos(page,"(")
  end<-cpos(page,")")  
  if(!is.na(start) && !is.na(end)) {
    take_string_inside_brackets<-substring(page,start+1,end-1)  
    return(grepl(tolower(take_string_inside_brackets),tolower(text),fixed=TRUE)) 
  }else{
    return(FALSE)
  }
}

TrainData$paren_match<-apply(TrainData,1,function(x){paren_match(x['page'],x['text'])}) #Applying the function

#As the variables obs_len, body_score and inlinks have high values we have to scale these variables
TrainData$obs_len <- apply(TrainData, 1, function(x) {nchar(x['text'])})
TrainData$scale_len <- scale(TrainData$obs_len)
TrainData$scale_score <- scale(TrainData$body_score)
TrainData$log_links <- scale(log(as.numeric(TrainData$inlinks) + 1))
str(TrainData)

head(TrainData) #Checking the head (first six rows) of the Training data after the changes made

index <- 1:nrow(TrainData) #splitting the data and working on the trainset data of the Train_data
testindex <- sample(index, trunc(length(index)/5))
testset <- TrainData[testindex,]
trainset <- TrainData[-testindex,]

#1. Working with Logistic Model

#The below code creates a function called Sfunction that checks the accuracy for each of Logistic Regression models we are trying to train in trainset and trying to apply in testset
Safunction<-function(Saatul)
{
  model1<-glm(corr~Saatul,family = binomial(link='logit'),data = trainset)
  trainset$prediction <- predict(model1,newdata=trainset,type='response')
  trainset$fitted.results <- ifelse(trainset$prediction > 0.5,1,0)
  
  trainset$corralter<-ifelse(trainset$corr=="True",1,0)
  misClasificError <- mean(trainset$fitted.results != trainset$corralter)
  Accuracy<-1-misClasificError
}

s<-c("body_score","scale_score","obs_len","obs_len+body_score","obs_len+paren_match","paren_match","scale_score+paren_match","scale_score+paren_match_scale_len","inlinks","log_links","scale_score+paren_match_scale_len+log_links") #Creating a vector s with the names of all the explanatory variables
a<-c(Safunction(trainset$body_score),Safunction(trainset$scale_score),Safunction(trainset$obs_len),Safunction(trainset$obs_len+trainset$body_score),Safunction(trainset$obs_len+trainset$paren_match),Safunction(trainset$paren_match),Safunction(trainset$scale_score+trainset$paren_match),Safunction(trainset$scale_score+trainset$paren_match+trainset$scale_len),Safunction(trainset$inlinks),Safunction(trainset$log_links),Safunction(trainset$scale_len+trainset$scale_score+trainset$paren_match+trainset$log_links)) #Applying the function Sfunction to get the accuracies 

Accuracy_table_trainset<-data.frame(s,a) #Creating a dataframe from the two vectors
colnames(Accuracy_table_trainset)<-c("Parameters","Accuracy")
Accuracy_table_trainset

#2. Working with Decision Trees
install.packages("rpart")
library(rpart)

#The below code creates a function called 'Decision Tree' that checks the accuracy of the decision tree models that we are trying to train in trainset and trying to apply in testset
decision_tree <- function(dframe, name, dmodel, testset) {
  decision <- predict(dmodel,newdata= testset,type="class")
  Chandler<-ifelse(decision!="False","TRUE","FALSE")
  confusionMatrix<- table(Chandler, true=testset$corr)
  accuracy<-sum(diag(confusionMatrix)/sum(confusionMatrix))
  dframe <- rbind(dframe, data.frame(model=c(name), accuracy=c(accuracy)))
  return(dframe)
}

#The below code trains the models and gets the accuracy

results1<- data.frame(matrix(ncol = 2, nrow = 0))

results1 <- decision_tree(results1, "body_score", rpart(corr ~ body_score,method="class",data=trainset), testset)
results1 <- decision_tree(results1, "scale_score", rpart(corr ~ scale_score,method="class",data=trainset), testset)
results1 <- decision_tree(results1, "obs_len", rpart(corr ~ obs_len,method="class",data=trainset), testset)
results1 <- decision_tree(results1, "score+len", rpart(corr ~ obs_len + body_score,method="class", data=trainset), testset)
results1 <- decision_tree(results1, "paren+len", rpart(corr ~ obs_len + paren_match,method="class", data=trainset), testset)
results1 <- decision_tree(results1, "paren_match", rpart(corr ~ paren_match,method="class", data=trainset), testset)
results1 <- decision_tree(results1, "score+paren_match", rpart(corr ~ scale_score + paren_match,method="class", data=trainset), testset)
results1 <- decision_tree(results1, "score+len+paren_match", rpart(corr ~ scale_len + scale_score + paren_match,method="class", data=trainset), testset)
results1 <- decision_tree(results1, "links", rpart(corr ~ inlinks,method="class", data=trainset), testset)
results1 <- decision_tree(results1, "loglinks", rpart(corr ~ log_links,method="class", data=trainset), testset)
results1 <- decision_tree(results1, "score+len+links+paren_match", rpart(corr ~ scale_len + scale_score + log_links + paren_match,method="class", data=trainset), testset)
results1 <- decision_tree(results1, "score+links+paren_match", rpart(corr ~ scale_len + scale_score + paren_match,method="class", data=trainset), testset)
results1 #The parameter with the highest accuracy in Decision Tree model is 'score+len+paren_match', 'score+len' and 'score+links+paren_match'.  Their accuracy is 0.7888545

#Working with SVM
#1. Radial SVM
SVM <- function(df, name, model, test) {
  svm.pred <- predict(model, test)
  svm.table <- table(pred = svm.pred, true=test$corr)
  df <- rbind(df, data.frame(model=c(name), score=c(classAgreement(svm.table)$diag)))
  return(df)
}

mfc_baseline <- sum(testset$corr == "False") / nrow(testset) #Getting the most frequent baseline
resultsSVM <- data.frame(model=c("MFC"), score=c(mfc_baseline))

#The below code applies the function created above

resultsSVM <- SVM(resultsSVM, "body_score", svm(corr ~ body_score, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "scale_score", svm(corr ~ scale_score, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "obs_len", svm(corr ~ obs_len, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "score+len", svm(corr ~ scale_len + obs_len, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "paren+obs_len", svm(corr ~ obs_len + paren_match, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "paren_match", svm(corr ~ paren_match, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "score+paren_match", svm(corr ~ scale_score + paren_match, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "score+len+paren_match", svm(corr ~ scale_len + scale_score + paren_match, data=trainset), testset)

resultsSVM <- SVM(resultsSVM, "loglinks", svm(corr ~ log_links, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "score+len+links+paren_match", svm(corr ~ scale_len + scale_score + log_links + paren_match, data=trainset), testset)
resultsSVM <- SVM(resultsSVM, "score+links+paren_match", svm(corr ~ scale_len + scale_score + paren_match, data=trainset), testset)
resultsSVM

#2. Linear Kernel
mfc_baseline <- sum(testset$corr == "False") / nrow(testset)
resultsSVMl <- data.frame(model=c("MFC"), score=c(mfc_baseline))

resultsSVMl <- SVM(resultsSVMl, "body_score", svm(corr ~ body_score, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "scale_score", svm(corr ~ scale_score, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "obs_len", svm(corr ~ obs_len, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "score+len", svm(corr ~ obs_len + body_score, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "paren+len", svm(corr ~ obs_len + paren_match, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "paren_match", svm(corr ~ paren_match, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "score+paren_match", svm(corr ~ scale_score + paren_match, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "score+len+paren_match", svm(corr ~ scale_len + scale_score + paren_match, data=trainset,kernel = 'linear'), testset)

resultsSVMl <- SVM(resultsSVMl, "loglinks", svm(corr ~ log_links, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "score+len+links+paren_match", svm(corr ~ scale_len + scale_score + log_links + paren_match, data=trainset,kernel = 'linear'), testset)
resultsSVMl <- SVM(resultsSVMl, "score+links+paren_match", svm(corr ~ scale_len + scale_score + paren_match, data=trainset,kernel = 'linear'), testset)
resultsSVMl


#3. Polynomial Kernel
mfc_baseline <- sum(testset$corr == "False") / nrow(testset)

resultsSVMp <- data.frame(model=c("MFC"), score=c(mfc_baseline))
resultsSVMp <- SVM(resultsSVMp, "body_score", svm(corr ~ body_score, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "scale_score", svm(corr ~ scale_score, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "obs_len", svm(corr ~ obs_len, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "score+len", svm(corr ~ obs_len + body_score, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "paren+len", svm(corr ~ obs_len + paren_match, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "paren_match", svm(corr ~ paren_match, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "score+paren_match", svm(corr ~ scale_score + paren_match, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "score+len+paren_match", svm(corr ~ scale_len + scale_score + paren_match, data=trainset,kernel = 'linear'), testset)

resultsSVMp <- SVM(resultsSVMp, "loglinks", svm(corr ~ log_links, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "score+len+links+paren_match", svm(corr ~ scale_len + scale_score + log_links + paren_match, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp <- SVM(resultsSVMp, "score+links+paren_match", svm(corr ~ scale_len + scale_score + paren_match, data=trainset,kernel = 'polynomial'), testset)
resultsSVMp

#4 Sigmoid Kernel
mfc_baseline <- sum(testset$corr == "False") / nrow(testset)
resultsSVMs <- data.frame(model=c("MFC"), score=c(mfc_baseline))

resultsSVMs <- SVM(resultsSVMs, "body_score", svm(corr ~ body_score, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "scale_score", svm(corr ~ scale_score, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "obs_len", svm(corr ~ obs_len, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "score+len", svm(corr ~ obs_len + body_score, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "paren+len", svm(corr ~ obs_len + paren_match, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "paren_match", svm(corr ~ paren_match, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "score+paren_match", svm(corr ~ scale_score + paren_match, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "score+len+paren_match", svm(corr ~ scale_len + scale_score + paren_match, data=trainset,kernel = 'sigmoid'), testset)

resultsSVMs <- SVM(resultsSVMs, "loglinks", svm(corr ~ log_links, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "score+len+links+paren_match", svm(corr ~ scale_len + scale_score + log_links + paren_match, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs <- SVM(resultsSVMs, "score+links+paren_match", svm(corr ~ scale_len + scale_score + paren_match, data=trainset,kernel = 'sigmoid'), testset)
resultsSVMs

#We come to know that among all models SVM model has the highest accuracy The SVM model that has the highest accuracy is Radial SVM model which is 0.8117647 that is of parameter score+len+links+paren_match


#Error Analysis of each of the models
#1. SVM model
#We chose SVM kernel to be Radial as it gave best result. Parameter selected is score+len+links+paren_match
SVMe <- predict(svm(corr ~ scale_len + scale_score + log_links + paren_match, data=trainset),testset)
tab1 <- table(Predicted = SVMe, Actual = testset$corr)
tab1 #Confusion Matrix

#The below code creates new variables for the type of the Error and Prediction Type

testsetdouble <- testset
SVM.new <-svm(corr ~ scale_len + scale_score + log_links + paren_match, data=trainset)
SVM.new.p<- predict(SVM.new,testsetdouble)
testsetdouble$predval <- SVM.new.p


#The below code creates new variables 'errorType' and predType'
for (i in 1:(nrow(testsetdouble)-1)) {
  if(testsetdouble$predval[i] == 'True' && testsetdouble$corr[i] == 'True')
  {
    testsetdouble$errorType[i] <- 'TP'
  }
  else if(testsetdouble$predval[i] == 'True' && testsetdouble$corr[i] == 'False')
  {
    testsetdouble$errorType[i] <- 'FP- Type I error'
  } 
  else if(testsetdouble$predval[i] == 'False' && testsetdouble$corr[i] == 'True')
  {
    testsetdouble$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    testsetdouble$errorType[i] <- 'TN'
  }
}


for (i in 1:(nrow(testsetdouble)-1)) {
  if(testsetdouble$errorType[i] == 'TP' || testsetdouble$errorType[i] == 'TN')
  {
    testsetdouble$predType[i] <- 'Correct Prediction'
  }
  else 
  {
    testsetdouble$predType[i] <- 'Wrong Prediction'
  } 
  
}

Typeoneerror<-testsetdouble[(testsetdouble$errorType=="FP- Type I error"),]

Typetwoerror<-testsetdouble[(testsetdouble$errorType=="FN - Type II error"),] #Trying to find the number of times FN-TYPE two error and FP-TYPE one error occurs for SVM
#272 occurences of Type Two and 32 occurences of Type 1


#2. Decision Tree
Decisione <- predict(rpart(corr ~ scale_len + scale_score + paren_match, method = 'class', data=trainset),testset, type = 'class')
tab2 <- table(Predicted = Decisione, Actual = testset$corr)
tab2

new.testset.decision <- testset
new.decision <-rpart(corr ~ scale_len + scale_score + paren_match, method = 'class', data=trainset)
new.p.decision <- predict(new.decision,new.testset.decision, type = 'class')
new.testset.decision$predval <- new.p.decision

#Creating new variables errorType and predType for this model
for (i in 1:(nrow(new.testset.decision)-1)) {
  if(new.testset.decision$predval[i] == 'True' && new.testset.decision$corr[i] == 'True')
  {
    new.testset.decision$errorType[i] <- 'TP'
  }
  else if(new.testset.decision$predval[i] == 'True' && new.testset.decision$corr[i] == 'False')
  {
    new.testset.decision$errorType[i] <- 'FP- Type I error'
  } 
  else if(new.testset.decision$predval[i] == 'False' && new.testset.decision$corr[i] == 'True')
  {
    new.testset.decision$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    new.testset.decision$errorType[i] <- 'TN'
  }
}

for (i in 1:(nrow(new.testset.decision)-1)) {
  if(new.testset.decision$errorType[i] == 'TP' || new.testset.decision$errorType[i] == 'TN')
  {
    new.testset.decision$predType[i] <- 'Correct Prediction'
  }
  else 
  {
    new.testset.decision$predType[i] <- 'Wrong Prediction'
  } 
  
}

#trying to find the number of times FN-TYPE two error and FP-TYPE one error occurs for Decision Tree
Typeoneerrordecision<-new.testset.decision[(new.testset.decision$errorType=="FP- Type I error"),]

Typetwoerrordecision<-new.testset.decision[(new.testset.decision$errorType=="FN - Type II error"),]
#43 occurences of type one and 322 occurences of type 2


#3. Logistic Regression
#Confusion Matrix 
new.testset.logistic<- (predict(glm(corr ~scale_score+ paren_match, family = binomial(link = 'logit'), data=trainset),testset, type = 'response'))>0.6
tab3 <- table(Predicted = new.testset.logistic, Actual = testset$corr)
tab3

new.testset.logistic1<- testset
new.lm <-glm(corr ~ scale_score + paren_match, family = binomial(link = 'logit'), data=trainset)
new.p.lm <- (predict(new.lm,new.testset.logistic1,type = 'response'))>0.6
new.testset.logistic1$predval <- new.p.lm

for (i in 1:(nrow(new.testset.logistic1)-1)) {
  if(new.testset.logistic1$predval[i] == 'TRUE' && new.testset.logistic1$corr[i] == 'True')
  {
    new.testset.logistic1$errorType[i] <- 'TP'
  }
  else if(new.testset.logistic1$predval[i] == 'TRUE' && new.testset.logistic1$corr[i] == 'False')
  {
    new.testset.logistic1$errorType[i] <- 'FP- Type I error'
  } 
  else if(new.testset.logistic1$predval[i] == 'FALSE' && new.testset.logistic1$corr[i] == 'True')
  {
    new.testset.logistic1$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    new.testset.logistic1$errorType[i] <- 'TN'
  }
}


for (i in 1:(nrow(new.testset.logistic1)-1)) {
  if(new.testset.logistic1$errorType[i] == 'TP' || new.testset.logistic1$errorType[i] == 'TN')
  {
    new.testset.logistic1$predType[i] <- 'Correct Prediction'
  }
  else 
  {
    new.testset.logistic1$predType[i] <- 'Wrong Prediction'
  } 
  
}

#Let's find the number of times FN-TYPE two error and FP-TYPE one error occurs for Logistic

Typeoneerrorlogistic<-new.testset.logistic1[(new.testset.logistic1$errorType=="FP- Type I error"),]

Typetwoerrorlogistic<-new.testset.logistic1[(new.testset.logistic1$errorType=="FN - Type II error"),]
#17 occurences of error type 1 and 488 occurences of error type 2


#Looking at the Error Type 
#For Type 1
Type_1_error<-data.frame(rbind("Logistic Regression","Decision Tree","SVM"))
Type_1_error$Error_occurences<-(rbind("17","43","32"))
names(Type_1_error)[names(Type_1_error) == 'rbind..Logistic.Regression....Decision.Tree....SVM..'] <- 'Error Model'


#For Type 2
Type_2_error<-data.frame(rbind("Logistic Regression","Decision Tree","SVM"))
Type_2_error$Error_occurences<-(rbind("488","322","272"))
names(Type_2_error)[names(Type_2_error) == 'rbind..Logistic.Regression....Decision.Tree....SVM..'] <- 'Error Model'
#SVM has least Type II error as compared to all other classifiers


#Let's try to plot this
#Installing the necessary libaries
install.packages("cowplot")
install.packages("ggplot2")
install.packages("rpart.plot")

#Downloading the libraries
library("ggplot2")
library("rpart.plot")
library("cowplot")

svm <-ggplot(data= testsetdouble, aes(x=predType, fill=errorType))+ geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 3)+geom_bar(position = 'dodge')+ ggtitle('SVM')

decision <- ggplot(data= new.testset.decision, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 3)+geom_bar(position = 'dodge')+ggtitle('Decision Tree')

logistic <- ggplot(data= new.testset.logistic1, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 3)+geom_bar(position = 'dodge')+ggtitle('Logistic Model')

plot_grid(svm,decision,logistic)

#In figure 1 
#Type II error for SVM is less as compared to other classifiers
#Total error for SVM is less as compared to others
#All three classifiers give maximum accuracy for different parameters. But it is seen that scale_score is common between all three classifiers

#SVM gives a good result but let's try to find a classfier that gives a much better result than this

install.packages("randomForest")
library(randomForest)

randomf <- function(dframe, name, dmodel, testset) {
  decision <- predict(dmodel,newdata= testset,type="class")
  Chandler<-ifelse(decision!="False","TRUE","FALSE")
  confusionMatrix<- table(Chandler, true=testset$corr)
  accuracy<-sum(diag(confusionMatrix)/sum(confusionMatrix))
  dframe <- rbind(dframe, data.frame(model=c(name), accuracy=c(accuracy)))
  return(dframe)
}
results12<- data.frame(matrix(ncol = 2, nrow = 0))

results12 <- randomf(results12, "body_score", randomForest(corr ~ body_score,data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "scale_score", randomForest(corr ~ scale_score,method="class",data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "obs_len", randomForest(corr ~ obs_len,data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "score+len", randomForest(corr ~ obs_len + body_score, data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "paren+len", randomForest(corr ~ obs_len + paren_match, data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "paren_match", randomForest(corr ~ paren_match, data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "score+paren_match", randomForest(corr ~ scale_score + paren_match, data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "score+len+paren_match", randomForest(corr ~ scale_len + scale_score + paren_match, data=trainset,proximty=TRUE), testset)

results12 <- randomf(results12, "loglinks", randomForest(corr ~ log_links, data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "score+len+links+paren_match", randomForest(corr ~ scale_len + scale_score + log_links + paren_match, data=trainset,proximty=TRUE), testset)
results12 <- randomf(results12, "score+links+paren_match", randomForest(corr ~ scale_len + scale_score + paren_match, data=trainset,proximty=TRUE), testset)
results12


#Let's understand this model

model1<-randomForest(corr~scale_len + scale_score + log_links + paren_match,data=trainset,proximty=TRUE)
model1


#Finding the parameter in Random Forest model with maximum accuracy

results12[which.max(results12$accuracy),]
#score+len+links+paren_match

#Error Analysis for Random Forest The parameters selected is score+len+links+paren_match Let's Create a confusion Matrix

e.p4 <- predict(randomForest(corr ~ scale_len + scale_score + log_links + paren_match, method = 'class', data= trainset),testset, type = 'class')
tab4 <- table(Predicted = e.p4, Actual = testset$corr)
tab4


new.testset.random <- testset
new.random <-randomForest(corr ~ scale_len + scale_score + log_links + paren_match, method = 'class', data= trainset,proximity=T)
new.p.random <- predict(new.random,new.testset.random)
new.testset.random$predval <- new.p.random

for (i in 1:(nrow(new.testset.random)-1)) {
  if(new.testset.random$predval[i] == 'True' && new.testset.random$corr[i] == 'True')
  {
    new.testset.random$errorType[i] <- 'TP'
  }
  else if(new.testset.random$predval[i] == 'True' && new.testset.random$corr[i] == 'False')
  {
    new.testset.random$errorType[i] <- 'FP- Type I error'
  } 
  else if(new.testset.random$predval[i] == 'False' && new.testset.random$corr[i] == 'True')
  {
    new.testset.random$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    new.testset.random$errorType[i] <- 'TN'
  }
}

for (i in 1:(nrow(new.testset.random)-1)) {
  if(new.testset.random$errorType[i] == 'TP' || new.testset.random$errorType[i] == 'TN')
  {
    new.testset.random$predType[i] <- 'Correct Prediction'
  }
  else 
  {
    new.testset.random$predType[i] <- 'Wrong Prediction'
  } 
  
}

ssrandomtype2<-new.testset.random[(new.testset.random$errorType=="FN - Type II error"),] #Number of type 2 error
#187
ssrandomtype1<-new.testset.random[(new.testset.random$errorType=="FP- Type I error"),]
#120


#For type 1 table
Type_1_error$`Error Model`<-as.character(Type_1_error$`Error Model`)
Type_1_error$Error_occurences<-as.character(Type_1_error$Error_occurences)

Type_1_error[4,1]<-"Random Forest"
Type_1_error[4,2]<-"120"


#For type 2 table
Type_2_error$`Error Model`<-as.character(Type_2_error$`Error Model`)
Type_2_error$Error_occurences<-as.character(Type_2_error$Error_occurences)
Type_2_error[4,1]<-"Random Forest"
Type_2_error[4,2]<-"187"


#We see that in Random Forest classifier the Type II error has decreased and the Type _I error has increased. The total error of SVM is near to the total error of Random forest Also, Type II error is more critical as ompared to Type I error which is decreased by using Random forest Classifier
#Let's plot this

svm1 <-ggplot(data= testsetdouble, aes(x=predType, fill=errorType))+ geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 3)+geom_bar(position = 'dodge')+ ggtitle('SVM')

decision1 <- ggplot(data= new.testset.decision, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 3)+geom_bar(position = 'dodge')+ggtitle('Decision Tree')

logistic1 <- ggplot(data= new.testset.logistic1, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 3)+geom_bar(position = 'dodge')+ggtitle('Logistic Model')

random1 <- ggplot(data= new.testset.random, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 3)+geom_bar(position = 'dodge')+ggtitle('Random Forest')


plot_grid(svm1,decision1,logistic1,random1)

#It can be seen that Random Forest not only decreases the type two error but also the TRUE positives have increased


#We see that by using Random Forest 1. The accuracy is almost equal to SVM 2. The Type Two error is the lowest. The overall error match with the overall error of SVM.so they both will have the lowest wrong predictions The True Positives of this model are the highest compared to other models
#So we will use this model for prediction

#Working with the Testing Data
#Since we have the accuracy of each parameters lets consider the parameter with the highest combination and then predict the original testdata using that parameter combination for which we have created a model
#We need all the columns which are in the Traiing data to be in the Testing Data

TestData$obs_len <- apply(TestData, 1, function(x) {nchar(x['text'])})
TestData$scale_len <- scale(TestData$obs_len)

TestData$scale_score <- scale(TestData$body_score)

TestData$paren_match <- apply(TestData, 1, function(x) {paren_match(x['page'], x['text'])})

TestData$log_links <- scale(log(as.numeric(TestData$inlinks) + 1))

#As scale_score+len+links+paren_match combination had the highest accuracy we will predict the test data using this combination

results4aprediction <- randomForest(corr ~ scale_score + paren_match +scale_len+log_links, data=TrainData,proximty=TRUE)
TestData$pred12<-predict(results4aprediction,newdata=TestData,type="class")

#created a new dataframe by taking only rowcolumn and pred12 column from the Testdata dataframe

pred_data<-data.frame(TestData$row,TestData$pred12)
colnames(pred_data)<-c("row","corr")
write.csv(pred_data,"prediction.csv",row.names = FALSE)
