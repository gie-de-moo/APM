### Model Training and Tuning
library(doMC)
registerDoMC(cores=4)

## an example
library(caret)
library(mlbench)
data(Sonar)
str(Sonar[,1:10])
set.seed(998)
inTraining = createDataPartition(Sonar$Class, p = 0.75, list = F)
training = Sonar[inTraining,]
testing = Sonar[-inTraining,]

## basic parameter tuning
fitControl = trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats = 10)
set.seed(825)
gbmFit1 = train(Class ~ ., data = training,
                method = 'gbm',
                trControl = fitControl,
                verbose = F)
gbmFit1

## customizing tuning process
# alternate tuning grids
gbmGrid = expand.grid(interaction.depth = c(1,5,9),
                      n.trees = (1:30)*50,
                      shrinkage = 0.1)
nrow(gbmGrid)
set.seed(825)
gbmFit2 = train(Class~., data = training,
                method = 'gbm',
                trControl = fitControl,
                verbose = F,
                tuneGrid = gbmGrid)

# plotting the resampling profile
trellis.par.set(caretTheme())
# line plot of accuracy with each tunecombo
plot(gbmFit2)
# line plot of kappa with each tunecombo
plot(gbmFit2, metric='Kappa')
# heatmap of kappy with each tunecombo
plot(gbmFit2, metric='Kappa', plotType='level',
     scales=list(x=list(rot=90)))
# using ggplt
ggplot(gbmFit2)

# can used update.train() to refit the final model
?update.train

# alternate performance metrics
fitControl = trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats = 10,
                          classProbs = T,
                          summaryFunction = twoClassSummary)
set.seed(825)
gbmFit3 = train(Class ~., data=training,
                method = 'gbm',
                trControl = fitControl,
                verbose = F,
                tuneGrid = gbmGrid,
                metric = 'ROC')

# since in gbmFit3 performance across tunegrid is pretty close
# find simpler model as final model
whichTwoPct = tolerance(gbmFit3$results, metric='ROC', 
                        tol = 2, maximize = T)
cat('best model within 2 pct of best:\n')
gbmFit3$results[whichTwoPct, 1:6]

## extracting predictions and class probs
predict(gbmFit23, newdata=head(testing))
predict(gbmFit3, newdata =head(testing), type = 'prob')

## exploreing and comparing resampling distribution
#1 within model
#2 between models
set.seed(825)
svmFit = train(Class~., data=training,
               method = 'svmRadial',
               trControl = fitControl,
               preProc = c('center','scale'),
               tuneLength = 8,
               metric = 'ROC')

resamps = resamples(list(GBM = gbmFit3,
                         SVM = svmFit))
summary(resamps)

trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1))

trellis.par.set(caretTheme())
dotplot(resamps, metric = "ROC")

trellis.par.set(theme1)
xyplot(resamps, what = "BlandAltman")

splom(resamps)

# compute diff between models and use a t-test to evaluate
# whether there is indeed a diff between models
difValues = diff(resamps)
difValues
summary(difValues)

trellis.par.set(theme1)
bwplot(difValues, layout = c(3, 1))

trellis.par.set(caretTheme())
dotplot(difValues)