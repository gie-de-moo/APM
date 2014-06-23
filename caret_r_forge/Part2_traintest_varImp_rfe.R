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

### variable importance
## within model, varImp estimate
gbmImp = varImp(gbmFit3, scale=F)
gbmImp

## regardless of model, rough estimates of varImp
# get auc for each predictor
RocImp = filterVarImp(x = training[,-ncol(training)],
                      y = training$Class)
head(RocImp)

## models like svm having no built-in importance score
## implemented or existing. We can still use varImp for 
## non-model based estiamtes
RocImp2 = varImp(svm Fit, scale=F)
RocImp2

plot(gbmImp, top=20)

### feature selection
## recursive feature elimination when there is
## predeined rfe algo for a model type, eg lm
library(caret)
library(mlbench)
library(Hmisc)
library(randomForest)

# simulate data sets
n <- 100
p <- 40
sigma <- 1
set.seed(1)
sim <- mlbench.friedman1(n, sd = sigma)
colnames(sim$x) <- c(paste("real", 1:5, sep = ""),
                     paste("bogus", 1:5, sep = ""))
bogus <- matrix(rnorm(n * p), nrow = n)
colnames(bogus) <- paste("bogus", 5+(1:ncol(bogus)), sep = "")
x <- cbind(sim$x, bogus)
y <- sim$y

# center and scale x
normalization = preProcess(x)
x = predict(normalization, x)
x = as.data.frame(x)
subsets = c(1:5, 10, 15, 20, 25)

# rfe with resampling
set.seed(10)
ctrl = rfeControl(functions = lmFuncs,
                  method = 'repeatedcv',
                  repeats = 5,
                  verbose = F)
lmProfile = rfe(x, y,
                sizes = subsets,
                rfeControl = ctrl)
lmProfile
names(lmProfile)
predictors(lmProfile)
lmProfile$fit

trellis.par.set(caretTheme())
plot(lmProfile, type = c("g", "o"))

## helper function when there is no predefined rfe for a given model
## To use feature elimination for an arbitrary model, a set 
## of functions must be passed to rfe for each of the steps 
## in Algorithm 2
# simplified illustrative example
rfRFE = list(summary = defaultSummary,
             fit = function(x, y, first, last, ...){
               library(randomForest)
               randomForest(x, y, importance = first, ...)
             },
             pred = function(object, x) predict(object, x),
             rank = function(object, x, y){
               vimp = varImp(object)
               vimp = vimp[order(vimp$Overall, decreasing = T), drop = F]
               vimp$var = rownames(vimp)
               vimp
               },
             selectSize = pickSizeBest,  
             selectVar = pickVars)

example <- data.frame(RMSE = c(3.215, 2.819, 2.414, 2.144,
                               2.014, 1.997, 2.025, 1.987,
                               1.971, 2.055, 1.935, 1.999,
                               2.047, 2.002, 1.895, 2.018),
                      Variables = 1:16)

## Find the row with the absolute smallest RMSE
smallest <- pickSizeBest(example, metric = "RMSE", maximize = FALSE)
smallest
## Now one that is within 10% of the smallest
within10Pct <- pickSizeTolerance(example, metric = "RMSE", tol = 10, maximize = FALSE)
within10Pct

minRMSE <- min(example$RMSE)
example$Tolerance <- (example$RMSE - minRMSE)/minRMSE * 100
## Plot the profile and the subsets selected using the 
## two different criteria
par(mfrow = c(2, 1), mar = c(3, 4, 1, 2))
plot(example$Variables[-c(smallest, within10Pct)],
     example$RMSE[-c(smallest, within10Pct)],
     ylim = extendrange(example$RMSE),
     ylab = "RMSE", xlab = "Variables")
points(example$Variables[smallest],
       example$RMSE[smallest], pch = 16, cex= 1.3)
points(example$Variables[within10Pct],
       example$RMSE[within10Pct], pch = 17, cex= 1.3)
with(example, plot(Variables, Tolerance))
abline(h = 10, lty = 2, col = "darkgrey")
