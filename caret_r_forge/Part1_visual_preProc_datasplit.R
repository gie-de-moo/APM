### learning on caret home page
# http://caret.r-forge.r-project.org/index.html
library(caret)
library(AppliedPredictiveModeling)
library(doMC)
registerDoMC(4)

install_load <- function(x) {
  install.packages(x)
  library(x, character.only = T)
}
install_load('QSARdata')
install_load('SMCRM')

## visualizations
# scatterplot matrix
str(iris)
transparentTheme(trans = .4)
featurePlot(x = iris[,1:4],
            # y controls color
            y = iris$Species,
            plot = 'pairs',
            # add legends at the top
            auto.key = list(columns = 3))

# scatterplot matrix with Ellipses
featurePlot(x = iris[,1:4],
            y = iris$Species,
            plot = 'ellipse',
            auto.key = list(columns = 3))

# overlayed density plots
transparentTheme(trans = .9)
featurePlot(x = iris[,1:4],
            y = iris$Species,
            plot = 'density',
            # pass in options to xyplot() to
            # make it prettier
            scales = list(x = list(relation='free'),
                          y = list(relation='free')),
            adjust = 1.5,
            pch = '|',
            layout = c(4,1),
            auto.key = list(columns = 3))

# box plots
featurePlot(x = iris[, 1:4],
            y = iris$Species,
            plot = 'box',
            ## pass in options to bwplot()
            scales = list(y = list(relation='free'),
                          x = list(rot = 90)),
            layout = c(4,1),
            auto.key = list(columns = 2))

# scatter plots
library(mlbench)
data(BostonHousing)
str(BostonHousing)
regVar = c('age','lstat','tax')
str(BostonHousing[,regVar])

theme1 = trellis.par.get()
theme1$plot.symbol$col = rgb(.2,.2,.2,.4)
theme1$plot.symbol$lwd = 2
theme1$plot.line$col = rgb(1,0,0,.7)
theme1$plot.line$lwd = 2
trellis.par.set(theme1)
featurePlot(x = BostonHousing[, regVar],
            y = BostonHousing$medv,
            plot = 'scatter',
            # add smoother
            type = c('p', 'smooth'),
            span = 0.5,
            layout = c(3,1))

## pre-processing
# creating dummy variables
library(earth)
data(etitanic)
# sex and pclass are factors
str(etitanic)
# using model.matrix, producing contracts
head(model.matrix(survived ~., data = etitanic))
# using dummyVars, producing a dummyvar for each level
# thus no intercept. May affect parameterization in some
# model functions, such as lm().
dummies = dummyVars(survived ~ ., data = etitanic)
head(dummies)
head(predict(dummies, newdata = etitanic))

## zero and near zero-variance predictors
# if a predictor is very unbalanced, cv and bootstrap
# may have a single unique value, many models can crash
data(mdrr)
data.frame(table(mdrrDescr$nR11))
# to identity such predictor, calculate 2 metrics
# frequency ratio: freq of most prevalent / 2nd prevalent
# ~= 1 is good; large bad
# perc of unique values: n_distinct() / n()
# ~ 0 and small is bad
# should use both metrics to avoid falsely detection
# as in a discrete uniform distribution
nzv = nearZeroVar(mdrrDescr, saveMetrics = T)
head(nzv)
nzv[nzv$nzv,][1:10,]
dim(mdrrDescr)
# filter out problematic variables
nzv = nearZeroVar(mdrrDescr) # return col num of nzv variables
filteredDescr = mdrrDescr[,-nzv]
dim(filteredDescr)

## identifying correlated predictors
descrCor = cor(filteredDescr)
# cols almost perfectly correlated
highCorr = sum(abs(descrCor[upper.tri(descrCor)])>0.999)
percHighCor = highCorr/(ncol(filteredDescr) ^ 2 - ncol(filteredDescr))
percHighCor
# remove descriptors with abs corr > 0.75
summary(descrCor[upper.tri(descrCor)])
# findCorrelation() return col with >= cutoff corr
highlyCorDescr = findCorrelation(descrCor, cutoff = 0.75)
filteredDescr = filteredDescr[, -highlyCorDescr]
descrCor2 = cor(filteredDescr)
summary(descrCor2[upper.tri(descrCor2)])

## linear dependencies
help.search(pattern =  'find', package = 'caret')
ltfrDesign = matrix(0, nrow=6, ncol=6)
ltfrDesign[, 1] <- c(1, 1, 1, 1, 1, 1)
ltfrDesign[, 2] <- c(1, 1, 1, 0, 0, 0)
ltfrDesign[, 3] <- c(0, 0, 0, 1, 1, 1)
ltfrDesign[, 4] <- c(1, 0, 0, 1, 0, 0)
ltfrDesign[, 5] <- c(0, 1, 0, 0, 1, 0)
ltfrDesign[, 6] <- c(0, 0, 1, 0, 0, 1)
ltfrDesign
rowSums(ltfrDesign)
colSums(ltfrDesign)
comboInfo = findLinearCombos(ltfrDesign)
comboInfo
ltfrDesign[,-comboInfo$remove]

## center and scaling
set.seed(96)
inTrain = sample(seq(along=mdrrClass), length(mdrrClass)/2)
training = filteredDescr[inTrain,]
test = filteredDescr[-inTrain,]
trainMDRR = mdrrClass[inTrain]
testMDRR = mdrrClass[-inTrain]
# preProcess only creates a 'preProcess' class object
# predict method is used to actually pre-process datasets
preProcValues = preProcess(training, method = c('center','scale'))
trainTransformed = predict(preProcValues, training)
testTransformed = predict(preProcValues, test)
# preProcess can be used for imputation as well

## transforming predictors
transparentTheme(trans = .4)
plotSubset = data.frame(scale(mdrrDescr[,c('nC','X4v')]))
xyplot(nC ~ X4v,
       data = plotSubset,
       groups = mdrrClass,
       auto.key = list(columns = 2))

# spatial sign transformation
transformed = spatialSign(plotSubset)
transformed = as.data.frame(transformed)
head(transformed)
head(plotSubset)
xyplot(nC ~ X4v,
       data = transformed,
       groups = mdrrClass,
       auto.key = list(columns = 2))

# box-cox transformation
preProcValues2 = preProcess(training, method = 'BoxCox')
trainBC = predict(preProcValues2, training)
testBC = predict(preProcValues2, test)
preProcValues2

## class distance calculations
centrorids = classDist(trainBC, trainMDRR)
distances = predict(centrorids, testBC)
distances = as.data.frame(distances)
head(distances)

xyplot(dist.Active ~ dist.Inactive,
       data = distances,
       groups = testMDRR,
       auto.key = list(columns = 2))

### Data Splitting
## simple splitting based on outcome
# createDataPartition(), createFolds(), createResample()
set.seed(3456)
trainIndex = createDataPartition(iris$Species, p=.8,
                                 list = FALSE,
                                 times = 1)
head(trainIndex)
irisTrain = iris[trainIndex,]
irisTest = iris[-trainIndex,]

## splitting based on predictors
# create sub-samples using a maximum dissimilarity approach
# small dataset A with m samples,
# larger dataset B with n samples
# create a subset of B that is diverse compared to A
# for every b, calculated dissimilarities with every a
# every round, add the most diss b to A

library(mlbench)
data(BostonHousing)
testing = scale(BostonHousing[,c('age','nox')])
set.seed(11)
# a rample sample of 5 data points
startSet = sample(1:dim(testing)[1],5)
samplePool = testing[-startSet,]
start = testing[startSet,]
newSamp = sapply(c(minDiss, sumDiss), function(obj) maxDissim(start, samplePool,
                                                    n = 20, obj = obj))
## splitting for time series


