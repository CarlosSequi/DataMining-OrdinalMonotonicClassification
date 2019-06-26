# importing libraries
library(plyr)
library(foreign)
library(RWeka)
library(dplyr)
library(caret)
library(xgboost)

# reading datasets
esl = read.arff("esl.arff")
era = read.arff("era.arff")
lev = read.arff("lev.arff")
swd = read.arff("swd.arff")

# function to create dataset train and test partitions
createPartitions = function(dataset){
  set.seed (20)
  # let's change class attribute name to "class"
  colnames(dataset)[length(dataset)] = "class"
  # we use 66% for train an the rest for test
  trainIndex = sample(1:nrow(dataset), (66*nrow(dataset))/100)
  # returning train and test
  list(as.data.frame(dataset[trainIndex,]),as.data.frame(dataset[-trainIndex,]))
}


# function to create n-1 models to ordinal classification
ordinalDatasets = function(dataset)
{
  # extracting the original class attribute
  # (assuming that class attribute is in the end)
  originalClassAttribute = dataset[,dim(dataset)[2]]
  
  # inspecting which classes contains class attribute of the dataset 
  classes = sort(unique(originalClassAttribute))
  
  # we select the first one as current class
  currentClasses = classes[1]
  
  # let's remove it from classes
  classes = classes[-1]
  
  # we only need to make numberOfClasses-1 models, so last "classes" 
  # item won't be considered.
  # we create a list containing each class combination attribute.
  newClassAttributes = originalClassAttribute
  for(class in classes){
    newClassAttributes = cbind(newClassAttributes,Newclass = ifelse(originalClassAttribute%in%currentClasses, 0, 1))
    currentClasses = c(currentClasses,class)
  }
  
  # we remove the original class attribute
  newClassAttributes = newClassAttributes[,-1]
  
  # let's create the models
  models = apply(newClassAttributes, 2, function(class){
    # we assign the new class attribute
    dataset[,dim(dataset)[2]] = class
    # xgboost only accepts matrix or xgb.DMatrix
    dataset = xgb.DMatrix(as.matrix(dataset[,-dim(dataset)[2]]), label=dataset[,dim(dataset)[2]])
    # let's call de model
    xgboost(data = dataset, monotone_constraints=1,nrounds = 2)
  })
}

# function to predict an instance (or set of instances) class
monotonePrediction = function(models, newInstances,classes){
  
  # firstly we get the probabilities of each instance
  probabilities = lapply(models, function(model){as.numeric(predict(model,as.matrix(newInstances[,-ncol(newInstances)])) > 0.5)})
  #probabilities = lapply(models, function(model){predict(model,as.matrix(newInstances[,-ncol(newInstances)]))})
  print(probabilities)
  
  # we transform the list into data.frame
  probs = do.call("cbind",probabilities)
  
  # let's predict
  indexOfPredictedClasses = apply(probs, 1, function(p){
    sum(p)+1
  })
  print(indexOfPredictedClasses)
  classes[indexOfPredictedClasses]
}


####################################################################################################
####################################################################################################

# firstly we partition datasets
# <datasetName>Partitiors[[1]] will contain train partition
# <datasetName>Partitiors[[2]] will contain test partition
eslPartitions = createPartitions(esl)
eraPartitions = createPartitions(era)
levPartitions = createPartitions(lev)
swdPartitions = createPartitions(swd)


# let's create the models with the train datasets
eslModels = ordinalDatasets(eslPartitions[[1]])
eraModels = ordinalDatasets(eraPartitions[[1]])
levModels = ordinalDatasets(levPartitions[[1]])
swdModels = ordinalDatasets(swdPartitions[[1]])

# we predict using test dataset without class attribute
eslPredictedResults = monotonePrediction(eslModels, eslPartitions[[2]], sort(unique(esl$out1)))
eraPredictedResults = monotonePrediction(eraModels, eraPartitions[[2]], sort(unique(era$out1)))
levPredictedResults = monotonePrediction(levModels, levPartitions[[2]], sort(unique(lev$Out1)))
swdPredictedResults = monotonePrediction(swdModels, swdPartitions[[2]], sort(unique(swd$Out1)))

# now we get the accuracy results
eslAccuracy <- sum(eslPredictedResults == eslPartitions[[2]]$class)/length(eslPredictedResults)
eraAccuracy <- sum(eraPredictedResults == eraPartitions[[2]]$class)/length(eraPredictedResults)
levAccuracy <- sum(levPredictedResults == levPartitions[[2]]$class)/length(levPredictedResults)
swdAccuracy <- sum(swdPredictedResults == swdPartitions[[2]]$class)/length(swdPredictedResults)

# let's see how this model fits the datasets
eslAccuracy
eraAccuracy
levAccuracy
swdAccuracy

# as we can see, accuracy values using ordinal classification method is not really good, so 
# it may be because datasets are note made to apply this classification technique