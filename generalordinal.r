# importing libraries
library(plyr)
library(foreign)
library(RWeka)
library(tree)
library(dplyr)
library(caret)
library(gbm)
library(randomForest)

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
ordinalDatasets = function(dataset, model)
{
  # extracting the original class attribute
  # (assuming that class attribute is in the end)
  originalClassAttribute = dataset[,length(dataset)]
  
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
        dataset[,length(dataset)] = as.factor(class)
        model(class~., data = dataset)
    })
}

# function to predict an instance (or set of instances) class
ordinalPrediction = function(models, newInstances,classes){
  # firstly we get the probabilities of each instance
  probabilities = lapply(models, function(model){predict(model,newInstances,type="prob")})
  
  # we transform the list into data.frame
  probs = do.call("cbind",probabilities)
  
  # let's remove columns that we aren't going to use
  probs = probs[,-seq(1,dim(probs)[2],2)]
  
  indexOfPredictedClasses = apply(probs, 1, function(p){
    # we store probability of the first class
    # to use it in the next probabilities calculations
    previous = p[1]
    
    # we store the set of probabilities
    probsSet = 1-previous
    
    # now we calculate the rest of the probabilities between the first one and the last one
    if(length(p)>2){
      for(i in p[2:length(p)-1]){
        probsSet=c(probsSet,previous[length(previous)] * (1-i))
        previous = c(previous,i)
      }
      #lapply(p[seq(4,length(p)-1,2)], function(pi){ probsSet=c(probsSet,probsSet[length(probsSet)] * (1-pi))})
    }
    
    # and the last item calculation:
    probsSet = c(probsSet,p[length(p)])
    
    # finally we get the class with major probability
    classOfMaxProbability = which(probsSet == max(probsSet))
  })
  
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
eslModels = ordinalDatasets(eslPartitions[[1]],randomForest)
eraModels = ordinalDatasets(eraPartitions[[1]],randomForest)
levModels = ordinalDatasets(levPartitions[[1]],randomForest)
swdModels = ordinalDatasets(swdPartitions[[1]],randomForest)

# we predict using test dataset without class attribute
eslPredictedResults = ordinalPrediction(eslModels, eslPartitions[[2]][,-length(eslPartitions[[2]])], sort(unique(esl$out1)))
eraPredictedResults = ordinalPrediction(eraModels, eraPartitions[[2]][,-length(eraPartitions[[2]])], sort(unique(era$out1)))
levPredictedResults = ordinalPrediction(levModels, levPartitions[[2]][,-length(levPartitions[[2]])], sort(unique(lev$Out1)))
swdPredictedResults = ordinalPrediction(swdModels, swdPartitions[[2]][,-length(swdPartitions[[2]])], sort(unique(swd$Out1)))

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


