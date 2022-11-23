import math
import matplotlib
import numpy as np
import pandas as pd
import pickle
import sklearn
import tensorflow as tf
import random
import shutil
import os
from enum import Enum
import imblearn
import time
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler














def print_positive_ratio(train_labels):
  neg, pos = np.bincount(train_labels)
  total = neg + pos
  print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

# Claculate weight for classes
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
def calculate_class_weights(train_labels):
  if True in train_labels and False in train_labels:
    neg, pos = np.bincount(train_labels)
    total = neg + pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight
  else:
    print('Using default weights(1, 1) due to the absence of either a positive or negative sample.')
    return {0: 1, 1: 1}

def oversample(train_array, train_labels):
  oversample = RandomOverSampler()
  train_array, train_labels = oversample.fit_resample(train_array, train_labels)
  return train_array, train_labels

def undersample(train_array, train_labels):
  undersample = RandomUnderSampler()
  train_array, train_labels = undersample.fit_resample(train_array, train_labels)
  return train_array, train_labels

########## Learning Model ##########
def make_model(evalMetrics, dropOut, learningRate, inputSize, numNodes, numLayers):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Input(shape=inputSize))
  for x in range(numLayers):
    model.add(tf.keras.layers.Dense(numNodes, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropOut))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=evalMetrics)
  return model

def calculateMetrics(tp, fp, tn, fn):
  if tp == 0: #meaningless performance (always predict no viol or there is no viol in labels)
    return 0, 0, 0, 0, -1 #return precision, recall, accuracy, fscore, mcc
  precision = tp/(tp + fp)
  recall = tp/(tp + fn)
  accuracy = (tp + tn)/(tp + fn + tn + fp)
  fscore = (2 * precision * recall)/(precision + recall)
  sqrt = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
  mcc = -1
  if sqrt != 0:
    mcc = ((tp * tn) - (fp * fn))/sqrt
  return precision, recall, accuracy, fscore, mcc

def numLines(file):
  lines = 0
  with open(file) as fp:
    for _ in fp:
      lines += 1
  return lines

class BalanceStrategy(Enum):
  NONE = 0
  WEIGHTS = 1
  OVERSAMPLE = 2
  UNDERSAMPLE = 3

class ValidationReader():
  def __init__(self, filePath, validationChunkSize):
    self.filePath = filePath
    self.validationChunkSize = validationChunkSize
    self.currentItr = 0

    valSizeRows = numLines(self.filePath) - 1 #skip header counting
    self.valChunkIterations = int(valSizeRows/validationChunkSize) #disregard last lines
    self.reader = pd.read_csv(self.filePath, dtype=np.float32, iterator=True)

  def getChunk(self):
    if self.currentItr == self.valChunkIterations:
      self.reader = pd.read_csv(self.filePath, dtype=np.float32, iterator=True)
      self.currentItr = 0
    self.currentItr += 1
    return self.reader.get_chunk(self.validationChunkSize)















strategy = BalanceStrategy.WEIGHTS
batch_size = 32 # is important to ensure that each batch has a decent chance of containing a few positive samples
numEpochs = 10
learningRate = 0.001 #Eh?Predictor=0.05, default=0.001
dropOut = 0.05 #Eh?Predictor=0.05
evalMetrics = [tf.keras.metrics.TruePositives(name='tp'),
               tf.keras.metrics.FalsePositives(name='fp'),
               tf.keras.metrics.TrueNegatives(name='tn'),
               tf.keras.metrics.FalseNegatives(name='fn'),
               tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall'),
               tf.keras.metrics.AUC(name='auc')]
scaler = pickle.load(open('data/scaler.pkl','rb'))
testCSVFile = 'data/test.csv'
validationCSVFile = 'data/validation.csv'
testChunkSize = 1e6 # 1e6 ~= 10 iterations to cover whole train dataset
validationChunkSize = int(testChunkSize * 0.2)
valReader = ValidationReader(validationCSVFile, validationChunkSize)

resultMetrics = ['TrainingRuntime', 'val_auc', 'auc', 'val_loss', 'loss',
                 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'fscore', 'mcc',
                 'val_tp', 'val_fp', 'val_tn', 'val_fn', 'val_accuracy', 'val_precision', 'val_recall',
                 'val_fscore', 'val_mcc']

for numNodes in [50, 100]:
  for numLayers in range(1,3):
    modelPath = 'savedModels/DNN_'+str(numLayers)+'L_'+str(numNodes)+'N'
    if os.path.exists(modelPath):
      shutil.rmtree(modelPath)
    os.mkdir(modelPath)
    os.mkdir(modelPath+'/model')
    os.mkdir(modelPath+'/modelWeights')

    inputSize = len(pd.read_csv(testCSVFile, nrows=1).columns)-2 # -2 to remove NodeID and label
    model = make_model(evalMetrics, dropOut, learningRate, inputSize, numNodes, numLayers)

    finalResults = {x:[] for x in resultMetrics}
    for epoch in range(numEpochs):
      epochResults = {}
      for train_df in pd.read_csv(testCSVFile, chunksize=testChunkSize):
        train_df = train_df.drop(columns=['NodeID'])
        train_df = train_df.sample(frac=1).reset_index(drop=True)#shuffle

        val_df = valReader.getChunk()
        val_df = val_df.drop(columns=['NodeID'])
        val_df = val_df.sample(frac=1).reset_index(drop=True)#shuffle

        train_labels = np.array(train_df.pop('HasDetailedRoutingViolation'))
        val_labels = np.array(val_df.pop('HasDetailedRoutingViolation'))

        train_df = scaler.transform(train_df)
        val_df = scaler.transform(val_df)

        train_array = np.array(train_df)
        val_array = np.array(val_df)

        weight = None
        if strategy == BalanceStrategy.OVERSAMPLE:
          train_array, train_labels = oversample(train_array, train_labels)
        elif strategy == BalanceStrategy.UNDERSAMPLE:
          train_array, train_labels = undersample(train_array, train_labels)
        elif strategy == BalanceStrategy.WEIGHTS:
          weight = calculate_class_weights(train_labels)

        timeStart = time.time()
        train_history = model.fit(x=train_array,
                                 y=train_labels,
                                 batch_size=batch_size,
                                 validation_data=(val_array, val_labels),
                                 class_weight=weight)
        timeEnd = time.time()

        if len(epochResults) == 0:
          epochResults = {x:[] for x in resultMetrics}
        epochResults['TrainingRuntime'].append(timeEnd - timeStart)
        for key, value in train_history.history.items():
          epochResults[key].append(value[0])
      # end chunk read iteration

      finalResults['TrainingRuntime'].append(sum(epochResults['TrainingRuntime']))
      for x in {'auc', 'loss', 'val_auc', 'val_loss'}:# AVG resultMetrics
        finalResults[x].append(sum(epochResults[x]) / len(epochResults[x]))

      tp = sum(epochResults['tp'])
      fp = sum(epochResults['fp'])
      tn = sum(epochResults['tn'])
      fn = sum(epochResults['fn'])
      precision, recall, accuracy, fscore, mcc = calculateMetrics(tp, fp, tn, fn)
      finalResults['tp'].append(tp)
      finalResults['fp'].append(fp)
      finalResults['tn'].append(tn)
      finalResults['fn'].append(fn)
      finalResults['precision'].append(precision)
      finalResults['recall'].append(recall)
      finalResults['accuracy'].append(accuracy)
      finalResults['fscore'].append(fscore)
      finalResults['mcc'].append(mcc)

      vtp = sum(epochResults['val_tp'])
      vfp = sum(epochResults['val_fp'])
      vtn = sum(epochResults['val_tn'])
      vfn = sum(epochResults['val_fn'])
      vprecision, vrecall, vaccuracy, vfscore, vmcc = calculateMetrics(vtp, vfp, vtn, vfn)
      finalResults['val_tp'].append(vtp)
      finalResults['val_fp'].append(vfp)
      finalResults['val_tn'].append(vtn)
      finalResults['val_fn'].append(vfn)
      finalResults['val_precision'].append(vprecision)
      finalResults['val_recall'].append(vrecall)
      finalResults['val_accuracy'].append(vaccuracy)
      finalResults['val_fscore'].append(vfscore)
      finalResults['val_mcc'].append(vmcc)
    # end Epoch
    pd.DataFrame(finalResults).to_csv(modelPath+'/trainingResults.csv', index=False)
    model.save(modelPath+'/model/savedModel')
    model.save_weights(modelPath+'/modelWeights/model.ckpt')
  # end all Epochs
# end all Learning Models
