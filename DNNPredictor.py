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

def oversample(train_array, train_labels):
  oversample = RandomOverSampler()
  train_array, train_labels = oversample.fit_resample(train_array, train_labels)
  return train_array, train_labels

def undersample(train_array, train_labels):
  undersample = RandomUnderSampler()
  train_array, train_labels = undersample.fit_resample(train_array, train_labels)
  return train_array, train_labels

# DatePaper
#Row size 5 to 10
#window from 5x5 to 13x13
#best
#7 row 13x13
#input_shape = (13,13,49)
# def makeCNNModel(evalMetrics, dropOut, learningRate, inputSize):
#   model = tf.keras.Sequential()
#   model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = inputSize))
#   model.add(tf.keras.layers.AveragePooling2D((3, 3)))
#   model.add(tf.keras.layers.Flatten())
#   model.add(tf.keras.layers.Dense(128, activation = 'relu'))
#   model.add(tf.keras.layers.Dense(128, activation = 'relu'))
#   model.add(tf.keras.layers.Dense(2, activation = 'relu'))
#   model.compile(optimizer = tf.keras.optimizers.Adam(),
#                 loss = tf.keras.losses.CategoricalCrossentropy(),
#                 metrics = evalMetrics)
#   return model

def makeDNNModel(evalMetrics, dropOut, learningRate, inputSize, numNodes, numLayers):
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













batch_size = 1024 # is important to ensure that each batch has a decent chance of containing a few positive samples
numEpochs = 500
# https://www.youtube.com/watch?v=DO-xv9WLvoM
# https://towardsdatascience.com/how-to-optimize-learning-rate-with-tensorflow-its-easier-than-you-think-164f980a7c7b
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
testChunkSize = 1e5 # 1e6 ~= 10 iterations to cover whole train dataset
validationChunkSize = int(testChunkSize * 0.2)
# convertToImg = False
collumnsToDrop = ['0_#Cells', '0_#CellPins', '0_#Macros', '0_#MacroPins', '0_HorizontalOverflow', '0_VerticalOverflow', '0_TileArea', '0_CellDensity', '0_MacroDensity', '0_MacroPinDensity', '0_Layer1BlkgDensity', '0_Layer2BlkgDensity', '0_Layer1PinDensity', '0_Layer2PinDensity', '1_#Cells', '1_#CellPins', '1_#Macros', '1_#MacroPins', '1_HorizontalOverflow', '1_VerticalOverflow', '1_TileArea', '1_CellDensity', '1_MacroDensity', '1_MacroPinDensity', '1_Layer1BlkgDensity', '1_Layer2BlkgDensity', '1_Layer1PinDensity', '1_Layer2PinDensity', '2_#Cells', '2_#CellPins', '2_#Macros', '2_#MacroPins', '2_HorizontalOverflow', '2_VerticalOverflow', '2_TileArea', '2_CellDensity', '2_MacroDensity', '2_MacroPinDensity', '2_Layer1BlkgDensity', '2_Layer2BlkgDensity', '2_Layer1PinDensity', '2_Layer2PinDensity', '3_#Cells', '3_#CellPins', '3_#Macros', '3_#MacroPins', '3_HorizontalOverflow', '3_VerticalOverflow', '3_TileArea', '3_CellDensity', '3_MacroDensity', '3_MacroPinDensity', '3_Layer1BlkgDensity', '3_Layer2BlkgDensity', '3_Layer1PinDensity', '3_Layer2PinDensity', '5_#Cells', '5_#CellPins', '5_#Macros', '5_#MacroPins', '5_HorizontalOverflow', '5_VerticalOverflow', '5_TileArea', '5_CellDensity', '5_MacroDensity', '5_MacroPinDensity', '5_Layer1BlkgDensity', '5_Layer2BlkgDensity', '5_Layer1PinDensity', '5_Layer2PinDensity', '6_#Cells', '6_#CellPins', '6_#Macros', '6_#MacroPins', '6_HorizontalOverflow', '6_VerticalOverflow', '6_TileArea', '6_CellDensity', '6_MacroDensity', '6_MacroPinDensity', '6_Layer1BlkgDensity', '6_Layer2BlkgDensity', '6_Layer1PinDensity', '6_Layer2PinDensity', '7_#Cells', '7_#CellPins', '7_#Macros', '7_#MacroPins', '7_HorizontalOverflow', '7_VerticalOverflow', '7_TileArea', '7_CellDensity', '7_MacroDensity', '7_MacroPinDensity', '7_Layer1BlkgDensity', '7_Layer2BlkgDensity', '7_Layer1PinDensity', '7_Layer2PinDensity', '8_#Cells', '8_#CellPins', '8_#Macros', '8_#MacroPins', '8_HorizontalOverflow', '8_VerticalOverflow', '8_TileArea', '8_CellDensity', '8_MacroDensity', '8_MacroPinDensity', '8_Layer1BlkgDensity', '8_Layer2BlkgDensity', '8_Layer1PinDensity', '8_Layer2PinDensity']

numNodes = 50
if "NNODES" in os.environ:
  numNodes = int(os.environ["NNODES"])
numLayers = 1
if "NLAYERS" in os.environ:
  numLayers = int(os.environ["NLAYERS"])
dataPath = 'data/'
if "DATAPATH" in os.environ:
  dataPath = os.environ["DATAPATH"]
modelName = ""
if "MNAME" in os.environ:
  modelName = os.environ["MNAME"]
useNeighborhood = True
if "NONEIGHBORHOOD" in os.environ:
  useNeighborhood = False

valReader = ValidationReader(dataPath+'validation.csv', validationChunkSize)
# scaler = pickle.load(open(dataPath+'scaler.pkl','rb'))
resultMetrics = ['TrainingRuntime', 'val_auc', 'auc', 'val_loss', 'loss',
                 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'fscore', 'mcc',
                 'val_tp', 'val_fp', 'val_tn', 'val_fn', 'val_accuracy', 'val_precision', 'val_recall',
                 'val_fscore', 'val_mcc']

modelPath = 'savedModels/DNN_'+str(numLayers)+'L_'+str(numNodes)+'N'+modelName
if useNeighborhood == False:
  modelPath = modelPath + '_NN'

if os.path.exists(modelPath):
  shutil.rmtree(modelPath)
os.mkdir(modelPath)
os.mkdir(modelPath+'/model')
os.mkdir(modelPath+'/modelWeights')

inputSize = len(pd.read_csv(dataPath+'test.csv', nrows=1).columns)-2 # -2 to remove NodeID and label
if useNeighborhood == False:
  inputSize /= 9
  inputSize = int(inputSize)
model = makeDNNModel(evalMetrics, dropOut, learningRate, inputSize, numNodes, numLayers)

scaler = sklearn.preprocessing.StandardScaler()
toDrop = ['NodeID', 'HasDetailedRoutingViolation']
if useNeighborhood == False:
  toDrop += collumnsToDrop

totalPos = 0
totalNeg = 0
for train_df in pd.read_csv(dataPath+'test.csv', chunksize=testChunkSize):
  pos = sum(train_df['HasDetailedRoutingViolation'])
  totalPos += pos
  totalNeg += len(train_df['HasDetailedRoutingViolation']) - pos
  train_df = train_df.drop(columns=toDrop)
  scaler.partial_fit(train_df)
for val_df in pd.read_csv(dataPath+'validation.csv', chunksize=testChunkSize):
  pos = sum(val_df['HasDetailedRoutingViolation'])
  totalPos += pos
  totalNeg += len(val_df['HasDetailedRoutingViolation']) - pos
  val_df = val_df.drop(columns=toDrop)
  scaler.partial_fit(val_df)

weight = None
if "USEWEIGHT" in os.environ:
  total = totalNeg + totalPos
  weight_for_0 = (1 / totalNeg)*(total)/2.0
  weight_for_1 = (1 / totalPos)*(total)/2.0
  class_weight = {0: weight_for_0, 1: weight_for_1}
  print('Pos, Neg, and Total ',totalPos, totalNeg, total)
  print('Weight for class 0: {:.2f}'.format(weight_for_0))
  print('Weight for class 1: {:.2f}'.format(weight_for_1))
else:
  weight = {0: 0.5, 1: 0.5}

finalResults = {x:[] for x in resultMetrics}
for epoch in range(numEpochs):
  print('Current epoch is: ', epoch)
  model.optimizer.lr = 1e-3 + epoch*(1.8e-05) # from 0.001 to 0.01
  epochResults = {}
  for train_df in pd.read_csv(dataPath+'test.csv', chunksize=testChunkSize):
    train_df = train_df.drop(columns=['NodeID'])
    train_df = train_df.sample(frac=1).reset_index(drop=True)#shuffle

    val_df = valReader.getChunk()
    val_df = val_df.drop(columns=['NodeID'])
    val_df = val_df.sample(frac=1).reset_index(drop=True)#shuffle

    train_labels = np.array(train_df.pop('HasDetailedRoutingViolation'))
    val_labels = np.array(val_df.pop('HasDetailedRoutingViolation'))

    if useNeighborhood == False:
      train_df = train_df.drop(columns=collumnsToDrop)
    train_df = scaler.transform(train_df)
    
    if useNeighborhood == False:
      val_df = val_df.drop(columns=collumnsToDrop)
    val_df = scaler.transform(val_df)

    train_array = np.array(train_df)
    val_array = np.array(val_df)

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
print('Saving models and training history')
pd.DataFrame(finalResults).to_csv(modelPath+'/trainingResults.csv', index=False)
model.save(modelPath+'/model/savedModel')
model.save_weights(modelPath+'/modelWeights/model.ckpt')
