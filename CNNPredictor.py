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

def makeCNNModel(evalMetrics, learningRate, inputSize):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = inputSize))
  model.add(tf.keras.layers.AveragePooling2D((3, 3)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation = 'relu'))
  model.add(tf.keras.layers.Dense(128, activation = 'relu'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer = tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = evalMetrics)
  return model
  
trainingCircuits = ['/data/Pickle/aes/', '/data/Pickle/jpeg/', '/data/Pickle/tinyRocket/',
  '/data/Pickle/bp/', '/data/Pickle/ibex/', '/data/Pickle/dynamic_node/',
  '/data/Pickle/bp_be/', '/data/Pickle/bp_fe/', '/data/Pickle/bp_multi/', '/data/Pickle/gcd/', '/data/Pickle/swerv/']

numEpochs = 100
inputSize = (22, 33, 33)
learningRate = 0.005
batchSize = 32
evalMetrics = [tf.keras.metrics.TruePositives(name='tp'),
               tf.keras.metrics.FalsePositives(name='fp'),
               tf.keras.metrics.TrueNegatives(name='tn'),
               tf.keras.metrics.FalseNegatives(name='fn'),
               tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall'),
               tf.keras.metrics.AUC(name='auc')]
model = makeCNNModel(evalMetrics, learningRate, inputSize)

trainingPickles = list()
for circuit in trainingCircuits:
  for pkl in os.listdir(circuit):
    if '.pkl' not in pkl:
      continue
    trainingPickles.append(circuit+pkl)
trainingPickles.sort()

historyDf = pd.DataFrame()
for epoch in range(numEpochs):
  random.shuffle(trainingPickles)
  train = trainingPickles
  for trainPickle in trainingPickles:
    trainDf = pd.read_pickle(trainPickle, compression='gzip')
    valDf = trainDf.sample(frac=0.2)
    trainDf = trainDf.drop(valDf.index)

    
    labels = trainDf.pop(trainDf.columns.values[-1])
    valLabels = valDf.pop(valDf.columns.values[-1])
    print('labels', sum(labels), 'valLabels', sum(valLabels))
    trainHyperImages = np.array(trainDf).reshape(len(trainDf),22,33,33)
    valHyperImages = np.array(valDf).reshape(len(valDf),22,33,33)
    train_history = model.fit(x=trainHyperImages,
                             y=labels,
                             batch_size=batchSize,
                             validation_data=(valHyperImages, valLabels))
    history = pd.DataFrame(train_history.history)
    historyDf = pd.concat([historyDf, history])
    historyDf.to_csv('model/history.csv', index=False)
  model.save('model/savedModel')
  model.save_weights('model/modelWeights/model.ckpt')
