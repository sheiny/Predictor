import math
import matplotlib
import numpy as np
import pandas as pd
import pickle
import seaborn
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
import random
import shutil
from keras.callbacks import CSVLogger
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
    neg, pos = np.bincount(train_labels)
    total = neg + pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight, neg, pos

def oversample(train_array, train_labels):
    oversample = RandomOverSampler()
    train_array, train_labels = oversample.fit_resample(train_array, train_labels)
    return train_array, train_labels

def undersample(train_array, train_labels):
    undersample = RandomUnderSampler()
    train_array, train_labels = undersample.fit_resample(train_array, train_labels)
    return train_array, train_labels

########## Learning Model ##########
def make_model(evalMetrics, dropOut, learningRate, inputSize, numNodes, numLayers, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=inputSize))
    for x in range(numLayers):
        model.add(tf.keras.layers.Dense(numNodes, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropOut))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=evalMetrics)
    return model

########## Test and check Performance ##########
def calculate_test_metrics(model, results):
    m = {}
    for name, value in zip(model.metrics_names, results):
        m[name] = value
    if m['precision'] + m['recall'] != 0:
        f_score = 2 * ((m['precision'] * m['recall'])/(m['precision'] + m['recall']))
        m['F-score'] = f_score
    sqrt = math.sqrt((m['tp']+m['fp'])*(m['tp']+m['fn'])*(m['tn']+m['fp'])*(m['tn']+m['fn']))
    if sqrt != 0:
        mcc = ((m['tp'] * m['tn']) - (m['fp'] * m['fn']))/sqrt
        m['MCC'] = mcc
    return m

class timecallback(tf.keras.callbacks.Callback):
  def __init__(self, filePath):
    self.start = None
    self.filePath = filePath
    pd.DataFrame(columns=['Runtime']).to_csv(self.filePath, mode='a', index=False, header=True)
  def on_epoch_begin(self,epoch,logs = {}):
    self.start = time.time()
  def on_epoch_end(self,epoch,logs = {}):
    duration = time.time() - self.start
    pd.DataFrame([duration], columns = ['runtime']).to_csv(self.filePath, mode='a', index=False, header=False)

class BalanceStrategy(Enum):
    NONE = 0
    WEIGHTS = 1
    OVERSAMPLE = 2
    UNDERSAMPLE = 3



df = pd.read_csv('data/train.csv')
# Remove NodeIDs (debug info)
df = df.drop(columns=['NodeID'])



########## Hyper parameters ##########
strategy = BalanceStrategy.WEIGHTS
batch_size = 32 # is important to ensure that each batch has a decent chance of containing a few positive samples
epochs = 10
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



# Split 80/20 (train 80% test 20%)
train_df, val_df = sklearn.model_selection.train_test_split(df, test_size=0.2)

# Build np arrays of labels and features.
train_labels = np.array(train_df.pop('HasDetailedRoutingViolation'))
val_labels = np.array(val_df.pop('HasDetailedRoutingViolation'))

scaler = pickle.load(open('data/scaler.pkl','rb'))
train_df = scaler.transform(train_df)
val_df = scaler.transform(val_df)

train_array = np.array(train_df)
val_array = np.array(val_df)

# Save some memory
del train_df
del val_df

print_positive_ratio(train_labels)
# Apply the selected strategy to handle umbalanced data.
weight = None
if strategy == BalanceStrategy.OVERSAMPLE:
  train_array, train_labels = oversample(train_array, train_labels)
elif strategy == BalanceStrategy.UNDERSAMPLE:
  train_array, train_labels = undersample(train_array, train_labels)
elif strategy == BalanceStrategy.WEIGHTS:
  weight = calculate_class_weights(train_labels)
  weight = weight[0]







for numNodes in [50, 100]:
  for numLayers in range(1,3):

    model_name = 'savedModels/DNN_'+str(numLayers)+'L_'+str(numNodes)+'N'
    if os.path.exists(model_name):
      shutil.rmtree(model_name)
    os.mkdir(model_name)
    checkpoint_path = model_name+'/model.ckpt'

    neg, pos = np.bincount(train_labels)
    initial_bias = np.log([pos/neg])

    inputSize = len(train_array[0])
    model = make_model(evalMetrics, dropOut, learningRate, inputSize, numNodes, numLayers, initial_bias)

    dataset = tf.data.Dataset.from_tensor_slices((train_array, train_labels))
    train_dataset = dataset.shuffle(len(train_array)).batch(batch_size)



    # Create a callback that saves the model's weights at the end of each epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
    # Create a callback that saves model history at the end of each epoch
    csv_logger = CSVLogger(model_name+'/model_history_log.csv', append=True)
    # Create a callback that saves runtimes of each epoch
    runtime_callback = timecallback(model_name+'/runtime.csv')

    train_history = model.fit(train_dataset,
                             batch_size=batch_size,
                             validation_data=(val_array, val_labels),
                             class_weight=weight,
                             epochs=epochs,
                             callbacks=[cp_callback, csv_logger, runtime_callback])
