from matplotlib import pyplot
from sklearn import metrics
import pandas
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Conv1D, Conv2D, MaxPooling1D, Dense, Softmax, Dropout, LSTM, Reshape, TimeDistributed
from keras.layers import Bidirectional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from keras import optimizers
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import mean
from numpy import std
from keras import regularizers
import time
import random

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#number of classes
Classes=10
# load dataset
dataframe = pandas.read_csv("TrainNoSLCor.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:3600].astype(float)
Y = dataset[:, 3600]
scaler = MinMaxScaler()
normalized = scaler.fit_transform(X)
seed = [7, 42, 56, 100, 448]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
le_name_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_)))
# convert integers to dummy variables (i.e. one hot encoded)
print (le_name_mapping)
dummy_y = np_utils.to_categorical(encoded_Y)
X_train, X_test, y_train, y_test = train_test_split(
    normalized, dummy_y, random_state=42, test_size=0.2)

scores = list()
TestTimes= list()
TrainTimes= list()
PWeightedAverages= list()
RWeightedAverages= list()
F1WeightedAverages= list()


i=0
for r in range(5):
 tf.random.set_seed(seed[r])
 np.random.seed(seed[r])
 model = Sequential()
 model.add(Conv2D(64, 7, activation='relu', kernel_initializer='uniform',
                 padding="same", input_shape=(100,6,6)))  # if bad results try to change 7 to 5
 model.add(BatchNormalization())
 model.add(Conv2D(64, 3, activation='relu',
                 kernel_initializer='uniform', padding="same"))
 model.add(BatchNormalization())
 model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
 model.add(TimeDistributed(Flatten()))
 model.add(Bidirectional(LSTM(64, kernel_initializer='uniform',
               recurrent_initializer='uniform', bias_regularizer=regularizers.l2(0.03), kernel_regularizer=regularizers.l2(0.03), recurrent_regularizer=regularizers.l2(0.03))))
 model.add(Dropout(0.4, seed=seed[r]))
 model.add(Dense(256, activation='relu'))
 model.add(Dropout(0.2, seed=seed[r]))
 model.add(Dense(Classes, activation='softmax'))
 model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
 tx = np.reshape(X_train, (len(X_train), 100,6,6))
 xt = np.reshape(X_test, (len(X_test), 100,6,6))
 start_time = time.time()
 history=model.fit(tx, y_train, epochs=100, validation_data=(xt, y_test), batch_size=32)
 TrainTime= time.time() - start_time
 TrainTimes.append(TrainTime)
 print("Training Time--- %s seconds ---" % (TrainTime))
 yhat_probs = model.predict(xt, verbose=0)
# predict crisp classes for test set
 dataframe = pandas.read_csv("TestNoSLCor.csv", header=None)
 dataset = dataframe.values
 X_testN = dataset[:,0:3600].astype(float)
 Y_testN = dataset[:, 3600]
# encode class values as integers
 #normalizedT = scaler.fit_transform(X_testN)
 normalizedT= (X_testN - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
 encoded_YTN = encoder.transform(Y_testN)
 dummy_yN = np_utils.to_categorical(encoded_YTN)
 t  = np.reshape(normalizedT, (len(normalizedT), 100, 6, 6))
 start_time = time.time()

 #yhat_probs = model.predict_proba(t, verbose=0)
 yhat_probs = model.predict(t, verbose=0)

 TestTime=time.time() - start_time
 TestTimes.append(TestTime)
 #yhat_classes = model.predict_classes(t, verbose=0)
 yhat = model.predict(t, verbose=0)
 yhat_classes = np.argmax(yhat, axis=1)


 #print(model(xt))
 #model.save("the_model_" + str(r), save_format='tf')
 #loaded_model = tf.keras.models.load_model('the_model_'+str(r))
 #model.save('Models/model_0')


 print(le_name_mapping)
 print('Accuracy: %f' % accuracy)
 print (metrics.confusion_matrix(rounded_label, yhat_classes))
 print(metrics.classification_report(rounded_label, yhat_classes, digits=3))
 PWeightedAverages.append(precision_recall_fscore_support(rounded_label, yhat_classes, average='weighted')[0])
 RWeightedAverages.append(precision_recall_fscore_support(rounded_label, yhat_classes, average='weighted')[1])
 F1WeightedAverages.append(precision_recall_fscore_support(rounded_label, yhat_classes, average='weighted')[2])

 print(model(xt))
 model.save("/the_model_" + str(r), save_format='tf')
 loaded_model = tf.keras.models.load_model('/the_model_'+str(r))
 model.save('Models/model_'+str(r))

m, s = mean(scores), std(scores)
print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
m, s = mean(TrainTimes), std(TrainTimes)
print('TrainTime: %.3f%% (+/-%.3f)' % (m, s))
m, s = mean(TestTimes), std(TestTimes)
print('TestTime: %.3f%% (+/-%.3f)' % (m, s))
m, s = mean(PWeightedAverages), std(PWeightedAverages)
print('Precision: %.3f%% (+/-%.3f)' % (m, s))
m, s = mean(RWeightedAverages), std(RWeightedAverages)
print('Recall: %.3f%% (+/-%.3f)' % (m, s))
m, s = mean(F1WeightedAverages), std(F1WeightedAverages)
print('F1: %.3f%% (+/-%.3f)' % (m, s)) 
