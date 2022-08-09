import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.keras import layers

import numpy as np
np.set_printoptions(precision=5)
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.float_format = '{:,.5f}'.format
import scipy as sp
from scipy import stats
import seaborn as sns
import sklearn

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

import time

pd.set_option('display.max_columns', None)

######################

CSN_path = "./"

def load_CD_data():
    csv_path = CSN_path + "ML_CD_database_test1_test3_with_Pred_Color_csv.csv"
    return pd.read_csv(csv_path)

CD = load_CD_data()
CD1 = CD.drop(['Example ID', 'DOI', 'Color_P', 'particle size (TEM)', 'Lattice obrserved or not', 'absorption', 'HCl', 'formic acid '], axis=1)
CD_new = pd.get_dummies(CD1)

CD_test = CD_new[-44:] 

CD_test_exp = CD_test[-16:]
CD_test_lit = CD_test[:-16]

AA=CD_new[:-44]

X2 = sklearn.utils.shuffle(AA, random_state=42)



######################

CSN_path = "./"


def load_CD_data():
    #csv_path = CSN_path + "Train407_test15byExp_with_Pred_Color_only_okone_1sttest_csv.csv"
    csv_path = CSN_path + "ML_CD_database_test1_test3_with_Pred_Color_only_csv.csv"
    return pd.read_csv(csv_path)

CD = load_CD_data()
CD2 = CD.drop(['Example ID', 'DOI', 'particle size (TEM)', 'Lattice obrserved or not', 'absorption', 'HCl', 'formic acid '], axis=1)
CD_new2 = pd.get_dummies(CD2)
CD_test2 = CD_new2[-44:] 

CD_test_exp2 = CD_test2[-16:]
CD_test_lit2 = CD_test2[:-16]

AA2=CD_new2[:-44]


X2_2 = sklearn.utils.shuffle(AA2, random_state=42)

#######################

for col in X2.columns:
    print(col)

######################

for col in X2_2.columns:
    print(col)
    
######################

#Optimized hyper-parameters are Adamax, Relu, 4 layers 5 nodes, 2000 epochs, 25 batchsize for the WithTempTime



def norm(x, train_dataset):
    train_stats = train_dataset.describe().transpose()
    return (x - train_stats['mean']) / train_stats['std'].replace(to_replace=0, value=1)

def build_model(data):
    model = keras.Sequential()
    model.add(keras.layers.Dense(5, activation=keras.layers.ReLU(), input_shape=(data.shape[1],)))
    for k in range(4-1):
        model.add(keras.layers.Dense(5, activation=keras.layers.ReLU()))
    model.add(keras.layers.Dense(1, activation=keras.layers.ReLU()))
    
    model.compile(loss='mse',
                optimizer=keras.optimizers.Adamax(),
                metrics=['mae'])
    
    return model

def plot_mae(history):
    plt.figure()
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Abs Error')
    plt.ylim(0, 500)
    plt.plot(history.epoch, np.array(history.history['mae']),
             label='Train MAE')
    plt.plot(history.epoch, np.array(history.history['val_mae']),
             label = 'Val MAE')
    plt.legend()
    plt.show()
    print(np.array(history.history['mae'])[-1], 
        np.array(history.history['val_mae'])[-1])
    
def plot_loss(history):
    plt.figure()
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.ylim(0, 250000)
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train MAE')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label = 'Val MAE')
    plt.legend()
    plt.show()
    print(np.array(history.history['loss'])[-1], 
        np.array(history.history['val_loss'])[-1])
    
def rc(x, y):
    mx = np.mean(x)
    my = np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    
    zx = (x - mx)/sx
    zy = (y - my)/sy
    
    return np.sum(zx*zy)/(x.size-1)

def non_zero_mean(a, axis=None):
    return np.nanmean(np.where(np.isclose(a,0), np.nan, a), axis=axis)

def non_zero_std(a, axis=None):
    return np.nanstd(np.where(np.isclose(a,0), np.nan, a), axis=axis)

#######################

def print_full(x):
    pd.set_option('display.max_rows', x.shape[0])
    if len(x.shape) < 2:
        pd.set_option('display.max_columns', 1)
    else:
        pd.set_option('display.max_columns', x.shape[1])
    return x
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    
#####################

#### Test 3 #########

modelnum = 100

train_v = np.zeros((X2.shape[0], modelnum))
num_train_v = np.zeros((X2.shape[0]))
test_v = np.zeros((CD_test_exp2.shape[0], modelnum))
num_test_v = np.zeros((CD_test_exp2.shape[0]))

start_time = time.time()

for i in range(modelnum):
        
    train = X2.sample(frac=3/4, random_state=i)
    test = CD_test_exp2#CSN_prepared.loc[~CSN_prepared.index.isin(train.index), :]
    
    train_index = train.index
    test_index = test.reset_index(drop=True).index
    
    train_f = train.drop(['Main Peak (in water)'], axis=1)
    test_f = test.drop(['Main Peak (in water)'], axis=1)
    
    ntrain_f = norm(train_f, train_f)
    ntrain_l = train['Main Peak (in water)']
    ntest_f = norm(test_f, train_f)
    ntest_l = test['Main Peak (in water)']
    
    nall_f = norm(CD_test_exp2.drop(['Main Peak (in water)'], axis=1), train_f)
    
    tf.keras.backend.clear_session()
    np.random.seed(i)
    tf.random.set_seed(i)
    model = build_model(ntrain_f)
    
    history = model.fit(ntrain_f,
    ntrain_l,
    #checsamping validation data after each epoch
    #validation_data=(ntest_f, ntest_l),
    epochs=2000,
    batch_size=25,
    verbose = 0
    )
    
    pred = model.predict(nall_f).flatten()
    
#    train_v[train_index, i] += pred[train_index] 
#    num_train_v[train_index] += 1
    test_v[test_index, i] += pred[test_index] 
    num_test_v[test_index] += 1
    
    elapsed_time = time.time() - start_time
    print(i, elapsed_time)
    
####################

