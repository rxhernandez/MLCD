### With temperature and time test1 and test2#####

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style="darkgrid")
import sklearn

#################

def plot_acc(history):
    plt.figure()
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.plot(history.epoch, np.array(history.history['acc']),
             label='Train Acc')
    plt.plot(history.epoch, np.array(history.history['val_acc']),
             label = 'Val Acc')
    plt.legend()
    plt.show()
    print(np.array(history.history['acc'])[-1], 
        np.array(history.history['val_acc'])[-1])
    
def plot_loss(history):
    plt.figure()
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.ylim(0, 150)
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train MAE')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label = 'Val MAE')
    plt.legend()
    plt.show()
    print(np.array(history.history['loss'])[-1], 
        np.array(history.history['val_loss'])[-1])
    
    
##################

CSN_path = "./"

def load_CD_data():
    csv_path = CSN_path + "ML_CD_database_test1_test2_csv.csv"
    return pd.read_csv(csv_path)

CD = load_CD_data()


CD1 = CD.drop(['Example ID', 'DOI', 'Color', 'particle size (TEM)', 'Lattice obrserved or not', 'absorption', 'HCl', 'formic acid '], axis=1)
CD_new = pd.get_dummies(CD1)
CD_test = CD_new[-44:] 

CD_test_exp = CD_test[-16:]
CD_test_lit = CD_test[:-16]

AA=CD_new[:-44]
X2 = sklearn.utils.shuffle(AA, random_state=42)

##################

CD2 = CD
CD_new2 =CD2 



CD_test2 = CD_new2[-44:] 

CD_test_exp2 = CD_test2[-16:]
CD_test_lit2 = CD_test2[:-16]

AA2=CD_new2[:-44]

X2_2 = sklearn.utils.shuffle(AA2, random_state=42)

###################

CD3 = CD.Color
CD_new3 =CD3 

CD_new3=pd.get_dummies(CD_new3)
CD_test3 = CD_new3[-44:] 

CD_test_exp3 = CD_test3[-16:]
CD_test_lit3 = CD_test3[:-16]

AA3=CD_new3[:-44]

X2_3 = sklearn.utils.shuffle(AA3, random_state=42)

##################

CD_test.head()
CD_test2.head()
CD_test3.head()

##################

y_test=CD_test3.values

y_train =X2_3.values

#################

from sklearn.model_selection import train_test_split
XTr, XTe, YTr, YTe = train_test_split(X2, y_train, test_size=0.25, random_state=0)

#Building model

from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model

input_layer = Input(shape=(X2.shape[1],))
dense_layer_1 = Dense(15, activation='sigmoid')(input_layer)
dense_layer_2 = Dense(10, activation='sigmoid')(dense_layer_1)
output = Dense(y.shape[1], activation='softmax')(dense_layer_2)
#output = Dense(y.shape[1], activation='relu')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


################

print(model.summary())

history = model.fit(X2, y_train, batch_size=64, epochs=2500, verbose=1, validation_split=0.25)

plot_acc(history)
plot_loss(history)

#Evaluate model
model.evaluate(CD_test, y_test)

pred_train= model.predict(XTr)
scores = model.evaluate(XTr, YTr, verbose=0)

print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1])) 

pred_test= model.predict(XTe)
scores2 = model.evaluate(XTe, YTe, verbose=0)

print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

pred_trainAll= model.predict(X2)

scores3 = model.evaluate(X2, y_train, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores3[1], 1 - scores3[1]))    


#Evaluate model
model.evaluate(X2, y_train)

#Predicting color for the whole 419 data 

Y=pd.get_dummies(CD.Color, prefix='color')
Y1=Y.values
X=CD_new
#predict for the 379
pred_all=model.predict(X)
print(pred_all)

pAll=np.argmax(pred_all, axis=1)
print(pAll)


yy5 = Y1
yyy5= np.argmax(yy5, axis=1)
print(yyy5)

#############################


############################

### With temperature and time test1 and test3#####

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style="darkgrid")
import sklearn

####################

def plot_acc(history):
    plt.figure()
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.plot(history.epoch, np.array(history.history['acc']),
             label='Train Acc')
    plt.plot(history.epoch, np.array(history.history['val_acc']),
             label = 'Val Acc')
    plt.legend()
    plt.show()
    print(np.array(history.history['acc'])[-1], 
        np.array(history.history['val_acc'])[-1])
    
def plot_loss(history):
    plt.figure()
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.ylim(0, 150)
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train MAE')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label = 'Val MAE')
    plt.legend()
    plt.show()
    print(np.array(history.history['loss'])[-1], 
        np.array(history.history['val_loss'])[-1])
    
#####################

CSN_path = "./"

def load_CD_data():
    csv_path = CSN_path + "ML_CD_database_test1_test3_csv.csv"
    return pd.read_csv(csv_path)

CD = load_CD_data()


CD1 = CD.drop(['Example ID', 'DOI', 'Color', 'particle size (TEM)', 'Lattice obrserved or not', 'absorption', 'HCl', 'formic acid '], axis=1)
CD_new = pd.get_dummies(CD1)



CD_test = CD_new[-44:] 
CD_test_exp = CD_test[-16:]
CD_test_lit = CD_test[:-16]

AA=CD_new[:-44]


X2 = sklearn.utils.shuffle(AA, random_state=42)

####################

CD2 = CD
CD_new2 =CD2 



CD_test2 = CD_new2[-44:]  

CD_test_exp2 = CD_test2[-16:]
CD_test_lit2 = CD_test2[:-16]

AA2=CD_new2[:-44]


X2_2 = sklearn.utils.shuffle(AA2, random_state=42)

###################

CD3 = CD.Color
CD_new3 =CD3 

CD_new3=pd.get_dummies(CD_new3)



CD_test3 = CD_new3[-44:] 

CD_test_exp3 = CD_test3[-16:]
CD_test_lit3 = CD_test3[:-16]

AA3=CD_new3[:-44]


X2_3 = sklearn.utils.shuffle(AA3, random_state=42)


####################

CD_test.head()
CD_test2.head()
CD_test3.head()

###################


y_test=CD_test3.values
y = X2_3

y_train =X2_3.values

from sklearn.model_selection import train_test_split
XTr, XTe, YTr, YTe = train_test_split(X2, y_train, test_size=0.25, random_state=0)

#Building model

from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model

input_layer = Input(shape=(X2.shape[1],))
dense_layer_1 = Dense(15, activation='sigmoid')(input_layer)
dense_layer_2 = Dense(10, activation='sigmoid')(dense_layer_1)
output = Dense(y.shape[1], activation='softmax')(dense_layer_2)
#output = Dense(y.shape[1], activation='relu')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


####################

print(model.summary())
history = model.fit(X2, y_train, batch_size=64, epochs=2500, verbose=1, validation_split=0.25)

plot_acc(history)
plot_loss(history)

#Evaluate model
model.evaluate(CD_test, y_test)

pred_train= model.predict(XTr)
scores = model.evaluate(XTr, YTr, verbose=0)

print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1])) 

pred_test= model.predict(XTe)
scores2 = model.evaluate(XTe, YTe, verbose=0)

print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

pred_trainAll= model.predict(X2)

scores3 = model.evaluate(X2, y_train, verbose=0)

print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores3[1], 1 - scores3[1]))    

#Evaluate model
model.evaluate(X2, y_train)

#Predicting color for the whole 419 data 

Y=pd.get_dummies(CD.Color, prefix='color')

Y1=Y.values

X=CD_new

#Evaluate model
model.evaluate(X, Y)

#predict for the 379
pred_all=model.predict(X)
print(pred_all)

pAll=np.argmax(pred_all, axis=1)
print(pAll)


yy5 = Y1
yyy5= np.argmax(yy5, axis=1)
print(yyy5)

####################
