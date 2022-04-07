
# coding: utf-8

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

import numpy as np
import pandas as pd


# In[ ]:


def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].values)
        docY.append(data.iloc[i+n_prev].values)
    matX = np.array(docX)
    matY = np.array(docY)

    return matX, matY


# In[ ]:


def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])
    return (X_train, y_train), (X_test, y_test)


# In[ ]:

# Retrieve data
(X_train, y_train), (X_test, y_test) = train_test_split(data)

# In[ ]:

print(X_train.shape,y_train.shape)

# In[ ]:

#Define your Sequential model structure
in_out_neurons = 1
hidden_neurons = 300
model = Sequential()
model.add(LSTM(input_dim=in_out_neurons, output_dim=hidden_neurons, return_sequences=False))
model.add(Dense(output_dim=in_out_neurons))
#Activation Function
model.add(Activation("linear"))
#Optimizer
model.compile(loss="mean_squared_error", optimizer="rmsprop")

# In[ ]:

#Train the model
#Batch_size should be appropriate to the memory size
model.fit(X_train, y_train, batch_size=50, nb_epoch=10, validation_split=0.05)

# In[ ]:

# evaluate model fit
score = model.evaluate(X_test, y_test)
print('Test score:', score)


# In[ ]:


#Predicition Function With Example :-

from keras.models import load_model
model=load_model('model.h5') # Pass your Model
df=pd.read_csv('dataset.csv') # Pass your Dataset

df=df.head(500)
df=df[['Voltage', 'Pressure', 'Speed', 'RPM', 'Engine_Temperature']]

len(df)

#Length of df passed to this function will be = n_prev (time-steps)
def predictFunction(time_steps_to_forecast,df):
    for i in range(time_steps_to_forecast):
        print("TIME STEP : ",i)
        docXX=[]
        docXX.append(df.iloc[i:len(df)].values)
        matXX = np.array(docXX)
        predicted=model.predict([matXX])
        docXX=[]
        lst=list(predicted[0])
        #SCALED PREDICTIONS WILL BE ATTACHED TO PASSED DF
        df.loc[len(df)] = lst  
		
predictFunction(2,df)

#Prediction Function for Univariate or Bivariate Target(Y) variable

#Length of df passed to this function will be = n_prev (time-steps)
def predictFunctionForUniBiTarget(time_steps_to_forecast,df,df_generated,model):
    df_copy=df.copy()
    df.drop(['INSERT YOUR Y COLUMNS'],axis=1,inplace=True)

    for i in range(time_steps_to_forecast):
        #print("TIME STEP : ",i)
        docXX=[]
        docXX.append(df.iloc[i:len(df)].values)
        matXX = np.array(docXX)
        predicted=model.predict([matXX])
        x_generated=df_generated.iloc[i].values
        newaRRAY=np.append(x_generated, predicted)
        df_copy.loc[len(df_copy)] = newaRRAY     #attached prediction
        df.loc[len(df_copy)] = x_generated
        
    return df_copy

#predictFunction(time_steps_to_forecast,df)
#predictFunctionForUniBiTarget(time_steps_to_forecast,df,df_generated,model)

len(df)

df.tail(2)		