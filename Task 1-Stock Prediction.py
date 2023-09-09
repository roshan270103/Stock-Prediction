#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


data = pd.read_csv('stock_data.csv')


# In[3]:


data.head()


# In[4]:


data.describe()


# In[7]:


scaler = MinMaxScaler()


# In[8]:


data_scaled = scaler.fit_transform(data[['Open', 'Close', 'Volume']])


# In[9]:


sequence_length = 10
X = []
y = []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length][1])  # Predicting the 'Close' price

X = np.array(X)
y = np.array(y)


# In[10]:


split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# In[11]:


model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(50, return_sequences=False),
    layers.Dense(1)
])


# In[12]:


model.compile(optimizer='adam', loss='mse')


# In[13]:


model.fit(X_train, y_train, epochs=50, batch_size=64)


# In[15]:


predictions = model.predict(X_test)


# In[16]:


predictions_actual = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], predictions.reshape(-1, 1)), axis=1))[:, -1]


# In[17]:


import matplotlib.pyplot as plt

# Plot actual and predicted prices
plt.figure(figsize=(12, 6))
plt.plot(data['Date'][split+sequence_length:], data['Close'][split+sequence_length:], label='Actual Close Price')
plt.plot(data['Date'][split+sequence_length:], predictions_actual, label='Predicted Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()


# In[ ]:




