#!/usr/bin/env python
# coding: utf-8

# In[51]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# In[52]:


numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1)) 
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[53]:


neuronas=np.empty((20,4))
for i in range(20):
    mlp= MLPClassifier(activation='logistic',hidden_layer_sizes=(i+1),max_iter=10000)
    mlp.fit(x_train,y_train)
    neuronas[i,0]= i+1
    neuronas[i,1]= mlp.loss_
    neuronas[i,2]= f1_score(y_test, mlp.predict(x_test), average='macro')
    neuronas[i,3]= f1_score(y_train,mlp.predict(x_train), average='macro')


# In[54]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(neuronas[:,0],neuronas[:,1],label='loss')
plt.xlabel('Número de neuronas')
plt.ylabel('loss')
plt.subplot(1,2,2)
plt.plot(neuronas[:,0],neuronas[:,2],label='f1 test')
plt.plot(neuronas[:,0],neuronas[:,3],label='f1 train')
plt.legend()
plt.xlabel('Número de neuronas')
plt.ylabel('f1')
plt.savefig('loss_f1.png')
plt.show()
print('El número óptimo de neuronas es 5')


# In[55]:


mlp= MLPClassifier(activation='logistic',hidden_layer_sizes=(5),max_iter=10000)
mlp.fit(x_train,y_train)
scale = np.max(mlp.coefs_[0])
plt.figure(figsize=(12,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.title('Neurona {}'.format(i+1))
    plt.imshow(mlp.coefs_[0][:,i].reshape(8,8),cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
plt.savefig('neuronas')


# In[ ]:




