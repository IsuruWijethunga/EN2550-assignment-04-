# -*- coding: utf-8 -*-
"""Assignmen4.1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nWMKOqgxER9FbDUlBL1CT08WVEtcnYIU
"""

import numpy as np 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10,mnist

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print('x_train',x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)
K=len(np.unique(y_train))
Ntr = x_train.shape[0]
Nte = x_test.shape[0]
Din = 3072
x_train , x_test = x_train/255.0,x_test/255.0
mean_image = np.mean(x_train,axis=0)
x_train = x_train - mean_image
x_test = x_test - mean_image

y_train = tf.keras.utils.to_categorical(y_train, num_classes=K)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=K)

x_train = np.reshape(x_train,(Ntr,Din))
x_test = np.reshape(x_test,(Nte,Din))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
std=1e-5
w1 = std*np.random.randn(Din, K)
b1 = np.zeros(K)
print("w1:", w1.shape)
print("b1:", b1.shape)
batch_size = Ntr

print('x_train',x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)

iterations = 300
lr =1.4e-2
lr_decay=0.999
reg =5e-6
loss_history = []
test_loss_history = []
train_acc_history = []
val_acc_history = []
seed = 0
rng = np.random.default_rng(seed=seed)

#Gradient decending and Calculating train loss throughout the loop

for t in range(iterations):
  indices = np.arange(Ntr)
  rng.shuffle(indices)
  x=x_train[indices]
  y=y_train[indices]
  y_pred = x.dot(w1) +b1
  loss = 1.0/batch_size*np.square(y_pred-y).sum()+reg*(np.sum(w1*w1))
  loss_history.append(loss)
  if t%10==0:
    print('iterations %d / %d : loss %f'%(t,iterations,loss))
  dy_pred = 1./batch_size*2.0*(y_pred-y)
  dw = x.T.dot(dy_pred) + reg * w1
  db = dy_pred.sum(axis=0)
  w1 -=lr *dw
  b1 -=lr*db
  lr *=lr_decay

#Calculating test loss
test_loss = 0

indices = np.arange(Nte)
rng.shuffle(indices)
x=x_test[indices]
y=y_test[indices]
y_pred = x.dot(w1) +b1
test_loss = 1.0/10000*np.square(y_pred-y).sum()+reg*(np.sum(w1*w1))
print("test loss ->",test_loss)
fig,ax =plt.subplots(1,10)
fig.set_size_inches(32,10)
print(np.min(w1))
print(np.max(w1))

#Plotting w images
for i in range(10):
  img = w1[:,i].reshape(32,32,3)
  nor_img =255*(img-np.min(w1))/(np.max(w1)-np.min(w1))
  ax[i].imshow(nor_img.astype('uint8'))
plt.show()

#Calculating Train accuracy
x_t = x_train
print("x_train->",x_t.shape)
y_pred = x_t.dot(w1)+b1
train_acc = 1.0 - 1/(Ntr*9.)*(np.abs(np.argmax(y_train,axis=1)-np.argmax(y_pred,axis=1))).sum()
print("train_acc =",train_acc)

#Calculating Test accuracy
x_t = x_test
print("x_test->",x_t.shape)
y_pred = x_t.dot(w1)+b1
test_acc = 1.0 - 1/(Nte*9.)*(np.abs(np.argmax(y_test,axis=1)-np.argmax(y_pred,axis=1))).sum()
print("test_acc =",test_acc)

print(test_loss_history)
