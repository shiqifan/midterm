#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[27]:


baseline=pd.read_table('E:/originnet_baseline/trainprocessbaseline.txt',sep='|',header=None)
acc=baseline[4]
loss=baseline[2]
lr=baseline[5]

acc=np.array(acc)
loss=np.array(loss)
lr=np.array(lr)

plt.plot(acc, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()

plt.plot(loss, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs')
plt.show()

plt.plot(lr, '-x')
plt.xlabel('epoch')
plt.ylabel('lr')
plt.title('LearningRate vs. No. of epochs')
plt.show()


# In[25]:


mixup=pd.read_table('E:/originnet_mixup/trainprocessmixup.txt',sep='|',header=None)
acc=mixup[4]
loss=mixup[2]
lr=mixup[5]

acc=np.array(acc)
loss=np.array(loss)
lr=np.array(lr)

plt.plot(acc, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()

plt.plot(loss, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs')
plt.show()

plt.plot(lr, '-x')
plt.xlabel('epoch')
plt.ylabel('lr')
plt.title('LearningRate vs. No. of epochs')
plt.show()


# In[28]:


cutout=pd.read_table('E:/originnet_cutout/trainprocesscutout.txt',sep='|',header=None)
acc=cutout[4]
loss=cutout[2]
lr=cutout[5]

acc=np.array(acc)
loss=np.array(loss)
lr=np.array(lr)

plt.plot(acc, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()

plt.plot(loss, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs')
plt.show()

plt.plot(lr, '-x')
plt.xlabel('epoch')
plt.ylabel('lr')
plt.title('LearningRate vs. No. of epochs')
plt.show()


# In[30]:


cutmix=pd.read_table('E:/originnet_cutmix/trainprocesscutmix.txt',sep='|',header=None)
acc=cutmix[4]
loss=cutmix[2]
lr=cutmix[5]

acc=np.array(acc)
loss=np.array(loss)
lr=np.array(lr)

plt.plot(acc, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()

plt.plot(loss, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs')
plt.show()

plt.plot(lr, '-x')
plt.xlabel('epoch')
plt.ylabel('lr')
plt.title('LearningRate vs. No. of epochs')
plt.show()


# In[ ]:




