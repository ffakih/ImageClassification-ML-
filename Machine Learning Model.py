#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy',metrics= ['accuracy'])
model.fit(x_train,y_train,epochs = 3)


# In[7]:


val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)


# In[15]:


import numpy as np
import matplotlib.pyplot as plt


# In[9]:


model.save('Classifier.model')


# In[10]:


new_model = tf.keras.models.load_model('Classifier.model')


# In[11]:


predictions = new_model.predict([x_test])


# In[12]:





# In[17]:


print (np.argmax(predictions[1]))


# In[18]:


plt.imshow(x_test[1])
plt.show()


# In[ ]:




