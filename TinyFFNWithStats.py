
# coding: utf-8

# In[1]:

import os, sys
currentFolder = os.path.abspath('')
projectFolder = 'F:/myProjects/tfKeras/UCSC/CMPS242/earthquake/'
sys.path.append(str(projectFolder))
#exec(open("inc_notebook.py").read())


# In[2]:


import logging, sys, math,os
exec(open("estimator/initKeras.py").read())


# In[3]:


from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'auto')
import seaborn as sns
sns.set(style="darkgrid")


# In[4]:


if sys.modules.get( 'library.MultipleBinDataGenerator', False ) != False :
    del sys.modules['library.MultipleBinDataGenerator'] 
if sys.modules.get( 'MultipleBinDataGenerator', False ) != False :
    del sys.modules['MultipleBinDataGenerator'] 
from library.MultipleBinDataGenerator import *

logging.warning( "MultipleBinDataGenerator loaded" )

trainGenerator = MultipleBinDataGenerator(batch_size=20, windowSize = 10, stride = 10)


# In[5]:


#aBatch = trainGenerator.__getitem__(0)


# In[5]:



if sys.modules.get( 'library.LivePlotKeras', False ) != False :
    del sys.modules['library.LivePlotKeras'] 
if sys.modules.get( 'LivePlotKeras', False ) != False :
    del sys.modules['LivePlotKeras'] 
from library.LivePlotKeras import *

logging.warning( "LivePlotKeras loaded" )

livePlotKeras = LivePlotKeras()


# In[6]:


trainGenerator.__len__()


# In[7]:


model_input = layers.Input( shape = ( 15 + 6 * 27 + 2 + 15+ 3 * 27,  ) )


# In[8]:


x = layers.Dense(64)(model_input)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(32)(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(16)(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(1, activation=activations.relu)(x)

model = models.Model(model_input, x, name = "TinyFFN")
model.summary()


# In[9]:


model.compile(optimizer=optimizers.Adam(lr=0.001),
             loss = losses.MSE,
             metrics = [metrics.MSE, metrics.MAE])


# In[10]:


sys.path.remove(str(projectFolder))
os.chdir(currentFolder)


# In[ ]:


np.seterr(invalid='ignore')
np.warnings.filterwarnings('ignore')
history = model.fit_generator( generator=trainGenerator,
                    use_multiprocessing=True, 
                    workers=4, 
                    initial_epoch = 1,
                    epochs=10,
                    max_q_size = 20,
                    steps_per_epoch = trainGenerator.__len__(),
                    callbacks = [livePlotKeras]
                   )


# In[ ]:


aBatch = trainGenerator.__getitem__(0)

