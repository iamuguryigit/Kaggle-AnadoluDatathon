#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train_path = "/Users/uguryigit/Downloads/archive/train-utf8.csv"
test_path = "/Users/uguryigit/Downloads/archive/test-utf8 (1).csv"
submission_path = "/Users/uguryigit/Downloads/archive/samplesubmission.csv"


# In[ ]:


train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)
submission = pd.read_csv(submission_path, low_memory=False)


# In[ ]:




