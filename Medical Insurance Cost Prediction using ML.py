#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[3]:


df = pd.read_csv("Downloads/insurance.csv")
df.head()


# # Data Analysis

# In[4]:


df.shape


# In[21]:


df.describe()


# In[22]:


df.info()


# In[5]:


df.isnull().sum()


# In[20]:


sns.set()
sns.distplot(df['age'])
plt.title('Age Distribution')
plt.show()


# In[15]:


sns.countplot(df['sex'])
plt.title('Sex Distribution')
plt.show()


# In[13]:


df['sex'].value_counts()


# In[19]:


sns.set()
sns.distplot(df['bmi'])
plt.title('BMI Distribution')
plt.show()


# In[33]:


sns.set()
sns.countplot(df['children'])
plt.title('Number of Children Distribution')
plt.show()


# In[26]:


df['children'].value_counts()


# In[32]:


sns.set()
sns.countplot(df['smoker'])
plt.title('Smoker Distribution')
plt.show()


# In[31]:


sns.set()
sns.countplot(df['region'])
plt.title('Region Distribution')
plt.show()


# In[28]:


df['region'].value_counts()


# In[30]:


sns.set()
sns.distplot(df['charges'])
plt.title('Charge Distribution')
plt.show()


# # Data Pre-processing

# In[34]:


df.replace({'sex':{'male':0,'female':1}},inplace=True)
df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,"northwest":3}},inplace=True)


# In[35]:


df.head(10)


# # Split the dataset as features and target

# In[44]:


X = df.iloc[:,0:6]


# In[45]:


Y = df.iloc[:,-1:]


# In[46]:


X


# In[47]:


Y


# In[48]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[49]:


print(X.shape)
print(X_train.shape,X_test.shape)


# # Model Training

# In[50]:


regressor = LinearRegression()


# In[51]:


regressor.fit(X_train,Y_train)


# In[52]:


training_data_prediction=regressor.predict(X_train)


# In[55]:


training_data_prediction


# In[54]:


r2_train = metrics.r2_score(Y_train,training_data_prediction)
print("R-squared value: ",r2_train)


# In[56]:


test_data_prediction=regressor.predict(X_test)


# In[58]:


test_data_prediction


# In[59]:


r2_test = metrics.r2_score(Y_test,test_data_prediction)
print("R-squared value: ",r2_test)


# # Building a predictive system

# In[60]:


input_data = (31,1,25.74,0,1,0)


# In[61]:


input_array = np.asarray(input_data)
input_reshaped = input_array.reshape(1,-1)
prediction = regressor.predict(input_reshaped)
print(prediction)


# In[65]:


print("Insurance cost in USD: ",prediction[0][0])


# In[ ]:




