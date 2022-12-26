#!/usr/bin/env python
# coding: utf-8

# # CO2 Emission

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(r'C:\Users\hp\OneDrive\MLProject1\co2-emission-sectors.csv')


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df['From_Cement(BT)'].value_counts()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


final_df = df.drop(['Country'], axis=1)


# In[9]:


final_df


# In[10]:


pip install matplotlib


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


import numpy as np


# In[13]:


final_df.plot(kind='scatter',x='Year',y='From_Oil(BT)',color='red')
plt.savefig('outputfile_OIL_Scatter.png')


# In[14]:


final_df.plot(kind='scatter',x='Year',y='From_Coal(BT)',color='blue')
plt.savefig('outputfile_COAL_Scatter.png')


# In[15]:


final_df.plot(kind='scatter',x='Year',y='From_Cement(BT)',color='green')
plt.savefig('outputfile_CEMENT_Scatter.png')


# In[16]:


final_df.plot(kind='scatter',x='Year',y='Total(BT)',color='cyan')
plt.savefig('outputfile_CEMENT_Scatter.png')


# In[17]:


x = df['Year'].to_numpy()
y=df['From_Oil(BT)'].to_numpy()
z=df['From_Coal(BT)'].to_numpy()
w=df['From_Cement(BT)'].to_numpy()
t=df['Total(BT)'].to_numpy()


# In[18]:


plt.plot(x, y, label = "oil")
plt.plot(x, z, label = "coal")
plt.plot(x,w,label="cement")
plt.plot(x,t,label="total")
plt.legend()
plt.savefig('outputfile_line_graph_All')


# In[19]:


ax = final_df.hist(column='Total(BT)', color='#86bf91')


# # Train-Test-Spliting

# In[20]:


#pip install sklearn


# In[21]:


final_df['From_Coal(BT)'].value_counts


# In[22]:


final_df['From_Oil(BT)'].value_counts


# In[23]:


final_df['From_Cement(BT)'].value_counts


# In[24]:


newdf = final_df[final_df["Year"] >= 1980]

newdf.plot(kind='scatter',x='Year',y='Total(BT)',color='black')
plt.savefig('outputfile_newtotal_Scatter.png')


# In[25]:


prevdf = final_df[final_df["Year"]<1980]
prevdf.plot(kind='scatter',x='Year',y='Total(BT)',color='blue')
plt.savefig('outputfile_prevtotal_Scatter.png')


# In[26]:


newdf


# # Looking for Correlations

# In[27]:


corr_matrix = newdf.corr()  #for new refined data set
corr_matrix['Total(BT)'].sort_values(ascending=False)


# In[28]:


from pandas.plotting import scatter_matrix
attributes = ['Year','From_Oil(BT)','From_Coal(BT)','From_Cement(BT)','Total(BT)']
scatter_matrix(newdf[attributes],figsize=(15,10))


# In[29]:


newdf.plot(kind='scatter',x='Year',y='Total(BT)',alpha = 0.8)


# In[30]:


# looking for non-divided data set correlation
org_corr = final_df.corr()
org_corr['Total(BT)'].sort_values(ascending=False)


# In[31]:


attributes = ['Year','From_Oil(BT)','From_Coal(BT)','From_Cement(BT)','Total(BT)']  #for original data set
scatter_matrix(final_df[attributes],figsize=(20,15))


# ## Modeling the data set

# In[32]:


X = newdf[['Year']]
y= newdf[['Total(BT)']]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2, random_state= 0)


# In[33]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)


# In[34]:


c = regr.intercept_
c


# In[35]:


m = regr.coef_
m


# In[36]:


y_pred_train = regr.predict(X_train)
y_pred_train


# In[37]:


plt.scatter(y_train, y_pred_train,color ="green")
plt.xlabel("Actual Emission")
plt.ylabel("Predicted Emission")
plt.show()


# In[38]:


from sklearn.metrics import r2_score
r2_score(y_train,y_pred_train)


# In[39]:


y_pred_test = regr.predict(X_test)
plt.scatter(y_test,y_pred_test)
plt.xlabel("Actual value")
plt.ylabel("predicted value")
plt.show()


# In[40]:


r2_score(y_test,y_pred_test)


# Best Fit Line

# In[41]:


plt.scatter(X, y, color='purple')
plt.plot(X, m*X+c, color='steelblue', linestyle='--', linewidth=2)


# Doing Statics

# In[47]:


import math 
mean = y_pred_train.mean()
print("predicted mean ", mean)
MSE = np.square(np.subtract(y_train,y_pred_train)).mean() 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:" ,RMSE)


# What I learned from this Project.
# 1) Ml concept of linear regression, multiple linear regression, logistic regression and many other.
# 2) coding 
# 3) Learned to do things systematically
# What are the answer of some major question I got.
# 1) Coal is the highest contributor of CO2
# 2) From 1980, emission of CO2 increases at drastic rate. which show large rate of development of the nation.
# 3) India is presently 3rd largest contributor of CO2
# 
# 
# 

# In[ ]:




