#!/usr/bin/env python
# coding: utf-8

# ## Data Loading
# Let's start by loading the dataset and taking a first look at the data.

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("C:\\Users\\sulai\\Downloads\\german_restaurnts_2024.csv")


# In[24]:


df


# ### Data Cleaning and Preprocessing
# Before diving into analysis, it's crucial to clean and preprocess the data. This includes handling missing values, converting data types, and ensuring consistency.

# In[27]:


# Check for missing values
df.isnull().sum()


# In[29]:


df.info()


# In[31]:


df.dtypes


# In[33]:


df['totalScore'].fillna(df['totalScore'].mean(), inplace=True)
df['popularTimesLiveText'].fillna(df['popularTimesLiveText'].mode(), inplace=True)
df['price'].fillna('Unknown', inplace=True)
df['street'].fillna('Unknown', inplace=True)


# ### Exploratory Data Analysis
# With a clean dataset, we can now explore the data to uncover interesting patterns and insights.

# In[36]:


# Distribution of total scores
sns.histplot(df['totalScore'],bins=20, kde=True)
plt.title('Distribution of Total Scores')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.show()


# In[52]:


df.boxplot()
plt.show()


# In[82]:


# Count of restaurants by city
plt.figure(figsize=(12,6))
sns.countplot(y='city', data=df, order = df['city'].value_counts().index)
plt.title('Number of Restaurants by City')
plt.xlabel('Count')
plt.ylabel('City')
plt.show()


# In[90]:


# correlation 
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(),annot=True, cmap='coolwarm', fmt = '.2f')
plt.title('Correlation Heatmap')
plt.show()


# ## Predictive Modeling
# Let's see if we can predict the total score of a restaurant based on other features. We'll use a simple linear regression model for this task.

# In[95]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#prepare the data
X = df[['reviewsCount','rank']]
y = df['totalScore']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the model
model = LinearRegression()
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse,r2


# ### Discussion and Conclusion
# In this notebook, we explored a dataset of German restaurants, cleaned and preprocessed the data, and performed exploratory data analysis to uncover interesting insights. We also built a simple linear regression model to predict the total score of a restaurant based on its reviews count and rank. The model's performance, as measured by mean squared error and R-squared, provides a baseline for further improvement.
# 
# Future analysis could involve more sophisticated models, feature engineering, and exploring additional data sources to enhance predictive accuracy. If you found this notebook useful or insightful, consider giving it an upvote.
# 
# Credits¶
# This notebook was created with the help of https://devra.ai/ref/kaggle

# In[ ]:




