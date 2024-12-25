#!/usr/bin/env python
# coding: utf-8

# ## Decision Tree Classifier Tutorial with Python

# Hello friends,
# 
# In this kernel, I build a Decision Tree Classifier to predict the safety of the car. I build two models, one with criterion gini index and another one with criterion entropy. I implement Decision Tree Classification with Python and Scikit-Learn.

# In[10]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Exploratory data analysis

# In[15]:


df = pd.read_csv("C:\\Users\\sulai\\Downloads\\car_evaluation.csv")
df


# In[17]:


df.head()


# ## Rename column names¶
# We can see that the dataset does not have proper column names. The columns are merely labelled as 0,1,2.... and so on. We should give proper names to the columns. I will do it as follows:-

# In[20]:


col_names = ['buying','maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

df.columns = col_names

col_names


# In[22]:


# let's again preview the dataset

df.head()


# In[24]:


df.info()


# ## Frequency distribution of values in variables
# Now, I will check the frequency counts of categorical variables.

# In[27]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


for col in col_names:
    
    print(df[col].value_counts()) 


# We can see that the doors and persons are categorical in nature. So, I will treat them as categorical variables.

# **Summary of variables**
# There are 7 variables in the dataset. All the variables are of categorical data type.
# These are given by buying, maint, doors, persons, lug_boot, safety and class.
# class is the target variable.

# In[34]:


df['class'].value_counts()


# The class target variable is ordinal in nature.

# Missing values in variables

# In[38]:


# check missing values in variables

df.isnull().sum()


# We can see that there are no missing values in the dataset. I have checked the frequency distribution of values previously. It also confirms that there are no missing values in the dataset.

# ## Declare feature vector and target variable

# In[42]:


X = df.drop(['class'],axis=1)

y = df['class']


# ## Split data into separate training and test set

# In[45]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[47]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# ## Feature Engineering

# In[50]:


# check data types in X_train

X_train.dtypes


# Encode categorical variables
# 
# 
# Now, I will encode the categorical variables.

# In[54]:


X_train.head()


# We can see that all the variables are ordinal categorical data type.

# In[59]:


pip install category_encoders


# In[61]:


# import category encoders
import category_encoders as ce


# In[63]:


# encode variables with ordinal encoding


# In[79]:


encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[81]:


X_train.head()


# In[83]:


X_test.head()


# We now have training and test set ready for model building.

# ## Decision Tree Classifier with criterion gini index

# In[87]:


# import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier


# In[89]:


# instantiate the DecisionTreeClassifier model with criterion gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3,random_state=0)
#fit the model
clf_gini.fit(X_train,y_train)


# Predict the Test set results with criterion gini index

# In[92]:


y_pred_gini = clf_gini.predict(X_test)


# In[94]:


y_pred_gini


# Check accuracy score with criterion gini index

# In[99]:


from sklearn.metrics import accuracy_score
print('Model accuracy score with criterion gini index: {0:0.4f}'.format(accuracy_score(y_test, y_pred_gini)))


# Here, y_test are the true class labels and y_pred_gini are the predicted class labels in the test-set.
# 
# **Compare the train-set and test-set accuracy**
# 
# 
# Now, I will compare the train-set and test-set accuracy to check for overfitting.

# In[103]:


y_pred_train_gini = clf_gini.predict(X_train)
y_pred_train_gini


# In[105]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))


# Check for overfitting and underfitting

# In[108]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))


# Here, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.

# ## Visualize decision-trees

# In[118]:


plt.figure(figsize=(12,8))

from sklearn import tree
tree.plot_tree(clf_gini.fit(X_train,y_train))
plt.legend
plt.show()


# ## Decision Tree Classifier with criterion entropy 

# In[123]:


# instantiate the DecisionTreeClassifier model with criterion entropy

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=0)

#fit the model
clf_en.fit(X_train,y_train)


# Predict the Test set results with criterion entropy

# In[128]:


y_pred_en = clf_en.predict(X_test)


# ## Check accuracy score with criterion entropy

# In[131]:


from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'.format(accuracy_score(y_test, y_pred_en)))


# Compare the train-set and test-set accuracy
# 
# 
# Now, I will compare the train-set and test-set accuracy to check for overfitting.

# In[134]:


y_pred_train_en = clf_en.predict(X_train)

y_pred_train_en


# In[136]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))


# Check for overfitting and underfitting

# In[139]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))


# We can see that the training-set score and test-set score is same as above. The training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.

# ## Visualize decision-trees

# In[147]:


plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_en.fit(X_train, y_train)) 
plt.show()


# Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.
# 
# But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making.
# 
# We have another tool called Confusion matrix that comes to our rescue.
# 
# ## Confusion matrix
# 
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
# 
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
# 
# True Positives (TP) – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
# 
# True Negatives (TN) – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
# 
# False Positives (FP) – False Positives occur when we predict an observation belongs to a certain class but the observation actually does not belong to that class. This type of error is called Type I error.
# 
# False Negatives (FN) – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called Type II error.
# 
# These four outcomes are summarized in a confusion matrix given below.

# In[152]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_en)

print('Confusion matrix\n\n', cm)


# ## Classification Report
# Table of Contents
# 
# Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. I have described these terms in later.
# 
# We can print a classification report as follows:-

# In[155]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_en))


# ## Results and conclusion
# Table of Contents
# 
# In this project, I build a Decision-Tree Classifier model to predict the safety of the car. I build two models, one with criterion gini index and another one with criterion entropy. The model yields a very good performance as indicated by the model accuracy in both the cases which was found to be 0.8021.
# 
# 
# In the model with criterion gini index, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.
# 
# 
# Similarly, in the model with criterion entropy, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021.We get the same values as in the case with criterion gini. So, there is no sign of overfitting.
# 
# 
# In both the cases, the training-set and test-set accuracy score is the same. It may happen because of small dataset.
# 
# 
# The confusion matrix and classification report yields very good model performance.
