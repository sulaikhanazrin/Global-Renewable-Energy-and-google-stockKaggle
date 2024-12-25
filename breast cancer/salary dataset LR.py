#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import plotly.express as ex


# In[4]:


df = pd.read_csv("C:\\Users\\sulai\\Downloads\\Salary_dataset.csv")


# In[6]:


df


# ## 1. Data Preprocessing & Exploration:

# In[9]:


df.head()


# In[11]:


# Checking for missing values
print("Missing values:", df.isnull().sum())


# In[13]:


# Displaying the basic info about the dataset (data types and non-null counts)
df.info()


# # Checking the summary statistics for numerical columns
# df.describe()

# In[17]:


# Checking for duplicates
print("Duplicate rows:", df.duplicated().sum())


# In[19]:


# Checking the unique values for each column
df.nunique()


# ## 2. Visualizing the Dataset:

# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

# Creating a scatter plot to visualize the relationship between YearsExperience and Salary
plt.figure(figsize=(10,6))
sns.scatterplot(x='YearsExperience', y='Salary', data=df, color='b', s=100)
plt.title('Scatter Plot: Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()


# In[25]:


# Visualizing the linear relationship using seaborn's regplot
plt.figure(figsize=(10, 6))
sns.regplot(x = 'YearsExperience', y = 'Salary', data = df, scatter_kws={'s':100}, line_kws={'color': 'red'})
plt.title('Regression plot: Years of Experience vs Salary')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()


# ## 3. Modeling:

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#defining features and target variable
X = df[['YearsExperience']]
y = df['Salary']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluating the model performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test , y_test_pred)

# Printing the results
print(f"Training MSE: {train_mse}")
print(f"Testing MSE: {test_mse}")
print(f"Training R²: {train_r2}")
print(f"Testing R²: {test_r2}")


# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt

# Creating a DataFrame for actual vs predicted values
train_results = pd.DataFrame({'Actual': y_train, 'predicted': y_train_pred, 'Dataset': 'Training'})
test_results = pd.DataFrame({'Actual': y_test, 'predicted': y_test_pred, 'Dataset':'Testing'})
combined_results = pd.concat([train_results, test_results])



# In[32]:


combined_results


# In[34]:


# Plotting the results
plt.figure(figsize=(10,6))
sns.scatterplot(x='Actual', y ='predicted', hue='Dataset', data=combined_results, s=100, alpha=0.8)
plt.plot([min(y.min(), y_train_pred.min()) - 1, max(y.max(), y_train_pred.max()) + 1],
        [min(y.min(), y_train_pred.min()) - 1,max(y.max(),y_train_pred.max())+ 1], 'r--')

plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary (Training & Testing)')
plt.grid(True)
plt.legend()
plt.show()


# ## 4. Model Evaluation Metrics:

# In[37]:


from sklearn.metrics import mean_squared_error, r2_score

# Calculate Mean Squared Error (MSE) and R² Score
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")


# ## 5. Cross-Validation:

# In[40]:


from sklearn.model_selection import cross_val_score
# Performing 5-fold cross-validation on the linear regression model
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Printing cross-validation scores and the mean score
print("Cross-validation MSE scores:", -cv_scores)
print("Mean cross-validation MSE:", -cv_scores.mean())


# ## 6. Polynomial Regression (for a non-linear trend):

# In[43]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[45]:


# Creating polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Creating and fitting the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Making predictions
y_poly_pred = poly_model.predict(X_poly)

# Plotting the polynomial regression curve
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearsExperience', y='Salary', data=df, color='blue', s=100)
plt.plot(df['YearsExperience'], y_poly_pred, color='red', linewidth=2)
plt.title('Polynomial Regression: YearsExperience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()


# ## 7. Feature Engineering (Interaction Terms, Log Transformation):

# In[48]:


# Applying log transformation to the 'Salary' column
df['LogSalary'] = np.log(df['Salary'])


# In[52]:


df['LogSalary']


# In[54]:


# Visualizing the transformed data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearsExperience', y='LogSalary', data=df, color='green', s=100)
plt.title('Log Transformation of Salary vs YearsExperience')
plt.xlabel('Years of Experience')
plt.ylabel('Log(Salary)')
plt.grid(True)
plt.show()


# In[ ]:




