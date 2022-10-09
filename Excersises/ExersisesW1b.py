# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:15:37 2022

@author: Jacob Berg Eriksen
"""

""" Reading txt file"""
#Start by reading the data create a data matrix x:
import numpy as np

in_dir = "02502Pythondata1b/data/"
txt_name = "irisdata.txt"
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
# x is a matrix with 50 rows and 4 columns
x = iris_data[0:50, 0:4]
#then check the data dimensions by writing:
n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat} and number of observations: {n_obs}")
#We have 50 flowers and for each flower there are 4 measurements (features):
#sepal length, sepal width, petal length and petal width. Are your matrix dimensions correct?
#yes


""" explorative data analysis """
#To explore the data, we can create vectors of the individual feature:
sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]
#Compute the variance of each feature like:
# Use ddof = 1 (delta degree of freedom) to make an unbiased estimate
# The variance is the average of the squared deviations from the mean, 
# i.e., var = mean(x), where x = abs(a - a.mean())**2.
var_sep_l = sep_l.var(ddof=1)
var_sep_w = sep_w.var(ddof=1)
var_pet_l = pet_l.var(ddof=1)
var_pet_w = pet_w.var(ddof=1)


#Exercise 3
#compute covariance between sepal lenth and petal length 
#compare it to covariance between sepal length and width
def covar(A,B):
    cv=(1/(len(A)-1))*sum(A*B)
    return cv

cv_length_lenght = covar(sep_l,pet_l)
cv_length_width = covar(sep_l,sep_w)
print(f"Covariance  sepal length and the petal length: {cv_length_lenght}") # = 7,484
print(f"Covariance  sepal length and the sepal width: {cv_length_width}") # = 17,61

#both covar are positive so there is positive relation between both sets
# covar(sepal_len, sepal_width) was much higher meaning there is a stronger relationship here!

# show plot of data!
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.figure() # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
d = pd.DataFrame(x, columns=['Sepal length', 'Sepal width',
'Petal length', 'Petal width'])
sns.pairplot(d)
plt.show()
# - recognize the high covar between sepal widht and length !
# - recognize the high varians in sepal lenth and width and low varians in 
    # petal lenth and width!

""" Principal Component Analysis (PCA) """
#Subtract the mean from the data
mn = np.mean(x, axis=0)
data = x - mn

#Compute the covariance matrix (MANUALLY)
c_x = 1 / (50 - 1) * data.T @ data 
print(c_x)

#Compute the covariance matrix with numpy.cov(data.T)
print(np.cov(data.T))
#same result!

#Now compute principal components using eigenvector analysis
values, vectors = np.linalg.eig(c_x)
#c_x is the covariance matrix
#values are the eigenvalues 
#vectors are the eigenvectors (the principal components).

# ploting the amount of explained variation for each principal compenent:
v_norm = values / values.sum() * 100
plt.plot(v_norm)
plt.xlabel("Principal component")
plt.ylabel("Percent explained variance")
plt.ylim([0, 100])
plt.show()

# Exercise 8 
# The data can be projected onto the pca space by using the dot-product.
pc_proj = vectors.T.dot(data.T)

#Try to use seaborns pairplot with the projected data? How does the covariance
#structure look like?
plt.figure() 
d = pd.DataFrame(pc_proj.T, columns=['PC1', 'PC2','PC4', 'PC3'])
sns.pairplot(d)
plt.show()
# - the covariance structure looks very random on all plots.

""" Direct PCA using the decompositions functions """
#The Python machine learning package sci-kit learn (sklearn) have several
#functions to do data decompositions, where PCA is one of them.
from sklearn import decomposition

#excersise 9
#read data matrix as before but do not subtract the mean - the procedure will do it!
x = iris_data[0:50, 0:4]

#the PCA can be computed like so:
pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_
data_transform = pca.transform(data)

#show on plot to compare with manual results
plt.figure() 
d = pd.DataFrame(data_transform, columns=['PC1', 'PC2',
'PC4', 'PC3'])
sns.pairplot(d)
plt.show()

print("comparison PCA vectors")
print(vectors)
print(vectors_pca)

















