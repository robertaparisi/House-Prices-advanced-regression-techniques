# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:56:15 2017

@author: Alessia
"""
#=============================================================================#
#                          IMPORTING PACKAGES                                 #
#=============================================================================#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, ElasticNet, Ridge, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
#from sklearn import cross_validation
#from sklearn.model_selection import KFold
import functions1548814 as lib
import imp
imp.reload(lib)
from sklearn.linear_model import Lasso, ElasticNet, Ridge, RidgeCV, LassoCV
from scipy.stats import skew


#=============================================================================#
#                          IMPORTING DATA                                     #
#=============================================================================#


train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0) 


#=============================================================================#
#                                                                             #
#                          DATA PRE-PROCESSING:                               #
#                                                                             #
#                                                                                #
#       1. Fill missing value and delete unusefull variable                   #
#       2. Delete inconsistency                                               #
#       3. Binning                                                            #  
#       4. Outlier analysis                                                   #   
#       5. Working on Categorical Feature:                                    #     
#           5.1 Aggregation of modalities                                     #
#           5.2 Rating Factorization for quality variable                     #
#           5.3 Factorization of categorical variables wrongly setted as int  #
#       6. Data Transformation                                                #
#                                                                             #
#=============================================================================#


#The following feauture is our outcome, the feauture that I want to predict
#usually all the variable like this one (earnings, price and so on) have a strong asymmetric distribution, like we can see in the first graph, and I want to have a distribution more similar to a Gaussian.  
target = train["SalePrice"]


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,8))

ax1.hist(target,50)
ax1.set_title("House's Price distribution", fontsize = 23)
ax1.set_xlabel('House Sale Price',  fontsize= 17)
ax1.set_ylabel('Frequencies',  fontsize= 15)

#For the reason that I said before, I will apply a logarithmic transformation to the outcome
logtarget = np.log1p(target)

ax2.hist(logtarget, 40)
ax2.set_title("House's Price log-distribution", fontsize = 23)
ax2.set_xlabel('House Sale Price (logaritmic scale)',  fontsize= 17)
ax2.set_ylabel('Frequencies',  fontsize= 15)
plt.close()

##### 1. Fill NA values and delete unusefull variable


#Here we have all the feautures that have NA inside, and we can see that test has more NA value, and i will analyze deeply all the feature
 
train.columns[train.isnull().any()] 
test.columns[test.isnull().any()]

#So for train the column that has 'real' NA are LotFrontage, Electrical
#For test the variables are MSZoning, LotFrontage, Utilities, Exterior1st, Exterior2nd, KitchenQual, Functional, Saletype

sum(train['LotFrontage'].isnull())/1460
sum(test['LotFrontage'].isnull())/1459

#The NA value are more than 10 percent, and since the importance of the feauture is not that much i will delete it from both the dataset (it represent the feet of street connected to the property)

train.drop('LotFrontage', axis=1, inplace=True)
test.drop('LotFrontage', axis=1, inplace=True)

#This is only one, so we can input this feauture with the mode since it is qualitative

sum(train['Electrical'].isnull())
train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)  

#The same thing for this variable that has only 4 missing value (TESTSET)

sum(test['MSZoning'].isnull())
test['MSZoning'].fillna(test['MSZoning'].mode()[0], inplace=True)  

#Here we have only two NA but looking at his distribution we can see that only 1 record assume a value different with respect to the other, so in my opinion it is not explanatory, so i will drop it

sum(test['Utilities'].isnull())
pd.crosstab(index=train['Utilities'], columns="count")
pd.crosstab(index=test['Utilities'], columns="count")
train.drop('Utilities', axis=1, inplace=True)
test.drop('Utilities', axis=1, inplace=True)

#Just 1 NA and qualitative value so, i impute it with mode

sum(test['Exterior1st'].isnull())
test['Exterior1st'].fillna(test['Exterior1st'].mode()[0], inplace=True)  

#Same like the previous one (always 1 NA, and qualitative)
sum(test['Exterior2nd'].isnull())
test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0], inplace=True) 

#IDEM (always 1 NA, and qualitative)
sum(test['KitchenQual'].isnull())
test['KitchenQual'].fillna(test['KitchenQual'].mode()[0], inplace=True)  

#IDEM (2 NA, and qualitative)
sum(test['Functional'].isnull())
test['Functional'].fillna(test['Functional'].mode()[0], inplace=True)  

#IDEM (1 NA, and qualitative)
sum(test['SaleType'].isnull())
test['SaleType'].fillna(test['SaleType'].mode()[0], inplace=True)  

#Let's check again the NA columns

train.columns[train.isnull().any()] 
test.columns[test.isnull().any()]

#The remaining missing value are all belongig to the "false" NA and now I can use the function and verify that all NA are gone

train = lib.fill_false_na (train)
test = lib.fill_false_na (test)

train.columns[train.isnull().any()] 
test.columns[test.isnull().any()]
#DONE!!!


##### 2. INCONSISTENCY
#Looking to the distribution of the various features i have found an inconsistency, regarding the year of building of the garage, that for a typo error was 2207, obviously is wrong 

pd.crosstab(index=test['GarageYrBlt'], columns="count")

test[test['GarageYrBlt']==2207] 

#The Inconsintent value has as ID 2593, so I replace it with 2007, that in my opinion is the more likely value  

test['GarageYrBlt'] = test['GarageYrBlt'].replace({2207:2007})



##### 3. BINNING

#I want to create some bins for certain variables in order to reduce the effects of minor observation.

train[['SalePrice','YearBuilt']].groupby('YearBuilt').mean().sort_values('SalePrice', ascending = False)
pd.crosstab(index=train['YearBuilt'], columns="count")
train[['SalePrice','GarageYrBlt']].groupby('GarageYrBlt').mean().sort_values('SalePrice', ascending = False)
pd.crosstab(index=train['GarageYrBlt'], columns="count")
train[['SalePrice','YearRemodAdd']].groupby('YearRemodAdd').mean().sort_values('SalePrice', ascending = False)
pd.crosstab(index=train['YearRemodAdd'], columns="count")


plt.scatter( y= train['YearBuilt'], x= train['SalePrice'],c='cyan', marker= '^', label= 'House Built')
plt.scatter( y= train['GarageYrBlt'], x= train['SalePrice'],c='lightgreen',  marker=r'$\clubsuit$', label='Garage Built')
plt.scatter( y= train['YearRemodAdd'], x= train['SalePrice'], c='pink',  marker='o', label= 'Remodel')
plt.xlabel('House Price', fontsize = 17)
plt.ylabel('Year', fontsize = 15)
plt.legend(loc=4)
plt.title("Year's features with respect to the price" , fontsize = 23)
plt.ylim(1850, 2020)

#create some group, very old house will be accorpate in bigger class because are with lower frequencies and also because 10 years in 1800 (for example 1880 e 1900) are differente if we look at the end of 1900 (for example 1990 e 2000) as we can see from the plot

#Before to do other modifications to the data i will add a variable because otherwise after it will not have the same meaning 


combi = pd.concat((train.iloc[:,:-1],test))

combi['RemodAfterSold'] =combi['YrSold'] - combi['YearRemodAdd']

combi = lib.factorization_year (combi, var='YearBuilt' , newvar= 'YearBuiltCat')        
combi = lib.factorization_year (combi, var='GarageYrBlt' , newvar= 'GarageYrBltCat') 
combi = lib.factorization_year (combi, var='YearRemodAdd' , newvar= 'YearRemodAddCat') 


##### 4. OUTLIER 

#One of the first feature that I think could represent an outlier is the size of the house

pd.crosstab(index=train['GrLivArea'], columns="count")
pd.crosstab(index=test['GrLivArea'], columns="count")

#Looking to the "Alternative to the Boston Housing Data as an End of Semester Regression Project" I scovered that is best to drop the house that has a living area bigger than 4000 squared foot, because could represent outliers

fig1, (axs1, axs2) = plt.subplots(2, 2, sharey=True, figsize=(20,8))
axs11 =sns.boxplot(train['GrLivArea'],orient='v', ax=axs1[0])
axs11.set_ylabel ("Area (in square feat)", fontsize = 16)
axs11.set_xlabel ("Ground Living Area", fontsize = 16)
axs11.set_title ("Boxplot with outliers", fontsize = 23)
axs12 =sns.regplot(train['SalePrice'],train['GrLivArea'],ax=axs1[1])
axs12.set_ylabel ("Sale price (in $)", fontsize = 14)
axs12.set_xlabel ("Ground Living Area", fontsize = 14)
axs12.set_xlim(0, 800000)
axs12.set_title ("Ground living Area distribution with respect\n to the Sale Price (with outliers)", fontsize = 19)


#As we could see from the plot the distribution has really few value higher than 4000 and at the same moment an house with this size has also strange sale price(really low) 

train = train[train['GrLivArea']<=4000] #524, 692, 1183,1299
combi = combi.drop(combi.index[[523, 691, 1182, 1298]])

axs21 = sns.boxplot(train['GrLivArea'],orient='v', ax= axs2[0])
axs21.set_ylabel ("Area (in square feat)", fontsize = 16)
axs21.set_xlabel ("Ground Living Area", fontsize = 16)
axs21.set_title ("Boxplot without outlier", fontsize = 23)
axs22 = sns.regplot(train['SalePrice'],train['GrLivArea'], ax=axs2[1])
axs22.set_ylabel ("Sale price (in $)", fontsize = 14)
axs22.set_xlabel ("Ground Living Area", fontsize = 14)
axs22.set_xlim(0, 800000)
axs22.set_title ("Ground living Area distribution with respect to the Sale Price (without outliers)", fontsize = 19)

combi.columns[combi.isnull().any()]

##### 5. Working on Categorical Feature:

### 5.1 Aggregation of modalities

pd.crosstab(index=combi['MSZoning'], columns="count")
train[['SalePrice','MSZoning']].groupby('MSZoning').max().sort_values('SalePrice', ascending = False)
#train.boxplot(column="SalePrice", by="MSZoning")

#looking to this boxplot and to the crosstable, since with the modalities RH (Residential High density) there are only 16 house (10 in the test) I think that is ok to merge this modalities with RM (Residential Medium density) where the distribution is similar!
combi['MSZoning'] = combi['MSZoning'].replace({'RH':'RHM','RM': 'RHM'}) 



pd.crosstab(index=combi['MSSubClass'], columns="count")
train[['SalePrice','MSSubClass']].groupby('MSSubClass').max().sort_values('SalePrice', ascending = False)
#train.boxplot(column="SalePrice", by="MSSubClass")

#since there is one modalities that has only 1 record, so i decided to accorpate it with '120' because are both house of pud's type and 80 and 85 together because are both split house
combi['MSSubClass'] = combi['MSSubClass'].replace({85:80, 150: 120})



pd.crosstab(index=combi['LandSlope'], columns="count")
#train.boxplot(column="SalePrice", by="LandSlope")
combi['LandSlope'] = combi['LandSlope'].replace({'Mod':'ModSev', 'Sev':'ModSev'})

#idem for this feature
pd.crosstab(index=combi['SaleCondition'], columns="count")
#train.boxplot(column="SalePrice", by="SaleCondition")
combi['SaleCondition'] = combi['SaleCondition'].replace({'Abnorml': 0, 'AdjLand': 0, 'Alloca': 0, 'Family':0, 'Normal':1, 'Partial':2})


### 5.2 Rating Factorization for quality variables

combi = lib.quality_var_and_ricategorizations(combi)
    
#Since that OverallQual and OverallCond are the only ones to have a value scale from 1 to 10, I want to create a new variable that is more similar to the other present in the dataset (between 0 and 5)
#I look at the distribution of this two features to see what will be the best solution in order to have a more clear idea of how to reconstruct the features

#pd.crosstab(index=train['OverallQual'], columns="count")
#pd.crosstab(index=test['OverallQual'], columns="count")
#train[['SalePrice','OverallQual']].groupby('OverallQual').mean().sort_values('SalePrice', ascending = False)
#train.boxplot(column="SalePrice", by='OverallCond')
#pd.crosstab(index=train['OverallCond'], columns="count")
#pd.crosstab(index=test['OverallCond'], columns="count")
#train[['SalePrice','OverallCond']].groupby('OverallCond').mean().sort_values('SalePrice', ascending = False)
#train.boxplot(column="SalePrice", by='OverallCond')

combi['OverallQualRevised']=combi['OverallQual'].copy()
combi['OverallQualRevised']= lib.over_allrev (combi, var='OverallQualRevised')
 

combi['OverallCondRevised']=combi['OverallCond'].copy()
combi['OverallCondRevised']= lib.over_allrev(combi, var='OverallCondRevised')  

###  5.3 Factorization of categorical variables wrongly setted as int

columns='MSSubClass','OverallQual', 'OverallCond', 'YearBuilt', 'GarageYrBlt', 'YearRemodAdd', 'ExterQual', 'ExterCond', 'BsmtQual','BsmtFinType1',  'BsmtFinType2', 'HeatingQC',  'KitchenQual','FireplaceQu', 'GarageQual', 'GarageCond','PoolQC', 'Fence', 'MoSold', 'YrSold', 'SaleCondition', 'OverallQualRevised', 'OverallCondRevised'

for i in columns:
    combi[i]= combi[i].astype(object) 


####   6. Data Transformations 


#To avoid the dropping of a lot of data but at the sametime to minimize the impact of the outliers I computed for each numerical feature that has a value of skeweness bigger than 0.8 I applied the log transformation (1+x in order to avoid problem) 

numeric= (combi.select_dtypes(exclude = ["object"]).columns)
skewness =combi[numeric].apply(lambda x: skew(x))
skewness= skewness[abs(skewness) > 0.8]
combi[skewness.index] = np.log1p(combi[skewness.index])

#scaler = preprocessing.StandardScaler()
#scaler.fit(c[numeric])
#scaled = scaler.transform([combi_num.columns])
#for i, col in enumerate(combi_num.columns):
#   combi_df[col] = scaled[:, i]


#=============================================================================#
#                          FEATURE ENGINEERING:                               #
#       1. Adding new features                                                # 
#                                                                             #
#       2. Creating dummies variable for categorical features                 #
#                                                                             #
#=============================================================================#

#####    1. Adding new features   

#to explain in a better way the relations between variable, basing on internet research and my previous experience about regressions' model I create some new features, i.e. some interactions between different variable.


combi = lib.Create_new_features(combi)


#####   2. Creating dummies variable for categorical features  

combi = pd.get_dummies(combi)


#=============================================================================#
#                          MODEL TRAINING                                     #
#       1. Adding new features                                                # 
#                                                                             #
#       2. Creating dummies variable for categorical features                 #
#                                                                             #                                                                             
#=============================================================================#

#lets define the train and test set

logtarget = np.log1p(train['SalePrice'])
train = combi[:train.shape[0]]
test = combi[train.shape[0]:]

Xtr, Xte , ytr, yte = train_test_split(train, logtarget, test_size=0.2, random_state=1)

#Since the linear regression is not good on my dataset because there are too much features so i have to pick another type of regression, i opted for a lasso, that tend to shrink the coefficents of the features to 0 in order to don't give too much importance at one features

lasso = Lasso(max_iter=1000000, normalize=True)
alphasl = [0.00001, 0.0001,0.0002,0.0004,0.005, 0.01 ]
coefs = []
for a in alphasl:
    lasso.set_params(alpha=a)
    lasso.fit(train, logtarget)
    coefs.append(lasso.coef_)


#lets compute the rmse for various alpha value  
cv_lasso = [lib.rmse_cv(Lasso(alpha = alpha), train, logtarget) for alpha in alphasl]

cv_lasso = pd.Series(cv_lasso, index = alphasl)

# Plot of the RMSE for each alpha tested

fig1, (axs1, axs2) = plt.subplots(1, 2, sharey=True, figsize=(20,8))

ax1=cv_lasso.plot(title = 'RMSE value for each Alpha computing a Lasso Model', ax=axs1)
plt.xlabel('Alpha values')
plt.ylabel('RMSE values ')

cv_lasso.min() #0.012793087177588843

#lets try to compute the mse also for the Ridge Model to see which one can be better 


# Ridge regression 
alphas_ridge = [0.05, 0.1, 0.3, 1, 3, 5, 10, 30] # The bigger is the alpha the less our model will be complex
cv_ridge = [lib.rmse_cv(Ridge(alpha = alpha), train, logtarget) for alpha in alphas_ridge]

cv_ridge = pd.Series(cv_ridge, index = alphas_ridge)

# Plot of the RMSE for each alpha tested
ax2=cv_ridge.plot(title = 'RMSE value for each Alpha computing a Ridge Model', ax=axs2)
plt.xlabel('Alpha values')
plt.ylabel('RMSE values ')
plt.show()

cv_ridge.min() # 0.013811473491252555
#so since with the lasso we can reach a bettr RMSE value and usually with a quantity of features so high works better than the ridge i prefear to use Lasso model to predict the Sale price 

lassoCV = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
lassoCV.fit(train, logtarget)
lasso.set_params(alpha=lassoCV.alpha_)
alfa=lassoCV.alpha_
lasso.fit(train,logtarget)
mean_squared_error(yte, lasso.predict(Xte)) #really good value of mse

#train.columns[pd.Series(lasso.coef_, index=train.columns)>0.05]
#Index(['LotArea', 'GrLivArea', 'Neighborhood_Crawfor', 'Neighborhood_StoneBr',YearBuilt_1932', 'Functional_Typ', 'PoolQC_3', 'GarageYrBltCat_14'], dtype='object')

#here we have the lasso score, and is a good one, so we use the model to predict on the test set

lasso.score(train, logtarget) #0.93718564646405356
predict_l=lasso.predict(test)
predict_l=np.expm1(predict_l)



#Plot of the goodness of my prediction between Train set and SalePrices
predX_lasso = np.expm1(lasso.predict(train))
plt.figure()
plt.scatter(predX_lasso, np.expm1(logtarget))
plt.title('Prediction vs SalePrice for Lasso', fontsize=23)
plt.xlabel('Prediction Sale Price', fontsize=17)
plt.ylabel('Observed Sale Price', fontsize=16)
plt.plot([min(predX_lasso),max(predX_lasso)], [min(predX_lasso),max(predX_lasso)], c="red")

#I'm not completely satisfied of the model, looking at this last plot, so i tried to fit another kind of model, Elastic Net, that use a linear combination between the 2 regularization (L1 and L2) and so between Lasso and Ridge. I choose as l1_ratio 0.7 to give more importance to the Lasso regularization since we see that works better


ela_net = ElasticNet(l1_ratio = 0.7, alpha =alfa, max_iter=70000) 
ela_net.fit(train,logtarget )
ela_net.score(train, logtarget)

#prediction
pred_ela = np.expm1(ela_net.predict(test)) # SalePrice prediction using Elastic Net
predX_elan = np.expm1(ela_net.predict(train))

plt.scatter(predX_elan, np.expm1(logtarget))
plt.title('Prediction vs SalePrice for Elastic Net', fontsize=23)
plt.xlabel('Prediction Sale Price', fontsize=17)
plt.ylabel('SalePrice', fontsize=16)
plt.plot([min(predX_elan),max(predX_elan)], [min(predX_elan),max(predX_elan)], c="red")

#not enough good neither this one (also for kaggle score), so i decide to do an ensemble model, mixing Lasso and Elasticnet

predEns=0.7*(predict_l)+ 0.3*pred_ela
predXEns = 0.7*np.expm1(lasso.predict(train))+ 0.3*np.expm1(ela_net.predict(train))

plt.scatter(predXEns, np.expm1(logtarget), color='yellow')
plt.title('Prediction vs SalePrice for Ensemble Model', fontsize=23)
plt.xlabel('Prediction Sale Price', fontsize=17)
plt.ylabel('SalePrice', fontsize=16)
plt.plot([min(predXEns),max(predXEns)], [min(predXEns),max(predXEns)], c="red")



solution = pd.DataFrame({'SalePrice' :predEns}, index = test.index)
#
#preds1 = pd.DataFrame({'SalePrice' :pred_ela}, index = test.index)
#
#preds2 = pd.DataFrame({'SalePrice' : 0.7*(np.expm1(lasso.predict(test)))+ 0.3*pred_ela}, index = test.index)
##risultato migliore ottenuto con 0.3 e 0.7
##provato a cambiare 0.
#
#preds3 = pd.DataFrame({"SalePrice":np.exp(lasso.predict(test))}, index=test.index)
#preds1.to_csv("pred1.csv")
#preds2.to_csv("pred2.csv")
#preds3.to_csv("pred3.csv")
#
solution.to_csv("solution.csv")
#
#ela_net2 = ElasticNet(l1_ratio = 0.8, alpha =alfa, max_iter=70000) 
#ela_net2.fit(train,logtarget )
#ela_net2.score(train, logtarget)
#pred_ela2 = np.expm1(ela_net2.predict(test)) # SalePrice prediction using Elastic Net
#
#
#preds22 = pd.DataFrame({'SalePrice' : 0.7*(np.expm1(lasso.predict(test)))+ 0.3*pred_ela2}, index = test.index)
#preds22.to_csv("pred22.csv")