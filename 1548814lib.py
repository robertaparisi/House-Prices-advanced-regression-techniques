# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:34:57 2017

@author: Alessia
"""
import sklearn
from sklearn.model_selection import cross_val_score
#=============================================================================#
#                                                                             #
#                          DATA PRE-PROCESSING:                               #
#                                                                             #
#       Data cleaning                                                         #
#       1. Fill missing value and delete unusefull variable                   #
#       2. Delete inconsistency                                               #
#       3. Binning                                                            #  
#       4. Outlier analysis                                                   #   
#       5. Working on Categorical Feature:                                    #     
#           5.1 Aggregation of modalities                                     #  
#           5.2 Factorization of categorical variable wrongly setted as int   #       
#           5.2 Rating Factorization for quality variable                     #
#           5.3 Encoding categorical feature                                  #
#       6. Data Transformation                                                #
#                                                                             #
#=============================================================================#

#       1. Fill missing value and delete unusefull variable 


#Some of the feautures assume na value when that particolar feauture is not present, and this information is write in the file that explane the variables. 

#This function will impute rispectivly the missing value of which i talk about before and that i called 'false' NA

def fill_false_na (dataframe):
    df=dataframe.copy()
    Nobasmt = "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"
    for n in Nobasmt:
       df[n].fillna("NoBasement", inplace=True) 
    Nobasmt_size = "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"
    for n in Nobasmt_size:
       df[n].fillna(0, inplace=True)                
    Nogar = "GarageType","GarageFinish", "GarageQual","GarageCond"
    for n in Nogar:
        df[n].fillna("NoGarage", inplace=True)
    Nogar_size = "GarageYrBlt", "GarageArea", "GarageCars"
    for n in Nogar_size:
        df[n].fillna(0, inplace=True)   
    df["FireplaceQu"].fillna("NoFireplace", inplace=True)  
    df["PoolQC"].fillna("NoPool", inplace=True)  
    df["Fence"].fillna("Fence", inplace=True) 
    df["Alley"].fillna("NoMiscFeature", inplace=True) 
    df["MiscFeature"].fillna("NoMiscFeature", inplace=True) 
    df["MasVnrType"].fillna("None", inplace=True) 
    df["MasVnrArea"].fillna(0, inplace=True) 
    return (df)

#       3. Binning

def factorization_year (dataset, var, newvar):
    d={'0': [0], '1':list(range(1870, 1901)), '2': list(range(1901,1921)), '3': list(range(1921, 1936)), '4': list(range(1936, 1951)), '5': list(range(1951, 1966)), '6': list(range(1966, 1976)), '7': list(range(1976, 1986)), '8': list(range(1986, 1991)), '9': list(range(1991, 1996)), '10': list(range(1996, 2001)), '11': list(range(2001, 2004)), '12': list(range(2004, 2006)), '13': list(range(2006, 2008)), '14': list(range(2008, 2011))}
    for i in range(1, dataset.shape[0]+1):
         for k in d.keys():
             if dataset.loc[i, var] in d[k]:
                 dataset.loc[i, newvar]=k
    return(dataset)


#           5.2 Rating Factorization for quality variable                    

def change_eval_var (df, var_eval, rate):
    for var in var_eval:
        df[var] =df[var].replace(rate)
    return (df)
    
def quality_var_and_ricategorizations(dataset):
    dataset['Fence'] = dataset['Fence'].replace({'Fence':0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4})
    dataset['PoolQC'] = dataset['PoolQC'].replace({'NoPool':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    dataset['FireplaceQu']=dataset['FireplaceQu'].replace({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4, 'NoBsmt':0})
    rate1 = {'Unf':3, 'LwQ':1, 'Rec':2, 'BLQ':1, 'ALQ':4, 'GLQ':4, 'NoBasement':0}
    rate2 = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'NoGarage':0, 'NoFireplace':0}
    rate3 = {'Fa':1, 'TA':2, 'Gd':3, 'Ex':4, 'NoBasement':0}
    eval_type1 = ['BsmtFinType1', 'BsmtFinType2']   
    eval_type2 = ['HeatingQC','GarageQual', 'GarageCond', 'FireplaceQu','ExterQual', 'ExterCond', 'KitchenQual']
    eval_type3 = ['BsmtQual', 'BsmtCond']
    dataset=change_eval_var (dataset, eval_type1, rate1)
    dataset=change_eval_var (dataset, eval_type2, rate2)
    dataset=change_eval_var (dataset, eval_type3, rate3)
    return (dataset)

def over_allrev (dataset, var):
    dataset.loc[dataset[var]<=3,var] = 0
    dataset.loc[((dataset[var]<=5)&(dataset[var]>=4)),var] = 1
    dataset.loc[dataset[var]==5,var] = 1
    dataset.loc[dataset[var]==6,var] = 2
    dataset.loc[((dataset[var]<=8 )&(dataset[var]>=7)),var] = 3
    dataset.loc[dataset[var]==9,var] = 4
    dataset.loc[dataset[var]==10,var] = 5
    return(dataset[var])

#=============================================================================#
#                          FEATURE ENGINEERING:                               #
#       1. Adding new features                                                # 
#                                                                             #
#       2. Creating dummies variable for categorical features                 #
#                                                                             #
#=============================================================================#


def Create_new_features (combi_df):
    # Overall quality of the house
    combi_df["OverallGrade"] = 0.5*(combi_df["OverallQual"].astype(int) + combi_df["OverallCond"].astype(int))
    # Overall quality of the garage
    combi_df["GarageGrade"] = 0.5*(combi_df["GarageQual"].astype(int) + combi_df["GarageCond"].astype(int))
    # Overall quality of the exterior
    combi_df["ExterGrade"] = 0.5*(combi_df["ExterQual"].astype(int) + combi_df["ExterCond"].astype(int))
    # Overall kitchen score
    combi_df["KitchenScore"] = 0.5*(combi_df["KitchenAbvGr"].astype(int) + combi_df["KitchenQual"].astype(int))
    # Overall Garage score
    combi_df["GarageScore"] = 0.5*(combi_df["GarageQual"].astype(int) + combi_df["GarageCond"].astype(int))*combi_df['GarageArea']
    # Overall Pool score considering the area
    combi_df["PoolScore"] = combi_df["PoolArea"] * combi_df["PoolQC"].astype(int)
    # Have the house enough bathroom considering the room? In my opinion is good to have 1 bath for 1 room, so if an house as this quality is a plus   
    combi_df["EnoughBath"] = combi_df["FullBath"]
    combi_df.loc[((combi_df["FullBath"]+combi_df["HalfBath"])>=combi_df["BedroomAbvGr"]),"EnoughBath" ] = 1
    combi_df.loc[((combi_df["FullBath"]+combi_df["HalfBath"])<combi_df["BedroomAbvGr"]),"EnoughBath" ] = 0
    #Now i create a feature that will group together the months by the season
    combi_df["Season"] = combi_df["MoSold"].replace({'1': '1', '2':'1', '2':'2','4':'2','5':'2','6':'3','7':'3','8':'3','9':'4','10':'4','11':'4','12':'1'})
    #Basement feature considering the score and also the area
    #combi_df["Bsmt_tot"] =0.5*(combi_df["BsmtCond"].astype(int)*combi_df["BsmtQual"].astype(int))*(combi_df['BsmtFinSF1'].astype(int)+combi_df['BsmtFinSF2'].astype(int))    
    # Join '1stFlrSF' and '2ndFlrSF'
    combi_df['FlrSF'] = combi_df['1stFlrSF'] +combi_df['2ndFlrSF']
    combi_df = combi_df.drop(['2ndFlrSF', '1stFlrSF'], axis=1)
    
    # Join 'BsmtFullBath' and 'FullBath'
    combi_df['FullBath'] = combi_df['FullBath'] +combi_df['BsmtFullBath']
    combi_df = combi_df.drop(['FullBath','BsmtFullBath'], axis=1)
 
    # Join 'HalfBath' and 'BsmtHalfBath'
    combi_df['HalfBath'] = combi_df['HalfBath'] +combi_df['BsmtHalfBath']
    combi_df = combi_df.drop(['HalfBath','BsmtHalfBath'], axis=1)
    
    #Join 'EnclosedPorch', '3SsnPorch'
    combi_df['EnclosedPorch'] = combi_df['EnclosedPorch'] + combi_df['3SsnPorch']
    combi_df = combi_df.drop('3SsnPorch' , axis=1)
    
    return (combi_df)


#=============================================================================#
#                          MODEL TRAINING                                     #
#       1. Adding new features                                                # 
#                                                                             #
#       2. Creating dummies variable for categorical features                 #
#                                                                             #
#       2. Creating dummies variable for categorical features                 #
#                                                                             #
#       2. Creating dummies variable for categorical features                 #
#                                                                             #
#=============================================================================#

  
def rmse_cv(model, X, y):
     return (-cross_val_score(model, X, y, scoring='neg_mean_squared_error')).mean()