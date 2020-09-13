
#missing + inc
#import libraries
import pandas as pd
import numpy as np

#read-in the data
df = pd.read_csv("titanic_traning.csv")
#print(df)

df_missing = df.isna()
#number of missing values per feature
missing_values = df_missing.sum().drop("ID")
#percentage of missing values per feature using mean
percent_missing = df_missing.mean()

#check inconsistency for each feature
inc_pcl = df["pclass"].value_counts(dropna=False)
inc_sex = df["sex"].value_counts(dropna=False)
inc_age = df["age"].value_counts(dropna=False)
inc_sib = df["sibsp"].value_counts(dropna=False)
inc_par = df["parch"].value_counts(dropna=False)
inc_far = df["fare"].value_counts(dropna=False)
inc_emb = df["embarked"].value_counts(dropna=False)
inc_sur = df["survived"].value_counts(dropna=False)

#make a Series of inconsistent values for each of the features
inc_values = pd.Series([0,4,0,0,0,0,3,0], index=["pclass","sex","age","sibsp","parch","fare","embarked","survived"])
#make a Series of % inconsistency for each of the features
percent_inc = pd.Series([0,4/len(df["sex"]),0,0,0,0,3/len(df["embarked"]),0], index=["pclass","sex","age","sibsp","parch","fare","embarked","survived"])

#initilize summary table
summary = pd.DataFrame(np.arange(32).reshape((8,4)),
                       index=["pclass","sex","age","sibsp","parch","fare","embarked","survived"],
                       columns=["Missing Values (MV)","% of MV (MV/n)","Inconsistency Values (IV)","% of IV (IV/n)"])
#fill summary table
summary["Missing Values (MV)"] = missing_values
summary["% of MV (MV/n)"] = percent_missing
summary["Inconsistency Values (IV)"] = inc_values
summary["% of IV (IV/n)"] = percent_inc
#display summary table
print(summary)
print(summary.to_string())
#export summary table to 'summary.csv'
summary.to_csv('summary.csv')

#display all records with missing values
missing_data = df[df.isna().any(axis=1)]
print("\tRecords with missing values: ")
print(missing_data)
#display all records with inconsistent values
print("\tRecords with inconsistent values: ")
inc_sex_row = df[df.sex.isin(['Male', 'Female'])]
print(inc_sex_row)
inc_embarked_row = df[df.embarked.isin(['Queenstown'])]
print(inc_embarked_row)

#initilize the result dataframe 'df_fill', then fill missing quantitative values by feature mean
df_fill = df.fillna(df.mean())
#modify all feature values of sex to lower case
df_fill["sex"] = df_fill["sex"].str.lower()
#modify all feature values of embarked to only have abbreviations
df_fill["embarked"] = df_fill["embarked"].replace("Queenstown","Q")
#double-check if any missing values (Answer: No)
missing_values_test = df_fill.isna().sum()

#double-check if any inconsistent values (Answer: No)
inc_pcl_1 = df_fill["pclass"].value_counts(dropna=False)
inc_sex_1 = df_fill["sex"].value_counts(dropna=False)
inc_age_1 = df_fill["age"].value_counts(dropna=False)
inc_sib_1 = df_fill["sibsp"].value_counts(dropna=False)
inc_par_1 = df_fill["parch"].value_counts(dropna=False)
inc_far_1 = df_fill["fare"].value_counts(dropna=False)
inc_emb_1 = df_fill["embarked"].value_counts(dropna=False)
inc_sur_1 = df_fill["survived"].value_counts(dropna=False)

#store the result 'df_fill' to 'clean_titanic_traning.csv'
df_fill.to_csv('clean_titanic_traning.csv',index=False)
