
import pandas as pd 

class DataValidation:

    @staticmethod
    def dataset_health(df):
        return pd.DataFrame({"Data Type":df.dtypes,
                             "Missing Values":df.isnull().sum(),
                             "Missing Percentage":(df.isnull().sum()/len(df)*100).round(2),
                             "Unique Values":df.nunique()})
    @staticmethod
    def categorical_and_numerical(df):
        categorical=(df.select_dtypes(include="object").columns.tolist())
        numerical=(df.select_dtypes(exclude="object").columns.tolist())

        return categorical,numerical

    @staticmethod

    def missing_summary(df):
        missing=pd.DataFrame({"Missing Values":df.inull().sum(),
                          "Missing Percentage":(df.innull().sum()/len(df)*100).round(2)})

        return missing.sort_values(by="Missing Percentage",ascendinf=False)


    @staticmethod 
    def duplicate_summary(df):
        return df.duplicated().sum()



