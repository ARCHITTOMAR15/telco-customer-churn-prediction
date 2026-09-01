
import pandas as pd
import numpy as np
import sys 

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml

class DataIngestion:
    def __init__(self,config_path="config/config.yaml"):
        self.config=read_yaml(config_path)

    def load_data(self):
        try:
            data_path=self.config["data"]["raw_data_path"]
            logger.info(f"Loading Dataset from Data path {data_path}")
            df=pd.read_csv(data_path)
            logger.info("Dataset Loaded Successfully")

            return df
        except Exception as e:
            logger.error("Error Occured while loading Dataset")
            raise CustomException(e,sys)

    @staticmethod
    def dataset_summary(df):
      summary = pd.DataFrame({
        "Data Types": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "Missing Percentage": (df.isnull().sum() / len(df) * 100).round(2),
        "Unique Values": df.nunique()
    })

      return summary

    @staticmethod
    def feature_types(df):
        categorical_columns=(df.select_dtypes(include="object").columns.tolist())
        numerical_columns=(df.select_dtypes(exclude="object").columns.tolist())

        return categorical_columns,numerical_columns

    @staticmethod

    def target_summary(df,target_column):
        summary=pd.DataFrame({"Count":df[target_column].value_counts(),
                              "Percentage":(df[target_column].value_counts(normalize=True)*100).round(2)})
        return summary








