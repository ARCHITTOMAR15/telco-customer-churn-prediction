
import os
import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler 
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )

    train_data_path = os.path.join(
        "data",
        "processed",
        "train.csv"
    )

    test_data_path = os.path.join(
        "data",
        "processed",
        "test.csv"
    )


class DataPreprocessing:

    def __init__(self):
        self.config = DataPreprocessingConfig()

    def get_data_preprocessor(self):

        try:
            logging.info("creating preprocessing pipeline")
            numerical_columns=["SeniorCitizen","tenure","MonthlyCharges","TotalCharges","TotalServices","AvgMonthlySpend","IsLongTermCustomer",
                               "HasStreamingServices","HasSecurityServices","HasFamily","MonthlyChargePerService","ContractRisk","PaperlessAutoPay"]
            categorical_columns=["gender", "Partner","Dependents", "PhoneService","MultipleLines","InternetService","OnlineSecurity",
                                 "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract",
                                 "PaperlessBilling","PaymentMethod","TenureGroup"]

            #numerical pipeline
            num_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                         ("scaler",StandardScaler())])
            cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                         ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"))])

            preprocessor=ColumnTransformer(transformers=[("num_pipeline",num_pipeline,numerical_columns),
                                                         ("cat_pipeline",cat_pipeline,categorical_columns)])

            logging.info("Preprocessing pipeline created successfully.")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)




    def initiate_data_preprocessing(self, raw_data_path):
        try:
            logging.info("read raw data set ")

            df=pd.read_csv(raw_data_path)

            logging.info("Dataset Loaded Successfully")


            #convert total charges to numeric

            df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

            #encode target variable 
            df["Churn"]=df["Churn"].map({"No":0,"Yes":1})

            logging.info("Basic Data Cleaning Completed")

            #seprate features and Target columns 

            if "CustomerID" in df.columns:
                df = df.drop(columns=["customerID"])

            X= df.drop(columns=["Churn"],axis=1)
            y=df["Churn"]

            #train_test_split
            X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

            logging.info("Train_test_split completed")

            # Get Preprocessor
            preprocessing_obj = self.get_data_preprocessor()

            #fir on train and transfor test
            X_train_processed=preprocessing_obj.fit_transform(X_train)
            X_test_processed = preprocessing_obj.transform(X_test)

            logging.info("Data Preprocessing Completed")

             # Combine Features and Target
            train_arr = np.c_[
                X_train_processed,
                np.array(y_train)
            ]

            test_arr = np.c_[
                X_test_processed,
                np.array(y_test)
            ]

            # Create output folder
            os.makedirs(
                os.path.dirname(
                    self.config.preprocessor_obj_file_path
                ),
                exist_ok=True
            )

            # Save Preprocessor
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor saved successfully.")

            # Save processed datasets
            pd.DataFrame(train_arr).to_csv(
                self.config.train_data_path,
                index=False
            )

            pd.DataFrame(test_arr).to_csv(
                self.config.test_data_path,
                index=False
            )

            logging.info("Processed train and test datasets saved.")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)




