
import os 
import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd 

from src.exception import CustomException
from src.logger import logging


@dataclass
class FeatureEngineeringConfig:

    raw_data_path = os.path.join("data","raw","WA_Fn-UseC_-Telco-Customer-Churn.csv")

    engineered_data_path = os.path.join("data","engineered","telco_engineered.csv")



class FeatureEngineering:
    def __init__(self):
        self.config=FeatureEngineeringConfig()


    def create_features(self,df):
        try:
            logging.info("Starting feature engineering.")

            df["TenureGroup"]=pd.cut(df["tenure"],bins=[0, 12, 24, 48, 72],labels=["New","Regular","Loyal","Very_Loyal"],include_lowest=True)

            service_columns = ["PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                                "StreamingTV","StreamingMovies"]

            df["TotalServices"]=(df[service_columns].isin(["Yes", "Fiber optic", "DSL"]).sum(axis=1))

            df["AvgMonthlySpend"]=(df["TotalCharges"]/df["tenure"].replace(0,np.nan))
            df["AvgMonthlySpend"] = (df["AvgMonthlySpend"].fillna(df["MonthlyCharges"]))

            df["IsLongTermCustomer"]=(df["tenure"]>=24).astype(int)
            df["HasStreamingServices"]=((df["StreamingTV"]=="Yes")|(df["StreamingMovies"] == "Yes")).astype(int)
            df["HasSecurityServices"] = ((df["OnlineSecurity"] == "Yes") |(df["TechSupport"] == "Yes")).astype(int)
            df["HasFamily"] = ((df["Partner"] == "Yes") |(df["Dependents"] == "Yes")).astype(int)
            df["MonthlyChargePerService"] = (df["MonthlyCharges"] /df["TotalServices"].replace(0, 1) )
            contract_map={"Month-to-month": 2,"One year": 1,"Two year": 0}
            df["ContractRisk"] = (df["Contract"].map(contract_map))
            df["PaperlessAutoPay"] = ((df["PaperlessBilling"] == "Yes") &
                (
                    (df["PaymentMethod"] == "Bank transfer (automatic)") |
                    (df["PaymentMethod"] == "Credit card (automatic)")
                )).astype(int)

            logging.info("Feature engineering completed successfully.")

            return df

        except Exception as e:
            raise CustomException(e, sys)



    def initiate_feature_engineering(self):
        try:



            logging.info("Reading Dataset")

            df=pd.read_csv(self.config.raw_data_path)

            logging.info("Raw dataset loaded successfully.")

            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

            df = self.create_features(df)

            os.makedirs(os.path.dirname(self.config.engineered_data_path),exist_ok=True)

            df.to_csv(self.config.engineered_data_path,index=False)

            logging.info("Engineered dataset saved successfully.")

            return df


        except Exception as e:
            raise CustomException(e, sys)

