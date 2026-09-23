import os
import sys
import time
from dataclasses import dataclass

import pandas as pd 
import numpy as np

#model importing 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#import evaluation matrics
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix, classification_report)

from imblearn.over_sampling import SMOTE


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


#defining dataclass 

@dataclass
class ModelTrainerConfig:
    train_data_path = os.path.join("data","final","train_selected.csv")

    test_data_path = os.path.join( "data","final","test_selected.csv")

    model_dir = os.path.join("artifacts", "models")

    report_path = os.path.join("reports","model_training_report.csv")


#model Trainer class

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

        self.models={"Logistic Regression": LogisticRegression(random_state=42,max_iter=1000),
                 "Decision Tree": DecisionTreeClassifier(random_state=42),
                 "Random Forest": RandomForestClassifier(random_state=42,n_estimators=200),
                 "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                 "XGBoost": XGBClassifier(random_state=42,eval_metric="logloss"),
                 "LightGBM": LGBMClassifier(random_state=42,verbose=-1),
                 "CatBoost": CatBoostClassifier( random_state=42,verbose=0),
                 "KNN": KNeighborsClassifier(n_neighbors=5)}

    logging.info("Model Trainer initialized successfully.")


    def load_data(self):
        try:
            logging.info("loadinf train and test dataset final after doinf feature engineering")
            train_df=pd.read_csv(self.config.train_data_path)
            test_df=pd.read_csv(self.config.test_data_path)

            logging.info(f"Train dataset shape{train_df.shape}")
            logging.info(f"Test dataset shape {test_df.shape}")

            #seprate feature anf target
            X_train=train_df.drop(columns=["Churn"],axis=1)
            y_train=train_df["Churn"]
            X_test=test_df.drop(columns=["Churn"],axis=1)
            y_test=test_df["Churn"]

            return(X_train,X_test,y_train,y_test)

        except Exception as e:
             raise CustomException(e, sys)

    #apply smote to treat imbalancing problem that we have anayzed in EDA 
    def apply_smote(self,X_train,y_train):

        try:
            logging.info("Applying SMOTE")

            smote=SMOTE(sampling_strategy="auto",random_state=42,k_neighbors=5)
            X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)
            logging.info("smote applied")
            logging.info(f"Shape before SMOTE{X_train.shape}")
            logging.info(f"Shape after SMOTE {X_train_smote.shape}")


            balanced_distribution = y_train_smote.value_counts().rename(index={0: "No Churn", 1: "Churn"})

            print("\nBalanced Training Distribution")
            print(balanced_distribution)

            return (X_train_smote,y_train_smote)

        except Exception as e:
            raise CustomException(e, sys)

            ## model training 

    def train_models(self,X_train_smote,y_train_smote):
        try:
            logging.info("Model training Starting")

            trained_model={}
            training_time={}

            for model_name,model in self.models.items():
                logging.info(f"Training {model_name}")
                start_time=time.time()

                model.fit(X_train_smote,y_train_smote)

                end_time=time.time()

                trained_model[model_name]=model
                training_time[model_name]=round(end_time-start_time,3)

                logging.info(f"{model_name} trained successfully" f"{training_time[model_name]} in seconds")

            return trained_model,training_time 

        except Exception as e:
                 raise CustomException(e, sys)


    ## Evaluating MODELS

    def evaluate_models(self,trained_models,training_time, X_test, y_test):
        try:
            logging.info("Starting model evaluation.")

        # Create folders if they don't exist
            os.makedirs(self.config.model_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)

            evaluation_results = []

            for model_name, model in trained_models.items():
                logging.info(f"Evaluating {model_name}")

            # Predictions
                y_pred = model.predict(X_test)

            # Probability predictions for ROC-AUC
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.decision_function(X_test)

            # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_prob)

            # Save model
                model_filename = (model_name.lower().replace(" ", "_")+ ".pkl")

                save_object(file_path=os.path.join(self.config.model_dir, model_filename),obj=model)

                logging.info(f"{model_name} saved successfully.")

                evaluation_results.append({
                "Model": model_name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "ROC AUC": round(roc_auc, 4),
                "Training Time (Seconds)": training_time[model_name]})

        # Create comparison DataFrame
                     # Create comparison DataFrame
            results_df = pd.DataFrame(evaluation_results)

            results_df=results_df.sort_values(by="ROC AUC",ascending=False).reset_index(drop=True)

            # add ranking column
            results_df["AUC ROC RANK"]= results_df["ROC AUC"].rank(method="dense",ascending=False).astype(int)

            results_df["Accuracy_Rank"]=results_df["Accuracy"].rank(method="dense",ascending=False).astype(int)

            results_df.to_csv(self.config.report_path,index=False)

            logging.info("Model training report saved successfully.")
            logging.info("Model evaluation completed successfully.")

            return results_df

        except Exception as e:
            raise CustomException(e, sys)







