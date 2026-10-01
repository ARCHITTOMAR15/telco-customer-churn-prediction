
import os
import sys 
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,Dropout,BatchNormalization)

from tensorflow.keras.callbacks import (EarlyStopping,ModelCheckpoint,ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report)

from src.exception import CustomException
from src.logger import logging

#DataClass

@dataclass
class ANNTrainerConfig:
     train_data_path = os.path.join("data","final","train_selected.csv")

     test_data_path = os.path.join("data","final","test_selected.csv")

     model_dir = os.path.join("artifacts","models")

     ann_model_path = os.path.join("artifacts","models","ann_model.keras")

     history_path = os.path.join( "reports","ann_training_history.csv")

     ann_report_path = os.path.join("reports","ann_model_report.csv")


class ANNTrainer:
    def __init__(self):
        self.config= ANNTrainerConfig()

        os.makedirs(self.config.model_dir,exist_ok=True)

        os.makedirs(os.path.dirname(self.config.history_path),exist_ok=True)

        logging.info("ANN Trainer initialized successfully.")


    def load_data(self):
        try:
            logging.info("Loading  train and test csv file for ANN")

            train_df=pd.read_csv(self.config.train_data_path)
            test_df=pd.read_csv(self.config.test_data_path)

            logging.info(f" train data shape{train_df.shape}")
            logging.info(f"test data shape{test_df.shape}")

            X_train=train_df.drop(columns=["Churn"],axis=1)
            y_train=train_df["Churn"]

            X_test=test_df.drop(columns=["Churn"],axis=1)
            y_test=test_df["Churn"]

            return X_train,X_test,y_train,y_test

        except Exception as e:
            raise CustomException(e, sys)


    def prepare_training_data(self,X_train,y_train):
        try:
            logging.info("Applying SMOTE to cure imbalance")

            smote=SMOTE(random_state=42)

            X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

            logging.info(f"Shape after applying SMOTE {X_train_smote.shape}")

            logging.info("Creating Validation dataset for ANN  on SMOTE dataset")

            X_train_final,X_val,y_train_final,y_val=train_test_split(X_train_smote,y_train_smote,test_size=0.2,
                                                                     random_state=42,stratify=y_train_smote)

            logging.info(f"Training Data shape:{X_train_final.shape}")
            logging.info(f" Validataon shape:{X_val.shape}")

            return (X_train_final,X_val,y_train_final,y_val)

        except Exception as e:
            raise CustomException(e, sys)


    def build_ann_model(self,input_dim):
        try:
            logging.info("Building ANN MODEL")

            model=Sequential([
                # first layer input+hidden

                Dense(units=64,activation="relu",input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),

                #hidden layer 2 
                Dense(units=32,activation="relu"),
                BatchNormalization(),
                Dropout(0.2),

                #output layer
                Dense(units=1,activation="sigmoid")])

            #compile model
            model.compile(optimizer= Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=["accuracy"])
            logging.info("Model Created sucessfully")

            return model
        except Exception as e:
            raise CustomException(e, sys)

    def train_ann_model(self,model,X_train,y_train,X_val,y_val):
        try:
            logging.info("Starting Training")
            #early stopping
            early_stopping = EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True,verbose=1)

        # ---------------- Save Best Model ----------------
            checkpoint = ModelCheckpoint(filepath=self.config.ann_model_path,monitor="val_loss",save_best_only=True,verbose=1)
            #reduce lr rate
            reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=5,min_lr=1e-6,verbose=1)

            start_time=time.time()

            history=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=40,batch_size=32,callbacks=[early_stopping,checkpoint,reduce_lr],
            verbose=1)

            end_time=time.time()

            training_time = round(end_time - start_time, 2)

            logging.info(f"training completed in {training_time} seconds.")

            return history, training_time

        except Exception as e:
            raise CustomException(e, sys)


    def evaluate_ann_model(self,model,history,training_time,X_test,y_test):
        try:
            logging.info("Model Evaluation")

            y_prob=model.predict(X_test,verbose=0).flatten()
            y_pred=(y_prob>=0.5).astype(int)

            #metrics

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            class_report=classification_report(y_test,y_pred)

            #training history 
            history_df=pd.DataFrame(history.history)
            history_df.to_csv(self.config.history_path,index=False)

            logging.info("ANN training history saved successfully.")

            # Ann Report 
            ann_results=pd.DataFrame([{"Model": "ANN",
            "Accuracy": round(accuracy, 4),"Precision": round(precision, 4),
            "Recall": round(recall, 4),"F1 Score": round(f1, 4),
            "ROC AUC": round(roc_auc, 4),"Training Time (Seconds)": training_time}])

            ann_results.to_csv(self.config.ann_report_path,index=False)

            logging.info("ANN evaluation completed successfully.")

            return ann_results, history_df

        except Exception as e:
            raise CustomException(e, sys)



























