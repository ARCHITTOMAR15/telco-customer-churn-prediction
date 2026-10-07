
import sys 
import os
import joblib
import pandas as pd
import time
import numpy as np 
import tensorflow as tf
import keras_tuner as kt

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from sklearn.metrics import(accuracy_score,precision_score,f1_score,recall_score,roc_auc_score)

from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging 


@dataclass
class ANNTunerConfig:

    train_data_path="data/final/train_selected.csv"
    test_data_path= "data/final/test_selected.csv"

    tuned_model_dir="artifacts/tuned_models"
    tuned_dir= "artifacts/ann_tuner"

    tuned_report_path="reports/ann_tuning_report.csv"

class ANNTuner:
    def __init__(self):
        self.config = ANNTunerConfig()

        os.makedirs(self.config.tuned_model_dir,exist_ok=True)
        os.makedirs(self.config.tuned_dir,exist_ok=True)
        os.makedirs(os.path.dirname(self.config.tuned_report_path),exist_ok=True)


    def load_data(self):

            try:
                logging.info("Loading Final Train and test data set")

                train_df=pd.read_csv(self.config.train_data_path)
                test_df=pd.read_csv(self.config.test_data_path)

                # split x and y 
                X_train=train_df.drop(columns=["Churn"])
                y_train=train_df["Churn"]

                X_test=test_df.drop(columns=["Churn"])
                y_test=test_df["Churn"]

                logging.info(f"X_train shape{X_train.shape}")
                logging.info(f"y_train shape{X_train.shape}")
                logging.info(f"X_test shape{X_test.shape}")
                logging.info(f"y_train shape{y_test.shape}")

                return(X_train,X_test,y_train,y_test)

            except Exception as e:
                raise CustomException(e, sys)


    def apply_smote(self,X_train,y_train):

        try:
            logging.info("Initilizing SMOTE to tacke Imbalancing")

            smote=SMOTE()

            X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

            logging.info(f"Value before SMOTE{X_train.value_counts()}")
            logging.info(f"Value before SMOTE{y_train.value_counts()}")
            logging.info(f"Value before SMOTE{X_train_smote.value_counts()}")
            logging.info(f"Value before SMOTE{y_train_smote.value_counts()}")

            return (X_train_smote,y_train_smote)

        except exception as e:
            raise CustomException(e, sys)

    def prepare_training_data(self,X_train_smote,y_train_smote):

        try:
            logging.info("Preparing training and validation set")

            X_train_final,X_val,y_train_final,y_val=train_test_split(X_train_smote,y_train_smote,test_size=0.2,random_state=42,
                                                                     stratify=y_train_smote)

            logging.info(f"X_train_final shape{X_train_final.shape}")
            logging.info(f"X_val shape{X_val.shape}")
            logging.info(f"y_train_final shape{y_train_final.shape}")
            logging.info(f"y_val_shape{y_val.shape}")

            return (X_train_final,X_val,y_train_final,y_val)

        except exception as e:
            raise CustomException(e, sys)


    def build_model(self,hp):

        try:
            input_dim=self.input_dim
            model=tf.keras.Sequential(name="tuned_model")
            #model define
            model.add(tf.keras.layers.Input(shape=(input_dim,)))
            #1 st dense layer 
            model.add(tf.keras.layers.Dense(hp.Int("units_1",min_value=32,max_value=256,step=32),activation="relu"))
            #dropout layer
            model.add(tf.keras.layers.Dropout(hp.Float("dropout_1",min_value=0.1,max_value=0.5,step=0.1)))
            #hidden layer 2 
            model.add(tf.keras.layers.Dense(hp.Int("units_2",min_value=16,max_value=128),activation="relu"))
            #dropout layer 2 
            model.add(tf.keras.layers.Dropout(hp.Float("dropout_2",min_value=0.1,max_value=0.5,step=0.1)))
            #output layer
            model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

            # learning rate 
            learning_rate=hp.Choice("learning_rate",values=[0.01,0.001, 0.0005,0.0001])
            # model_compile 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
            #model compile 
            model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy",tf.keras.metrics.AUC(name="auc")])

            return model

        except Exception as e:
            raise CustomException(e, sys)


        #hyper tuning

    def tune_ann(self,X_train_final,y_train_final,X_val,y_val):
         try:
             logging.info(" Keras Tuner Starting")

             self.input_dim=X_train_final.shape[1]

             logging.info(f" Ann Input dim {self.input_dim}")

             tuner=kt.Hyperband(hypermodel=self.build_model,objective=kt.Objective("val_auc",direction="max"),max_epochs=25,
                                factor=3,hyperband_iterations=1,
                                directory=self.config.tuned_dir,project_name="telco_churn_ann",overwrite=True)

             ## lets add call backs 
             #Early Stopping 
             early_stopping=(tf.keras.callbacks.EarlyStopping(monitor="val_auc",mode="max",patience=8,restore_best_weights=True,verbose=1))

             #ReduceLearning Rate 
             reduce_lr = (tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",factor=0.5, patience=4, min_lr=1e-6,verbose=1 ))

             # Start Timer 

             tune_start_time=time.time()

             #tuner search 
             tuner.search(X_train_final,y_train_final,validation_data=(X_val,y_val),epochs=50,batch_size=32,
                          callbacks=[early_stopping,reduce_lr],verbose=1)


             tuning_time=time.time()-tune_start_time
             logging.info("Hyperband Tuning Completed")


             #get best hyperparameter
             best_hps=(tuner.get_best_hyperparameters(num_trials=1)[0])
             logging.info("Best Hyper Parameter")

             logging.info(f"units_1: "f"{best_hps.get('units_1')}")
             logging.info(f"dropout_1: "f"{best_hps.get('dropout_1')}")
             logging.info(f"units_2: "f"{best_hps.get('units_2')}")
             logging.info(f"dropout_2: "f"{best_hps.get('dropout_2')}")
             logging.info(f"learning_rate: "f"{best_hps.get('learning_rate')}")

             #get best model
             best_model=(tuner.get_best_models(num_models=1)[0])
             best_trial=(tuner.oracle.get_best_trials(num_trials=1)[0])
             best_val_auc=best_trial.score

             return(tuner,best_model,best_hps,best_val_auc,tuning_time)

         except Exception as e :
            raise CustomException(e, sys)


    def evaluate_tuned_ann(self,best_model,X_test,y_test,best_hps,best_val_auc,tuning_time):

        try:
            logging.info(" Stating evaluation on on best model")

            y_prob=best_model.predict(X_test,verbose=0).flatten()
            y_pred= (y_prob>=0.5).astype(int)


            #metrics

            accuracy=accuracy_score(y_test,y_pred)

            precision=precision_score(y_test,y_pred,zero_division=0)

            recall=recall_score(y_test,y_pred,zero_division=0) 
            f_score=f1_score(y_test,y_pred,zero_division=0)

            roc_auc=roc_auc_score(y_test,y_prob)

            logging.info(f"Accuracy is :{accuracy}")
            logging.info(f"precision is :{precision}")
            logging.info(f"recall is :{recall}")
            logging.info(f"roc_auc_score is :{roc_auc}")


            #model saving 

            model_path = os.path.join(self.config.tuned_model_dir,"ann_tuned.keras")

            best_model.save(model_path)

            logging.info("model saved to {model_path}")

            # creating results 
            results = {"Model": "ANN Tuned", "Accuracy": accuracy,"Precision": precision,"Recall": recall,
                        "F1 Score":f_score,"ROC AUC": roc_auc,"Best Validation ROC AUC":best_val_auc,"Tuning Time":tuning_time,
                        "units_1": best_hps.get("units_1"),"units_2":best_hps.get("units_2"),"dropout_1":best_hps.get("dropout_1"),
                        "dropout_2":best_hps.get("dropout_2"),"learning_rate":best_hps.get("learning_rate")}


            results_df=pd.DataFrame([results])

            results_df.to_csv( self.config.tuned_report_path,index=False)
            logging.info(f"Results_df save to path:{self.config.tuned_report_path}")
            return results_df
        except Exception as e:
             raise CustomException(e, sys)
































