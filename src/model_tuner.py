

import sys 
import os 
import time 
import joblib
import pandas as pd 

from dataclasses import dataclass

from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report)

from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTunerConfig:


    train_data_path= os.path.join("data","final","train_selected.csv")
    test_data_path=  os.path.join("data","final","test_selected.csv")

    tuned_model_dir=os.path.join("artifacts","tuned_models")
    tuning_report_path=os.path.join("reports","hyperparameter_tuning_report.csv")


class ModelTuner:
    def __init__(self):

        self.config= ModelTunerConfig()

        os.makedirs(self.config.tuned_model_dir,exist_ok=True)
        os.makedirs(os.path.dirname(self.config.tuning_report_path),exist_ok=True)

        #stratifided k fold
        self.cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    #load data
    def load_data(self):
        try:
            train_df=pd.read_csv(self.config.train_data_path)
            test_df=pd.read_csv(self.config.test_data_path)

        #split into traget and feature 
            X_train=train_df.drop(columns=["Churn"])
            y_train=train_df["Churn"]
            X_test=test_df.drop(columns=["Churn"])
            y_test=test_df["Churn"]

            logging.info(f" Training Data Shape{X_train.shape}")
            logging.info(f" Testing Data Shape{X_test.shape}")
            logging.info(f" Target Data Shape{y_train.shape}")
            logging.info(f" Target Data Shape{y_test.shape}")

            return(X_train,X_test,y_train,y_test)

        except Exception as e:
            raise CustomException(e, sys)


            #apply smote 

    def smote(self,X_train,y_train):
        try:
            logging.info("Applying SMOTE to training data")

            smote=SMOTE(random_state=42)

            X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

            #before smote
            logging.info(f"Before applying smote{y_train.value_counts().to_dict()}")
            logging.info(f"After Applying Smote{y_train_smote.value_counts().to_dict()}")

            return(X_train_smote,y_train_smote)
        except Exception as e:
            raise CustomException(e, sys)

        #catboost hyperparameter tuning 

    def tune_catboost(self,X_train_smote,y_train_smote):
        try:
            logging.info("Catboost Hyperparameter Tuning Started")

            catboost_model=CatBoostClassifier(random_state=42,verbose=0,thread_count=2)

            param_grid={"iterations":[100,200,300],
                        "depth":[3,4,5,6,7,8],
                        "learning_rate":[0.03,0.05,0.1],
                        "l2_leaf_reg":[1,3,5],
                        "random_strength":[0,1,2]}

        #randomized search

            random_search=RandomizedSearchCV(estimator=catboost_model,param_distributions=param_grid,n_iter=10,
                                             scoring="roc_auc",cv=self.cv,random_state=40,n_jobs=1,verbose=10)

            start_time=time.time()
            print ("Starting Randomized Searc CV")

            random_search.fit(X_train_smote,y_train_smote)

            tune_time=time.time()-start_time

            print("Randomized Search CV Completed")

            logging.info(f"Catboost tuning completed in {tune_time:.2f} seconds")

            #best parameters
            best_params=(random_search.best_params_)
            best_cv_score=(random_search.best_score_)

            logging.info(f"Best catboostparameter{best_params}")
            logging.info(f"Best CV ROC AUC: "f"{best_cv_score:.4f}")

            return(random_search.best_estimator_,best_params,best_cv_score,tune_time)

        except Exception as e:
            raise CustomException(e, sys)


    #evaluste catboost performance 
    def evaluate_tuned_catboost(self,best_model,X_test,y_test,best_params,best_cv_score,tuning_time):
        try:
            logging.info("Evaluation Started")

            y_prob=best_model.predict_proba(X_test)[:,1]
            y_pred= (y_prob>=0.5).astype(int)

            #metrics
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test,y_pred,zero_division=0)
            f1 = f1_score(y_test,y_pred, zero_division=0)
            roc_auc = roc_auc_score( y_test, y_prob)



            #creating result dictionary 
            results={"Model1":"Catboost Tuned",
                     "Accuracy": round(accuracy,4),
                     "precision":round(precision,4),
                     "recall":round(recall,4),
                     "f1_score":round(f1,4),
                     "roc_auc":round(roc_auc,4),
                     "CV ROC AUC": round(best_cv_score,4),
                     "Tuning Time (Seconds)": round(tuning_time,2),
                     "Best Parameters": str(best_params)}


            #saving tuned model 

            model_path = os.path.join(self.config.tuned_model_dir,"catboost_tuned.pkl")

            joblib.dump(best_model,model_path)

            logging.info( f"Tuned CatBoost model saved to: "f"{model_path}")


          #saving result report '

            results_df=pd.DataFrame([results])
            results_df.to_csv(self.config.tuning_report_path,index=False)

            logging.info(f"CatBoost tuning report saved to: "f"{self.config.tuning_report_path}")

            return results_df

        except Exception as e:
             raise CustomException(e, sys)
