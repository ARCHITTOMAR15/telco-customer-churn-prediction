
import os
import sys
from dataclasses import dataclass

import pandas as pd 
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class FeatureSelectionConfig:
    train_data_path = os.path.join(
        "data",
        "processed",
        "train.csv"
    )

    report_path = os.path.join(
        "reports",
        "feature_selection_report.csv"
    )

    summary_report_path = os.path.join(
        "reports",
        "feature_selection_summary.csv"
    )

    selected_features_path = os.path.join(
        "artifacts",
        "selected_features.pkl"
    )

    feature_names_path = os.path.join(
        "artifacts",
        "feature_names.pkl"
    )


class FeatureSelection:
    def __init__(self):
        self.config = FeatureSelectionConfig()

    def initiate_feature_selection(self):
        try:
            logging.info("Loading Processesd Training Dataset")
            train_df=pd.read_csv(self.config.train_data_path)

            X=train_df.iloc[:,:-1]
            y=train_df.iloc[:,-1]

            logging.info(f"Original Feature Count: {X.shape[1]}")

            #Variance Tersold

            selector = VarianceThreshold(threshold=0.0)

            selector.fit(X)

            selected_columns = X.columns[selector.get_support()]

            removed_variance_features = X.columns[~selector.get_support()].tolist()

            X = X[selected_columns]

            logging.info(f"Removed {len(removed_variance_features)} low variance features.")

            # corelation filter

            corr_matrix=X.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            columns_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

            removed_correlation_features = columns_to_drop.copy()

            X = X.drop(columns=columns_to_drop)

            logging.info(f"Removed {len(columns_to_drop)} highly correlated features.")

            # Step 3 : Mutual Information

            mi_scores= mutual_info_classif(X,y,random_state=42)
            mi_df = pd.DataFrame({"Feature": X.columns,"MutualInformation": mi_scores})

            mi_df["Rank"] = (mi_df["MutualInformation"].rank(ascending=False, method="dense").astype(int))

            mi_df = mi_df.sort_values(by="MutualInformation",ascending=False).reset_index(drop=True)

            #report save 


            os.makedirs("reports", exist_ok=True)

            mi_df.to_csv(
                self.config.report_path,
                index=False
            )

            logging.info("Feature selection report saved.")

            # --------------------------------------------------
            # Save Feature List
            # --------------------------------------------------

            selected_feature_list = mi_df["Feature"].tolist()

            feature_names = X.columns.tolist()

            save_object(file_path=self.config.feature_names_path,obj=feature_names)

            logging.info("Feature names saved successfully.")

            os.makedirs("artifacts", exist_ok=True)

            save_object(file_path=self.config.selected_features_path, obj=selected_feature_list)

            logging.info("Selected feature list saved.")


            summary_df = pd.DataFrame({"Stage": ["Original Features","Removed by Variance Threshold","Removed by Correlation","Final Selected Features"],
                       "Count": [train_df.shape[1] - 1,len(removed_variance_features),len(removed_correlation_features),len(selected_feature_list)]})

            summary_df.to_csv(self.config.summary_report_path,index=False)

            logging.info("Feature selection summary report saved.")

            return mi_df, summary_df

        except Exception as e:
            raise CustomException(e, sys)
